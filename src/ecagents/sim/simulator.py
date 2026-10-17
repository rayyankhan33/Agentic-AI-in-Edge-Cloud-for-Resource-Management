"""Trace-driven discrete-event simulator for edge-cloud task offloading.

Each node is a pool of identical cores with one FCFS queue. Tasks arrive from
the trace, wait for the next decision epoch, are placed by a scheduler, travel
over the network, queue, run and send their result back. An optional
orchestrator observes the system periodically and may migrate queued tasks.

All schedulers see the same workload, network trace and decision epochs, so
their metrics are directly comparable.
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..contracts import (CandidateEstimate, EpisodeMetrics, MigrationAction, NodeSpec,
                         NodeState, PlacementDecision, SystemState, TaskSpec)
from .network import Network, Objective, exec_energy_j, exec_s, money_usd

ARRIVAL, EPOCH, READY, FINISH, OBSERVE = range(5)


@dataclass
class TaskRecord:
    task: TaskSpec
    node_id: int = -1
    decided_s: float = 0.0
    ready_s: float = 0.0
    start_s: float = 0.0
    finish_s: float = 0.0
    latency_s: float = 0.0
    energy_j: float = 0.0
    cost_usd: float = 0.0
    weighted_cost: float = 0.0
    missed: bool = False
    migrated: int = 0
    source: str = ""


@dataclass
class NodeRuntime:
    spec: NodeSpec
    running: dict = field(default_factory=dict)       # task_id -> finish time
    waiting: deque = field(default_factory=deque)     # task ids
    in_transit: dict = field(default_factory=dict)    # task_id -> ready time


class PlanView:
    """Scheduler-facing view: per-node core availability plus a cost estimator.

    ``commit`` records a provisional placement so later tasks in the same
    batch see the extra load. This is the ``estimate_placement`` tool.
    """

    def __init__(self, sim: "Simulator"):
        self.sim = sim
        self.t = sim.now
        self.nodes = sim.nodes
        self.core_free: list[list[float]] = []
        for rt in sim.runtime:
            free = sorted(rt.running.values()) + [self.t] * (rt.spec.cores - len(rt.running))
            heapq.heapify(free)
            pending = [(sim.records[tid].ready_s, tid) for tid in rt.in_transit]
            queue = [(self.t, tid) for tid in rt.waiting] + sorted(pending)
            for ready, tid in queue:
                start = max(heapq.heappop(free), ready)
                heapq.heappush(free, start + exec_s(sim.records[tid].task, rt.spec))
            self.core_free.append(free)

    def state(self) -> SystemState:
        return self.sim.system_state()

    def estimate(self, task: TaskSpec, node_id: int) -> CandidateEstimate:
        node = self.nodes[node_id]
        net, cfg = self.sim.net, self.sim.cfg
        transfer = net.transfer_s(task.input_mb, task.source_edge, node, self.t)
        start = max(self.core_free[node_id][0], self.t + transfer)
        ex = exec_s(task, node)
        back = net.transfer_s(task.output_mb, task.source_edge, node, self.t)
        latency = (start - task.arrival_s) + ex + back
        energy = (exec_energy_j(task, node)
                  + net.tx_energy_j(task.input_mb + task.output_mb, task.source_edge, node, self.t))
        cost = money_usd(task, node, energy, cfg)
        meets = latency <= task.deadline_s
        return CandidateEstimate(
            node_id=node_id, tier=node.tier, est_latency_s=latency, est_energy_j=energy,
            est_cost_usd=cost, weighted_cost=self.sim.objective(latency, energy, cost, not meets),
            meets_deadline=meets)

    def estimate_all(self, task: TaskSpec) -> list[CandidateEstimate]:
        return [self.estimate(task, n.node_id) for n in self.nodes]

    def commit(self, task: TaskSpec, node_id: int) -> None:
        node = self.nodes[node_id]
        transfer = self.sim.net.transfer_s(task.input_mb, task.source_edge, node, self.t)
        start = max(heapq.heappop(self.core_free[node_id]), self.t + transfer)
        heapq.heappush(self.core_free[node_id], start + exec_s(task, node))


class Simulator:
    def __init__(self, cfg: dict, nodes: list[NodeSpec], net: Network, objective: Objective,
                 scheduler, orchestrator=None, epoch_s: float | None = None):
        self.cfg = cfg
        self.nodes = nodes
        self.net = net
        self.objective = objective
        self.scheduler = scheduler
        self.orchestrator = orchestrator
        self.epoch_s = epoch_s if epoch_s is not None else cfg["sim"]["epoch_s"]
        self.obs_period = cfg["sim"]["orchestrator_period_s"]
        self.runtime = [NodeRuntime(n) for n in nodes]
        self.records: dict[int, TaskRecord] = {}
        self.pending: list[TaskSpec] = []
        self.now = 0.0
        self.events: list = []
        self._seq = itertools.count()
        self.decision_wall_s = 0.0
        self.migrations = 0
        self.arrival_log: deque = deque()

    # ------------------------------------------------------------------ events
    def _push(self, t: float, kind: int, payload=None) -> None:
        heapq.heappush(self.events, (t, next(self._seq), kind, payload))

    def run(self, tasks: list[TaskSpec]) -> EpisodeMetrics:
        for task in tasks:
            self.records[task.task_id] = TaskRecord(task)
            self._push(task.arrival_s, ARRIVAL, task)
        last = tasks[-1].arrival_s if tasks else 0.0
        t = self.epoch_s
        while t <= last + self.epoch_s:
            self._push(t, EPOCH)
            t += self.epoch_s
        if self.orchestrator is not None:
            t = self.obs_period
            while t <= last:
                self._push(t, OBSERVE)
                t += self.obs_period

        while self.events:
            self.now, _, kind, payload = heapq.heappop(self.events)
            if kind == ARRIVAL:
                self.pending.append(payload)
                self.arrival_log.append(self.now)
            elif kind == EPOCH:
                self._decide()
            elif kind == READY:
                self._ready(*payload)
            elif kind == FINISH:
                self._finish(*payload)
            elif kind == OBSERVE:
                self._observe()
        return self.metrics()

    def _decide(self) -> None:
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        view = PlanView(self)
        t0 = time.perf_counter()
        decisions: list[PlacementDecision] = self.scheduler.decide_batch(batch, view)
        self.decision_wall_s += time.perf_counter() - t0
        by_id = {d.task_id: d for d in decisions}
        for task in batch:
            d = by_id[task.task_id]
            self._dispatch(task, d.node_id, d.source)

    def _dispatch(self, task: TaskSpec, node_id: int, source: str) -> None:
        rec = self.records[task.task_id]
        node = self.nodes[node_id]
        rec.node_id, rec.decided_s, rec.source = node_id, self.now, source
        rec.energy_j += self.net.tx_energy_j(task.input_mb + task.output_mb, task.source_edge, node, self.now)
        rec.ready_s = self.now + self.net.transfer_s(task.input_mb, task.source_edge, node, self.now)
        self.runtime[node_id].in_transit[task.task_id] = rec.ready_s
        self._push(rec.ready_s, READY, (node_id, task.task_id))

    def _ready(self, node_id: int, task_id: int) -> None:
        rt = self.runtime[node_id]
        if task_id not in rt.in_transit:        # superseded by a migration
            return
        del rt.in_transit[task_id]
        rt.waiting.append(task_id)
        self._try_start(node_id)

    def _try_start(self, node_id: int) -> None:
        rt = self.runtime[node_id]
        while rt.waiting and len(rt.running) < rt.spec.cores:
            tid = rt.waiting.popleft()
            rec = self.records[tid]
            rec.start_s = self.now
            finish = self.now + exec_s(rec.task, rt.spec)
            rt.running[tid] = finish
            self._push(finish, FINISH, (node_id, tid))

    def _finish(self, node_id: int, task_id: int) -> None:
        rt = self.runtime[node_id]
        del rt.running[task_id]
        rec = self.records[task_id]
        task, node = rec.task, rt.spec
        back = self.net.transfer_s(task.output_mb, task.source_edge, node, self.now)
        rec.finish_s = self.now + back
        rec.latency_s = rec.finish_s - task.arrival_s
        e_exec = exec_energy_j(task, node)
        rec.energy_j += e_exec
        rec.cost_usd = money_usd(task, node, rec.energy_j, self.cfg)
        rec.missed = rec.latency_s > task.deadline_s
        rec.weighted_cost = self.objective(rec.latency_s, rec.energy_j, rec.cost_usd, rec.missed)
        self._try_start(node_id)

    def _observe(self) -> None:
        actions: list[MigrationAction] = self.orchestrator.observe(PlanView(self), self)
        for a in actions:
            self.migrate(a)

    # --------------------------------------------------------------- actions
    def migrate(self, action: MigrationAction) -> bool:
        """Move a task that is still queued (not running) to another node."""
        src = self.runtime[action.from_node]
        if action.task_id not in src.waiting:
            return False
        src.waiting.remove(action.task_id)
        rec = self.records[action.task_id]
        a, b = self.nodes[action.from_node], self.nodes[action.to_node]
        hop = self.net.hop_s(rec.task.input_mb, a, b, self.now)
        rec.energy_j += (self.cfg["network"]["p_tx_wan_w"] if "cloud" in (a.tier, b.tier)
                         else self.cfg["network"]["p_tx_edge_w"]) * hop
        rec.node_id, rec.migrated = action.to_node, rec.migrated + 1
        rec.ready_s = self.now + hop
        self.runtime[action.to_node].in_transit[action.task_id] = rec.ready_s
        self._push(rec.ready_s, READY, (action.to_node, action.task_id))
        self.migrations += 1
        return True

    # ----------------------------------------------------------------- state
    def system_state(self, window_s: float = 10.0) -> SystemState:
        while self.arrival_log and self.arrival_log[0] < self.now - window_s:
            self.arrival_log.popleft()
        states = []
        for rt in self.runtime:
            queued = sum(exec_s(self.records[t].task, rt.spec) for t in rt.waiting)
            queued += sum(exec_s(self.records[t].task, rt.spec) for t in rt.in_transit)
            remaining = sum(max(f - self.now, 0) for f in rt.running.values())
            states.append(NodeState(
                node_id=rt.spec.node_id, tier=rt.spec.tier, busy_cores=len(rt.running),
                queue_len=len(rt.waiting) + len(rt.in_transit),
                backlog_s=(queued + remaining) / rt.spec.cores,
                utilisation=len(rt.running) / rt.spec.cores))
        return SystemState(time_s=self.now, nodes=states, wan_bandwidth_mbps=self.net.wan_mbps(self.now),
                           recent_arrival_rate=len(self.arrival_log) / window_s)

    def metrics(self) -> EpisodeMetrics:
        recs = [r for r in self.records.values() if r.finish_s > 0]
        lat = np.array([r.latency_s for r in recs])
        n = len(recs)
        stats = getattr(self.scheduler, "stats", {})
        return EpisodeMetrics(
            scheduler=getattr(self.scheduler, "name", type(self.scheduler).__name__)
            + ("+orch" if self.orchestrator is not None else ""),
            n_tasks=n,
            mean_latency_s=float(lat.mean()),
            p95_latency_s=float(np.percentile(lat, 95)),
            deadline_miss_rate=float(np.mean([r.missed for r in recs])),
            total_energy_kj=float(sum(r.energy_j for r in recs) / 1000),
            total_cost_usd=float(sum(r.cost_usd for r in recs)),
            mean_weighted_cost=float(np.mean([r.weighted_cost for r in recs])),
            edge_share=float(np.mean([self.nodes[r.node_id].tier == "edge" for r in recs])),
            migrations=self.migrations,
            decision_ms_per_task=1000 * self.decision_wall_s / max(n, 1),
            llm_calls=stats.get("llm_calls", 0),
            llm_fallback_rate=stats.get("fallback_rate", 0.0),
        )
