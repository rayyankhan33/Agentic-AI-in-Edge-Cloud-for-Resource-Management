"""Observation & Orchestration Agent: closed-loop monitoring and task migration.

Every observation period it reads the system state (``get_system_state``
tool), finds overloaded nodes, and for tasks still waiting there asks the
estimate tool whether another node would now finish them sooner after paying
the migration transfer. Phase 1 uses this rule-based policy; Phase 2 adds
LLM reasoning over the same tools for non-obvious cases (e.g. WAN drops).
"""
from __future__ import annotations

from ..contracts import MigrationAction
from ..sim.network import exec_s


class OrchestrationAgent:
    name = "orchestrator-rules"

    def __init__(self, backlog_threshold_s: float = 2.0, gain_threshold: float = 0.2,
                 max_moves_per_round: int = 20):
        self.backlog_threshold_s = backlog_threshold_s
        self.gain_threshold = gain_threshold
        self.max_moves = max_moves_per_round
        self.log: list[dict] = []

    def observe(self, view, sim) -> list[MigrationAction]:
        state = view.state()
        hot = [n for n in state.nodes if n.backlog_s > self.backlog_threshold_s]
        actions = []
        for ns in sorted(hot, key=lambda n: -n.backlog_s):
            rt = sim.runtime[ns.node_id]
            src = sim.nodes[ns.node_id]
            # Tasks at the back of the queue gain the most from moving.
            for pos, tid in reversed(list(enumerate(rt.waiting))):
                if len(actions) >= self.max_moves:
                    return actions
                task = sim.records[tid].task
                ahead = sum(exec_s(sim.records[t].task, src) for t in list(rt.waiting)[:pos])
                stay = view.t + ahead / src.cores + exec_s(task, src)
                best, best_finish = None, stay
                for node in sim.nodes:
                    if node.node_id == ns.node_id:
                        continue
                    if task.preferred_location in ("edge", "cloud") and node.tier != task.preferred_location:
                        continue
                    hop = sim.net.hop_s(task.input_mb, src, node, view.t)
                    finish = max(view.core_free[node.node_id][0], view.t + hop) + exec_s(task, node)
                    if finish < best_finish:
                        best, best_finish = node, finish
                if best is not None and (stay - best_finish) > self.gain_threshold * max(stay - view.t, 1e-6):
                    view.commit(task, best.node_id)
                    actions.append(MigrationAction(task_id=tid, from_node=ns.node_id, to_node=best.node_id,
                                                   reason=f"backlog {ns.backlog_s:.1f}s, finish "
                                                          f"{stay - view.t:.2f}s -> {best_finish - view.t:.2f}s"))
        self.log.extend(a.model_dump() for a in actions)
        return actions
