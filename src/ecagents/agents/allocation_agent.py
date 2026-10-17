"""Resource Allocation (Offloading) Agent.

Per decision epoch the agent:
  1. calls the ``estimate_placement`` tool for every pending task and node,
  2. calls the ``get_forecast`` tool (arrival rate, WAN bandwidth),
  3. escalates only the *ambiguous* tasks (top-2 candidates within a margin)
     to the LLM, sending a compact JSON of their top-k candidates,
  4. validates the LLM's answer (known task, candidate node, schema) and
     falls back to the weighted-greedy choice for anything invalid.

Clear-cut tasks never reach the LLM, which bounds call volume and latency.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..contracts import PlacementDecision
from ..schedulers.heuristics import Scheduler
from .forecast_agent import ForecastingAgent
from .llm import LLMClient, to_prompt_json

SYSTEM = """You are the resource allocation agent of an edge-cloud platform.
For each task you get candidate nodes with tool-estimated latency (s), energy (J), cost (USD),
weighted cost J (lower is better; it already includes a deadline-miss penalty) and whether the
deadline is met. You also get a short-term forecast of arrivals and WAN bandwidth.
Choose one candidate node per task. Prefer lower weighted cost and meeting the deadline. When the
forecast says load is rising, keep edge capacity for tight-deadline tasks and send delay-tolerant
work to the cloud; when WAN bandwidth is forecast to drop, avoid sending large inputs to the cloud.
Respect preferred_location when it is edge or cloud. Only use node_ids that appear in the candidates."""


class LLMChoice(BaseModel):
    task_id: int
    node_id: int
    reason: str = Field(default="", max_length=200)


class LLMBatch(BaseModel):
    decisions: list[LLMChoice]


class AllocationAgent(Scheduler):
    name = "agent-llm"

    def __init__(self, llm: LLMClient | None = None, forecaster: ForecastingAgent | None = None,
                 margin: float = 0.15, top_k: int = 3, max_tasks_per_call: int = 12,
                 max_calls: int | None = None):
        super().__init__()
        self.llm = llm or LLMClient()
        self.forecaster = forecaster or ForecastingAgent()
        self.margin, self.top_k = margin, top_k
        self.max_tasks_per_call = max_tasks_per_call
        self.max_calls = max_calls
        self.n_escalated = self.n_llm_ok = self.n_fallback = 0
        self.n_calls = 0
        self.llm_wall_s = 0.0
        self.trace: list[dict] = []
        self._seen = 0

    def decide_batch(self, tasks, view):
        state = view.state()
        self.forecaster.observe(view.t, len(tasks), state.wan_bandwidth_mbps)
        fc = self.forecaster.forecast()

        out, ambiguous = {}, []
        for task in tasks:
            cands = sorted(view.estimate_all(task), key=lambda c: c.weighted_cost)
            if task.preferred_location in ("edge", "cloud"):
                pref = [c for c in cands if c.tier == task.preferred_location]
                cands = pref or cands
            best = cands[0]
            close = len(cands) > 1 and cands[1].weighted_cost - best.weighted_cost <= self.margin * best.weighted_cost
            budget_left = self.max_calls is None or self.n_calls < self.max_calls
            if close and budget_left and self.llm.available and len(ambiguous) < self.max_tasks_per_call:
                ambiguous.append((task, cands[: self.top_k]))
            else:
                view.commit(task, best.node_id)
                out[task.task_id] = PlacementDecision(task_id=task.task_id, node_id=best.node_id,
                                                      source="heuristic")
        if ambiguous:
            self._ask_llm(ambiguous, fc, state, view, out)
        self.stats = {"llm_calls": self.n_calls,
                      "fallback_rate": self.n_fallback / max(self.n_escalated, 1),
                      "escalated": self.n_escalated, "llm_wall_s": self.llm_wall_s}
        return [out[t.task_id] for t in tasks]

    def _ask_llm(self, ambiguous, fc, state, view, out):
        payload = {
            "forecast": {"horizon_s": fc.horizon_s, "arrival_rate": round(fc.arrival_rate, 2),
                         "current_arrival_rate": round(state.recent_arrival_rate, 2),
                         "wan_mbps_forecast": round(fc.wan_bandwidth_mbps, 1),
                         "wan_mbps_now": round(state.wan_bandwidth_mbps, 1)},
            "tasks": [{
                "task_id": t.task_id, "type": t.task_type.value, "deadline_s": round(t.deadline_s, 3),
                "input_mb": round(t.input_mb, 2), "preferred_location": t.preferred_location,
                "candidates": [{"node_id": c.node_id, "tier": c.tier, "latency_s": round(c.est_latency_s, 3),
                                "energy_j": round(c.est_energy_j, 2), "cost_usd": f"{c.est_cost_usd:.2e}",
                                "J": round(c.weighted_cost, 3), "meets_deadline": c.meets_deadline}
                               for c in cands]} for t, cands in ambiguous],
        }
        self.n_escalated += len(ambiguous)
        t0 = time.perf_counter()
        resp = self.llm.generate_json(to_prompt_json(payload), LLMBatch, system=SYSTEM)
        call_s = time.perf_counter() - t0
        self.llm_wall_s += call_s
        self.n_calls += 1
        chosen = {d.task_id: d for d in resp.decisions} if resp else {}
        for task, cands in ambiguous:
            allowed = {c.node_id for c in cands}
            d = chosen.get(task.task_id)
            if d is not None and d.node_id in allowed:
                node_id, source, reason = d.node_id, "llm", d.reason
                self.n_llm_ok += 1
            else:
                node_id, source, reason = cands[0].node_id, "fallback", "invalid or missing LLM output"
                self.n_fallback += 1
            view.commit(task, node_id)
            out[task.task_id] = PlacementDecision(task_id=task.task_id, node_id=node_id,
                                                  rationale=reason, source=source)
            self.trace.append({"t": round(view.t, 2), "task_id": task.task_id, "node_id": node_id,
                               "source": source, "reason": reason, "call_s": round(call_s, 3),
                               "greedy_choice": cands[0].node_id})
