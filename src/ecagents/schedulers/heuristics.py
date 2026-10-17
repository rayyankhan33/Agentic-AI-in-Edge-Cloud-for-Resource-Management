"""Non-agentic baseline schedulers."""
from __future__ import annotations

import numpy as np

from ..contracts import PlacementDecision, TaskSpec


def allowed(task: TaskSpec, cands: list) -> list:
    """Honour a hard edge/cloud preference from the user's request when one exists."""
    if task.preferred_location in ("edge", "cloud"):
        keep = [c for c in cands if c.tier == task.preferred_location]
        return keep or cands
    return cands


class Scheduler:
    """Places each task of a batch in arrival order, committing as it goes."""

    name = "base"

    def __init__(self):
        self.stats: dict = {}

    def decide(self, task: TaskSpec, view) -> int:
        raise NotImplementedError

    def decide_batch(self, tasks: list[TaskSpec], view) -> list[PlacementDecision]:
        out = []
        for task in tasks:
            node_id = self.decide(task, view)
            view.commit(task, node_id)
            out.append(PlacementDecision(task_id=task.task_id, node_id=node_id, source="heuristic"))
        return out


class LocalEdge(Scheduler):
    """Run every task on the edge the device is attached to (no offloading)."""

    name = "local-edge"

    def decide(self, task, view):
        return task.source_edge


class AllCloud(Scheduler):
    name = "all-cloud"

    def decide(self, task, view):
        return view.nodes[-1].node_id


class RandomPlacement(Scheduler):
    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def decide(self, task, view):
        return int(self.rng.integers(len(view.nodes)))


class RoundRobin(Scheduler):
    name = "round-robin"

    def __init__(self):
        super().__init__()
        self.i = 0

    def decide(self, task, view):
        self.i = (self.i + 1) % len(view.nodes)
        return self.i


class GreedyLatency(Scheduler):
    """Earliest estimated completion (classic EFT list scheduling)."""

    name = "greedy-latency"

    def decide(self, task, view):
        return min(allowed(task, view.estimate_all(task)), key=lambda c: c.est_latency_s).node_id


class GreedyWeighted(Scheduler):
    """Minimum estimated weighted cost J (same objective the agents optimise)."""

    name = "greedy-weighted"

    def decide(self, task, view):
        return min(allowed(task, view.estimate_all(task)), key=lambda c: c.weighted_cost).node_id
