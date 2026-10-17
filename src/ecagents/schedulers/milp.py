"""Optimisation baseline: one mixed-integer program per decision epoch.

For the batch B of pending tasks and node set N:

    min  sum_{i,n} c_in x_in + lam * sum_n o_n / cores_n
    s.t. sum_n x_in = 1                                  for every task i
         o_n >= sum_i p_in x_in - cores_n * epoch_s      for every node n
         x_in in {0,1},  o_n >= 0

c_in is the estimated weighted cost of task i on node n given the load
already in the system, p_in its run time on n, and o_n the work that
overflows what node n can finish within the epoch (a linear congestion
term coupling tasks in the same batch).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..contracts import PlacementDecision
from ..sim.network import exec_s
from .heuristics import Scheduler


class BatchMILP(Scheduler):
    name = "batch-milp"

    def __init__(self, congestion_weight: float | None = None, time_limit_s: float = 2.0):
        super().__init__()
        self.lam = congestion_weight
        self.time_limit = time_limit_s

    def decide_batch(self, tasks, view):
        nodes = view.nodes
        T, N = len(tasks), len(nodes)
        c = np.array([[view.estimate(t, n.node_id).weighted_cost for n in nodes] for t in tasks])
        p = np.array([[exec_s(t, n) for n in nodes] for t in tasks])
        w_lat, t_ref = view.sim.objective.w[0], view.sim.objective.ref[0]
        lam = self.lam if self.lam is not None else w_lat / t_ref * max(T / N, 1.0)

        n_x = T * N
        obj = np.concatenate([c.ravel(), [lam / n.cores for n in nodes]])
        rows, lb, ub = [], [], []
        for i in range(T):                       # assignment
            r = np.zeros(n_x + N)
            r[i * N:(i + 1) * N] = 1
            rows.append(r); lb.append(1); ub.append(1)
        for j, node in enumerate(nodes):         # overflow
            r = np.zeros(n_x + N)
            r[j:n_x:N] = p[:, j]
            r[n_x + j] = -1
            rows.append(r); lb.append(-np.inf); ub.append(node.cores * view.sim.epoch_s)
        integrality = np.concatenate([np.ones(n_x), np.zeros(N)])
        bounds = Bounds(np.zeros(n_x + N), np.concatenate([np.ones(n_x), np.full(N, np.inf)]))
        res = milp(obj, constraints=LinearConstraint(np.array(rows), lb, ub),
                   integrality=integrality, bounds=bounds,
                   options={"time_limit": self.time_limit, "disp": False})
        if res.x is None:                        # solver failed: fall back to per-task argmin
            choice = c.argmin(axis=1)
        else:
            choice = res.x[:n_x].reshape(T, N).argmax(axis=1)
        out = []
        for task, j in zip(tasks, choice):
            view.commit(task, int(j))
            out.append(PlacementDecision(task_id=task.task_id, node_id=int(j), source="milp"))
        return out
