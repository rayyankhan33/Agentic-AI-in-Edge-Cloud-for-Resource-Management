"""DRL baseline in the spirit of the base paper (TF-DDRL, Wang et al., IEEE TSC 2025).

Phase 1 ships a single-learner Double DQN with the base paper's MLP sizes
([256, 256, 128], ReLU, lr 1e-3, gamma 0.99) and its reward: the negative
weighted cost of the task once it has finished (or a penalty if it misses
its deadline, which the objective already adds). The distributed
IMPALA/V-trace learner, prioritised replay and the Gated Transformer-XL
encoder of TF-DDRL are planned for Phase 2.
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..contracts import PlacementDecision, TaskSpec
from .heuristics import Scheduler


def featurize(task: TaskSpec, view) -> np.ndarray:
    """Task features + per-node load features (no cost-model estimates)."""
    t = view.t
    n_edges = len(view.nodes) - 1
    feats = [task.work_gcycles / 2.0, task.input_mb / 5.0, task.deadline_s / 5.0,
             view.sim.net.wan_mbps(t) / 40.0]
    feats += list(np.eye(n_edges)[task.source_edge])
    for node, free in zip(view.nodes, view.core_free):
        waits = np.maximum(np.array(free) - t, 0)
        transfer = view.sim.net.transfer_s(task.input_mb, task.source_edge, node, t)
        feats += [waits.min() / 5.0, waits.mean() / 5.0, node.freq_ghz / 3.0,
                  node.cores / 64.0, float(node.tier == "cloud"), transfer]
    return np.asarray(feats, dtype=np.float32)


class QNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_out))

    def forward(self, x):
        return self.net(x)


class DQNScheduler(Scheduler):
    name = "drl-ddqn"

    def __init__(self, n_in: int, n_actions: int, seed: int = 0, lr: float = 1e-3, gamma: float = 0.99,
                 eps: float = 1.0, eps_min: float = 0.05, eps_decay: float = 0.9995,
                 buffer: int = 50_000, batch: int = 128, target_sync: int = 500):
        super().__init__()
        torch.manual_seed(seed)
        random.seed(seed)
        self.q = QNet(n_in, n_actions)
        self.target = QNet(n_in, n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.eps, self.eps_min, self.eps_decay = gamma, eps, eps_min, eps_decay
        self.buffer = deque(maxlen=buffer)
        self.batch, self.target_sync = batch, target_sync
        self.n_actions = n_actions
        self.training = True
        self.trajectory: list[tuple[int, np.ndarray, int]] = []   # (task_id, state, action)
        self.steps = 0

    def decide(self, task, view):
        s = featurize(task, view)
        if self.training and random.random() < self.eps:
            a = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                a = int(self.q(torch.from_numpy(s)).argmax())
        if self.training:
            self.trajectory.append((task.task_id, s, a))
        return a

    def decide_batch(self, tasks, view):
        out = []
        for task in tasks:
            a = self.decide(task, view)
            view.commit(task, a)
            out.append(PlacementDecision(task_id=task.task_id, node_id=a, source="drl"))
        return out

    def learn_from_episode(self, records: dict, updates_per_step: float = 0.5) -> float:
        """Turn the finished episode into transitions (reward = -J) and train."""
        traj = self.trajectory
        for k, (tid, s, a) in enumerate(traj):
            r = -records[tid].weighted_cost
            s2 = traj[k + 1][1] if k + 1 < len(traj) else s
            self.buffer.append((s, a, r, s2, k + 1 == len(traj)))
        self.trajectory = []
        losses = []
        for _ in range(int(len(traj) * updates_per_step)):
            losses.append(self._update())
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return float(np.mean(losses)) if losses else 0.0

    def _update(self) -> float:
        if len(self.buffer) < self.batch:
            return 0.0
        s, a, r, s2, d = zip(*random.sample(self.buffer, self.batch))
        s, s2 = torch.from_numpy(np.stack(s)), torch.from_numpy(np.stack(s2))
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)                 # Double DQN target
            y = r + self.gamma * (1 - d) * self.target(s2).gather(1, a2).squeeze(1)
        loss = nn.functional.smooth_l1_loss(self.q(s).gather(1, a).squeeze(1), y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss)

    def save(self, path: str | Path) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.q.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.q.state_dict())
