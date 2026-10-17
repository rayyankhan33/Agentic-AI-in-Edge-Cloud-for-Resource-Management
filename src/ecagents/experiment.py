"""Builds identical scenarios for every scheduler and runs them."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import REPO_ROOT, build_nodes, load_config
from .contracts import EpisodeMetrics, NodeSpec, TaskSpec
from .sim.network import Network, Objective
from .sim.simulator import Simulator
from .trace import to_tasks


@dataclass
class Scenario:
    cfg: dict
    nodes: list[NodeSpec]
    tasks: list[TaskSpec]
    objective: Objective
    drops: list | None
    seed: int

    def network(self) -> Network:
        horizon = self.tasks[-1].arrival_s + 600
        return Network(self.cfg, horizon, seed=self.seed, drops=self.drops)

    def run(self, scheduler, orchestrator=None) -> tuple[EpisodeMetrics, Simulator]:
        sim = Simulator(self.cfg, self.nodes, self.network(), self.objective, scheduler, orchestrator)
        return sim.run(self.tasks), sim


def make_scenario(n_tasks: int = 3000, offset: int = 0, cfg: dict | None = None, rate: float | None = None,
                  drops: list | None = None, seed: int | None = None) -> Scenario:
    cfg = cfg or load_config()
    seed = cfg["seed"] if seed is None else seed
    nodes = build_nodes(cfg)
    tc = cfg["trace"]
    df = pd.read_csv(REPO_ROOT / tc["processed_csv"])
    edges = [n for n in nodes if n.tier == "edge"]
    tasks = to_tasks(df, len(edges), rate or tc["target_rate"], tc["target_load_ghz"], tc["median_input_mb"],
                     [e.freq_ghz for e in edges], limit=n_tasks, offset=offset)
    net = Network(cfg, tasks[-1].arrival_s + 600, seed=seed, drops=drops)
    objective = Objective.calibrate(cfg, tasks, nodes, net)
    return Scenario(cfg, nodes, tasks, objective, drops, seed)
