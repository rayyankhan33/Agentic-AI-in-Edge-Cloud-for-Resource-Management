"""Run every baseline scheduler on identical trace-driven scenarios.

    python scripts/run_baselines.py              # writes results/baseline_results.csv
    python scripts/run_baselines.py --quick      # 1 segment, short DRL training

Scenarios: three disjoint 3000-task segments of the Alibaba window under
normal conditions, plus the same segments with a WAN outage (bandwidth x0.15
for 150 s). The DRL baseline is trained on segments that do not overlap the
test segments.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ecagents.agents.orchestrator import OrchestrationAgent  # noqa: E402
from ecagents.experiment import make_scenario  # noqa: E402
from ecagents.schedulers.drl import DQNScheduler, featurize  # noqa: E402
from ecagents.schedulers.heuristics import (AllCloud, GreedyLatency, GreedyWeighted, LocalEdge,  # noqa: E402
                                            RandomPlacement, RoundRobin)
from ecagents.schedulers.milp import BatchMILP  # noqa: E402
from ecagents.sim.simulator import PlanView, Simulator  # noqa: E402

TEST_OFFSETS = [0, 20_000, 40_000]
TRAIN_OFFSETS = [5_000, 8_000, 11_000, 14_000, 26_000, 29_000, 32_000, 35_000]
N = 3000
DROP = [(150, 300, 0.15)]


def train_drl(episodes: int) -> DQNScheduler:
    sc = make_scenario(N, offset=TRAIN_OFFSETS[0])
    sim = Simulator(sc.cfg, sc.nodes, sc.network(), sc.objective, scheduler=None)
    n_in = len(featurize(sc.tasks[0], PlanView(sim)))
    agent = DQNScheduler(n_in, len(sc.nodes), seed=sc.seed)
    for ep in range(episodes):
        off = TRAIN_OFFSETS[ep % len(TRAIN_OFFSETS)]
        sc = make_scenario(N, offset=off, drops=DROP if ep % 2 else None, seed=ep)
        agent.training = True
        m, sim = sc.run(agent)
        loss = agent.learn_from_episode(sim.records)
        print(f"  drl ep {ep:2d} offset {off:5d} J {m.mean_weighted_cost:7.3f} miss {m.deadline_miss_rate:.3f} "
              f"eps {agent.eps:.2f} loss {loss:.3f}", flush=True)
    agent.training = False
    agent.save(ROOT / "results" / "drl_ddqn.pt")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=24)
    args = ap.parse_args()
    offsets = TEST_OFFSETS[:1] if args.quick else TEST_OFFSETS
    (ROOT / "results").mkdir(exist_ok=True)

    t0 = time.time()
    drl = train_drl(4 if args.quick else args.episodes)
    print(f"DRL training took {time.time() - t0:.0f}s")

    rows = []
    for condition, drops in (("normal", None), ("wan-outage", DROP)):
        for off in offsets:
            sc = make_scenario(N, offset=off, drops=drops)
            runs = [(LocalEdge(), None), (AllCloud(), None), (RandomPlacement(sc.seed), None),
                    (RoundRobin(), None), (GreedyLatency(), None), (GreedyWeighted(), None),
                    (BatchMILP(), None), (drl, None),
                    (GreedyWeighted(), OrchestrationAgent())]
            for sched, orch in runs:
                m, _ = sc.run(sched, orch)
                rows.append({"condition": condition, "segment": off, **m.model_dump()})
                print(f"{condition:10s} seg {off:5d} {m.scheduler:22s} J {m.mean_weighted_cost:8.3f} "
                      f"lat {m.mean_latency_s:8.3f} p95 {m.p95_latency_s:8.3f} miss {m.deadline_miss_rate:.3f} "
                      f"E {m.total_energy_kj:6.1f}kJ ${m.total_cost_usd:.3f} edge {m.edge_share:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "baseline_results.csv", index=False)
    cols = ["mean_weighted_cost", "mean_latency_s", "p95_latency_s", "deadline_miss_rate",
            "total_energy_kj", "total_cost_usd", "edge_share", "decision_ms_per_task", "migrations"]
    summary = df.groupby(["condition", "scheduler"])[cols].mean().round(4)
    summary.to_csv(ROOT / "results" / "baseline_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
