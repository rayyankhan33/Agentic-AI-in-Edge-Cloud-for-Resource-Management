"""End-to-end agentic workflow on a short trace segment (uses Gemini if GEMINI_API_KEY is set).

    python scripts/run_agentic_demo.py --tasks 1200 --max-calls 40 --no-cache

Pipeline per run:
  natural-language requests --NLP agent--> TaskSpecs, merged into the Alibaba task stream
  -> Allocation agent (tools: estimate_placement, get_forecast, get_system_state; LLM for
     ambiguous tasks, validated, greedy fallback)
  -> simulator -> Orchestration agent (periodic observe + migrate)
Writes results/agentic_demo.csv, results/agent_trace.jsonl and results/agentic_demo_llm.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecagents.agents.allocation_agent import AllocationAgent  # noqa: E402
from ecagents.agents.forecast_agent import ForecastingAgent  # noqa: E402
from ecagents.agents.llm import LLMClient  # noqa: E402
from ecagents.agents.nlp_agent import NLPInterpretationAgent, intent_to_tasks  # noqa: E402
from ecagents.agents.orchestrator import OrchestrationAgent  # noqa: E402
from ecagents.experiment import make_scenario  # noqa: E402
from ecagents.schedulers.heuristics import GreedyWeighted  # noqa: E402

REQUESTS = [
    (15.0, 0, "Classify this 4 MB X-ray image with the pneumonia model; the result must come back in under "
              "800 ms and the image must stay on-premise."),
    (30.0, 1, "Run object detection on 30 MB of CCTV footage from gate 3 within 2 seconds, it is urgent."),
    (145.0, 2, "Transcribe 12 voice notes of 1.5 MB each using the speech model, finish within a minute."),
    (160.0, 3, "Retrain the demand forecasting model on the 800 MB sales dataset in the cloud, low priority."),
    (175.0, 1, "Detect helmets in 3 frames (2 MB each) from the construction site camera; high priority, under 1.5 s."),
]


def overhead_adjusted(sim, trace):
    """Add each LLM call's wall-clock latency to the tasks it decided."""
    call_s = {r["task_id"]: r["call_s"] for r in trace}
    lat, miss = [], []
    for rec in sim.records.values():
        extra = call_s.get(rec.task.task_id, 0.0)
        lat.append(rec.latency_s + extra)
        miss.append(rec.latency_s + extra > rec.task.deadline_s)
    return float(np.mean(lat)), float(np.mean(miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=1200)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--min-gap", type=float, default=4.0, help="simulated seconds between LLM calls")
    ap.add_argument("--no-cache", action="store_true", help="always call the API (true latencies)")
    args = ap.parse_args()

    llm = LLMClient(use_cache=not args.no_cache)
    print(f"LLM available: {llm.available} (model {llm.model})")
    nlp = NLPInterpretationAgent(llm)
    rows, all_trace = [], []
    for condition, drops in (("normal", None), ("wan-outage", [(140, 200, 0.15)])):
        sc = make_scenario(args.tasks, drops=drops)
        next_id = max(t.task_id for t in sc.tasks) + 1
        user_tasks, parsed = [], []
        for t, edge, text in REQUESTS:
            intent = nlp.interpret(text)
            parsed.append({"request": text, "intent": intent.model_dump(mode="json")})
            new = intent_to_tasks(intent, next_id, t, edge)
            next_id += len(new)
            user_tasks += new
        sc.tasks = sorted(sc.tasks + user_tasks, key=lambda x: x.arrival_s)
        user_ids = {t.task_id for t in user_tasks}

        runs = [("greedy-weighted", GreedyWeighted(), None),
                ("greedy-weighted+orch", GreedyWeighted(), OrchestrationAgent()),
                ("agent-llm", AllocationAgent(llm, ForecastingAgent(), max_calls=args.max_calls, min_gap_s=args.min_gap), None),
                ("agent-llm+orch", AllocationAgent(llm, ForecastingAgent(), max_calls=args.max_calls, min_gap_s=args.min_gap),
                 OrchestrationAgent())]
        for label, sched, orch in runs:
            m, sim = sc.run(sched, orch)
            row = {"condition": condition, **m.model_dump(), "scheduler": label}
            trace = getattr(sched, "trace", [])
            if trace:
                row["adj_mean_latency_s"], row["adj_deadline_miss_rate"] = overhead_adjusted(sim, trace)
                row["escalated_tasks"] = sched.n_escalated
                row["llm_decided"] = sched.n_llm_ok
                row["llm_agrees_with_greedy"] = float(np.mean([r["node_id"] == r["greedy_choice"] for r in trace]))
                row["llm_wall_s"] = sched.llm_wall_s
                all_trace += [{"condition": condition, "run": label, **r} for r in trace]
            tr = [r for i, r in sim.records.items() if i not in user_ids]
            row["trace_tasks_J"] = float(np.mean([r.weighted_cost for r in tr]))
            row["trace_tasks_miss_rate"] = float(np.mean([r.missed for r in tr]))
            ur = [sim.records[i] for i in user_ids]
            row["user_req_constraint_ok"] = float(np.mean([
                r.task.preferred_location == "any" or sc.nodes[r.node_id].tier == r.task.preferred_location
                for r in ur]))
            row["user_req_miss_rate"] = float(np.mean([r.missed for r in ur]))
            row["user_req_mean_latency_s"] = float(np.mean([r.latency_s for r in ur]))
            rows.append(row)
            print(f"{condition:10s} {label:22s} J {m.mean_weighted_cost:.3f} lat {m.mean_latency_s:.3f} "
                  f"miss {m.deadline_miss_rate:.3f} calls {m.llm_calls} fallback {m.llm_fallback_rate:.2f} "
                  f"mig {m.migrations}", flush=True)
    out = ROOT / "results"
    pd.DataFrame(rows).to_csv(out / "agentic_demo.csv", index=False)
    with open(out / "agent_trace.jsonl", "w", encoding="utf-8") as f:
        for r in all_trace:
            f.write(json.dumps(r) + "\n")
    (out / "agentic_demo_llm.json").write_text(json.dumps(
        {"model": llm.model, "llm": llm.log.summary(), "nlp_parsed_requests": parsed}, indent=2))
    print(json.dumps(llm.log.summary(), indent=2))


if __name__ == "__main__":
    main()
