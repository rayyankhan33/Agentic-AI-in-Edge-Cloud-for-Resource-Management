"""Evaluate the NLP Interpretation Agent (Gemini) against the rule-based parser.

    python scripts/eval_nlp.py            # writes results/nlp_eval.csv and results/nlp_eval_summary.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ecagents.agents.llm import LLMClient  # noqa: E402
from ecagents.agents.nlp_agent import NLPInterpretationAgent, RuleBasedParser  # noqa: E402

FIELDS = ["task_type", "input_size_mb", "deadline_ms", "priority", "preferred_location", "count"]


def field_ok(name, pred, gold):
    if name in ("input_size_mb", "deadline_ms"):
        return abs(float(pred) - float(gold)) <= 0.05 * abs(float(gold)) + 1e-3   # within 5%
    return str(pred) == str(gold)


def main():
    items = [json.loads(l) for l in open(ROOT / "data/eval/nl_requests.jsonl", encoding="utf-8")]
    llm = LLMClient()
    parsers = [RuleBasedParser(), NLPInterpretationAgent(llm)]
    rows = []
    for item in items:
        for p in parsers:
            intent = p.interpret(item["request"]).model_dump(mode="json")
            row = {"id": item["id"], "parser": p.name}
            for f in FIELDS:
                row[f] = field_ok(f, intent[f], item["gold"][f])
                row[f"pred_{f}"] = intent[f]
            row["exact"] = all(row[f] for f in FIELDS)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "nlp_eval.csv", index=False)
    summary = df.groupby("parser")[FIELDS + ["exact"]].mean().round(3)
    summary["n"] = len(items)
    print(summary.to_string())
    res = {"per_parser": summary.reset_index().to_dict(orient="records"), "llm": llm.log.summary(),
           "llm_fallbacks": parsers[1].fallbacks, "model": llm.model}
    (out / "nlp_eval_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["llm"], indent=2))


if __name__ == "__main__":
    main()
