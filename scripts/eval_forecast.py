"""Evaluate the forecasting agent's models on the Alibaba per-minute arrival series.

    python scripts/eval_forecast.py      # writes results/forecast_eval.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecagents.agents.forecast_agent import Holt, MovingAverage, Naive, rolling_eval  # noqa: E402


def main():
    s = pd.read_csv(ROOT / "data/processed/alibaba2018_arrivals_per_min.csv")
    # Skip the sparse first hours of the trace, keep 2 days of per-minute counts.
    y = s.arrivals.to_numpy(dtype=float)[6 * 60: 6 * 60 + 2 * 24 * 60]
    rows = []
    for h in (1, 5, 10):
        for model in (Naive(), MovingAverage(10), Holt(0.5, 0.1), Holt(0.3, 0.05)):
            r = rolling_eval(y, model, h=h)
            if isinstance(model, Holt):
                r["model"] = f"holt(a={model.alpha},b={model.beta})"
            rows.append(r)
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/forecast_eval.csv", index=False)
    print(f"series: {len(y)} minutes, mean {np.mean(y):.1f} arrivals/min, std {np.std(y):.1f}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
