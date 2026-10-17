"""Prediction & Forecasting Agent: short-horizon forecasts of arrivals and WAN bandwidth.

Phase 1 uses lightweight statistical forecasters (Holt double exponential
smoothing, EWMA) that run in microseconds inside the decision loop; a learned
model (e.g. a small temporal transformer) is a Phase 2 candidate and will be
compared against these on the same series.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..contracts import Forecast


class Naive:
    name = "naive"

    def fit_predict(self, y: np.ndarray, h: int = 1) -> float:
        return float(y[-1])


class MovingAverage:
    name = "moving-avg"

    def __init__(self, k: int = 10):
        self.k = k

    def fit_predict(self, y, h=1):
        return float(np.mean(y[-self.k:]))


class Holt:
    """Double exponential smoothing (level + trend)."""

    name = "holt"

    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        self.alpha, self.beta = alpha, beta

    def fit_predict(self, y, h=1):
        level, trend = float(y[0]), 0.0
        for v in y[1:]:
            prev = level
            level = self.alpha * v + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev) + (1 - self.beta) * trend
        return max(level + h * trend, 0.0)


def rolling_eval(series: np.ndarray, model, warmup: int = 60, window: int = 120, h: int = 1) -> dict:
    """One-step-ahead (or h-step) rolling-origin evaluation: MAE, RMSE, sMAPE."""
    preds, truth = [], []
    for t in range(warmup, len(series) - h + 1):
        preds.append(model.fit_predict(series[max(0, t - window):t], h))
        truth.append(series[t + h - 1])
    p, y = np.array(preds), np.array(truth)
    return {"model": model.name, "h": h, "n": len(y), "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "smape": float(np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-9)))}


class ForecastingAgent:
    """Keeps online histories and answers the ``get_forecast`` tool."""

    name = "forecast-holt"

    def __init__(self, bin_s: float = 2.0, horizon_s: float = 10.0, history: int = 120):
        self.bin_s, self.horizon_s = bin_s, horizon_s
        self.arrivals: deque = deque(maxlen=history)
        self.bandwidth: deque = deque(maxlen=history)
        self._bin_start, self._count = 0.0, 0
        self.model = Holt(alpha=0.3, beta=0.05)   # best one-step setting in scripts/eval_forecast.py

    def observe(self, now: float, new_arrivals: int, wan_mbps: float) -> None:
        self._count += new_arrivals
        while now - self._bin_start >= self.bin_s:
            self.arrivals.append(self._count / self.bin_s)
            self.bandwidth.append(wan_mbps)
            self._count, self._bin_start = 0, self._bin_start + self.bin_s

    def forecast(self) -> Forecast:
        h = max(int(self.horizon_s / self.bin_s), 1)
        if len(self.arrivals) < 3:
            rate = self.arrivals[-1] if self.arrivals else 0.0
            bw = self.bandwidth[-1] if self.bandwidth else 0.0
        else:
            rate = self.model.fit_predict(np.array(self.arrivals), h)
            bw = Holt(alpha=0.3, beta=0.05).fit_predict(np.array(self.bandwidth), h)
        return Forecast(horizon_s=self.horizon_s, arrival_rate=rate, wan_bandwidth_mbps=bw, method="holt")
