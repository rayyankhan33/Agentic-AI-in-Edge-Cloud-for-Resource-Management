"""Alibaba cluster-trace-v2018 (batch_task) -> TaskSpec stream.

The raw trace describes datacenter batch tasks (start/end time, planned CPU
and memory). We keep its arrival process and its relative size distribution,
then rescale both to edge-cloud magnitudes:

* arrivals are time-compressed to a target rate (keeps burstiness),
* work = (planned cores x duration) ** 0.5 (compresses the datacenter tail),
  rescaled so the offered load is ``target_load_ghz`` on average,
* input size follows planned memory, rescaled to ``median_input_mb``,
* the task type is derived from the work quantile, and the deadline from the
  type's slack factor times the task's run time on its own edge.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import TaskSpec, TaskType

RAW_COLUMNS = ["task_name", "instance_num", "job_name", "task_type", "status",
               "start_time", "end_time", "plan_cpu", "plan_mem"]

# (task type, deadline slack over standalone run time) by work quantile band
TYPE_BANDS = [
    (0.20, TaskType.WEB_REQUEST, 3.0),
    (0.40, TaskType.IOT_AGGREGATION, 4.0),
    (0.65, TaskType.ML_INFERENCE, 3.0),
    (0.85, TaskType.VIDEO_ANALYTICS, 4.0),
    (1.01, TaskType.BATCH_ANALYTICS, 10.0),
]


def load_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, nrows=nrows)
    df = df[(df.status == "Terminated") & (df.end_time > df.start_time) & (df.plan_cpu > 0)]
    df = df.assign(duration=df.end_time - df.start_time)
    return df.sort_values("start_time").reset_index(drop=True)


def arrivals_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    minute = (df.start_time // 60).astype(int)
    counts = minute.value_counts().sort_index()
    full = pd.RangeIndex(counts.index.min(), counts.index.max() + 1)
    counts = counts.reindex(full, fill_value=0)
    return pd.DataFrame({"minute": counts.index, "arrivals": counts.values})


def window(df: pd.DataFrame, start_s: int, hours: float) -> pd.DataFrame:
    end_s = start_s + int(hours * 3600)
    keep = ["job_name", "start_time", "duration", "plan_cpu", "plan_mem", "instance_num"]
    return df[(df.start_time >= start_s) & (df.start_time < end_s)][keep].reset_index(drop=True)


def to_tasks(df: pd.DataFrame, n_edges: int, target_rate: float, target_load_ghz: float,
             median_input_mb: float, edge_freq_ghz: list[float], limit: int | None = None,
             offset: int = 0) -> list[TaskSpec]:
    """Map processed trace rows to simulator tasks (deterministic)."""
    df = df.iloc[offset: offset + limit if limit else None].reset_index(drop=True)
    t = df.start_time.to_numpy(dtype=float)
    t = t - t[0]
    span = max(t[-1], 1.0)
    t = t * (len(df) / target_rate) / span          # compress to the target mean rate

    # Stable sub-second jitter so tasks sharing a trace second don't arrive together.
    jitter = np.array([(zlib.crc32(f"{i}".encode()) % 1000) / 1000.0 for i in range(len(df))])
    t = np.sort(t + jitter * (1.0 / target_rate))

    raw_work = (df.plan_cpu.to_numpy() / 100.0) * np.minimum(df.duration.to_numpy(), 3600)
    raw_work = np.clip(raw_work, 1e-3, np.quantile(raw_work, 0.99)) ** 0.5
    work = raw_work * (target_load_ghz / target_rate / raw_work.mean())   # giga-cycles

    mem = np.clip(df.plan_mem.to_numpy(), 0.01, None)
    input_mb = np.clip(mem * (median_input_mb / np.median(mem)), 0.05, 50.0)

    ranks = pd.Series(work).rank(pct=True).to_numpy()
    tasks = []
    for i in range(len(df)):
        src = zlib.crc32(str(df.job_name.iloc[i]).encode()) % n_edges
        for upper, ttype, slack in TYPE_BANDS:
            if ranks[i] <= upper:
                break
        standalone = work[i] / edge_freq_ghz[src] + input_mb[i] * 8 / 100.0
        tasks.append(TaskSpec(
            task_id=i, arrival_s=float(t[i]), work_gcycles=float(work[i]),
            input_mb=float(input_mb[i]), output_mb=float(input_mb[i] * 0.1),
            deadline_s=float(slack * standalone), source_edge=int(src), task_type=ttype,
        ))
    return tasks
