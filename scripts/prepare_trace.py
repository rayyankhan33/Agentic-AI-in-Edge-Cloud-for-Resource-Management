"""Fetch (optionally) and preprocess the Alibaba cluster-trace-v2018 batch_task table.

    python scripts/prepare_trace.py --fetch          # stream ~12 MB of the public trace
    python scripts/prepare_trace.py                  # only preprocess data/raw/batch_task_head.csv

Outputs (committed, small):
  data/processed/alibaba2018_tasks_sample.csv   6-hour window of tasks used by the simulator
  data/processed/alibaba2018_arrivals_per_min.csv  per-minute arrival counts (forecasting data)
"""
import argparse
import gzip
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecagents.trace import arrivals_per_minute, load_raw, window  # noqa: E402

URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz"
RAW = ROOT / "data" / "raw" / "batch_task_head.csv"


def fetch(n_bytes: int = 12_000_000) -> None:
    """Download the first n_bytes of the archive and decompress what is complete."""
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{n_bytes}"})
    blob = urllib.request.urlopen(req, timeout=120).read()
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gz:
        try:
            while chunk := gz.read(1 << 20):
                out.write(chunk)
        except (EOFError, OSError):
            pass                                   # truncated stream: keep what we have
    data = out.getvalue()[512:]                    # skip the tar header of the single member
    data = data[: data.rfind(b"\n") + 1]           # drop the partial last line
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(data)
    print(f"wrote {RAW} ({len(data) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--window-start-h", type=float, default=48.0)
    ap.add_argument("--window-hours", type=float, default=6.0)
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        fetch()
    df = load_raw(RAW)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    w = window(df, int(args.window_start_h * 3600), args.window_hours)
    w.to_csv(out / "alibaba2018_tasks_sample.csv", index=False)
    arrivals_per_minute(df).to_csv(out / "alibaba2018_arrivals_per_min.csv", index=False)
    print(f"{len(df)} usable tasks; window has {len(w)} tasks")


if __name__ == "__main__":
    main()
