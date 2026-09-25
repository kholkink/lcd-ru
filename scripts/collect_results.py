#!/usr/bin/env python
"""Append run records shipped back by the Colab worker into results/<file>.jsonl (append-only, deduplicated).

    python colab/jobs.py results --id p0-smoke          # downloads build/jobs_status/a/results/p0-smoke/
    python scripts/collect_results.py --job p0-smoke --into phase0
"""

import argparse
import glob
import json
import os

import _common  # noqa: F401

from src.utils import REPO, append_jsonl


def key(rec: dict) -> tuple:
    return rec.get("run"), rec.get("kind"), rec.get("cond"), rec.get("time_utc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--into", required=True, help="results/<into>.jsonl")
    ap.add_argument("--worker", default="a")
    a = ap.parse_args()
    src = REPO / "build" / "jobs_status" / a.worker / "results" / a.job
    dst = REPO / "results" / f"{a.into}.jsonl"
    seen = set()
    if dst.exists():
        seen = {key(json.loads(line)) for line in dst.read_text().splitlines() if line.strip()}
    n = 0
    for p in sorted(glob.glob(str(src / "*results.jsonl"))):
        for line in open(p):
            rec = json.loads(line)
            if key(rec) not in seen:
                append_jsonl(dst, {**rec, "job": a.job, "source_file": os.path.basename(p)})
                seen.add(key(rec))
                n += 1
    print(f"{n} new records -> {dst}")


if __name__ == "__main__":
    main()
