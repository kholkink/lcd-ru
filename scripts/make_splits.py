#!/usr/bin/env python
"""Build and declare the fixed splits (PLAN.md §3.3, §5.4): data/splits/*.json (ids + sha1, committed) and
data/texts/*.jsonl (texts, shipped to Colab as the `data` artifact). Existing split files are never overwritten.

    python scripts/make_splits.py                 # everything that is missing
    python scripts/make_splits.py val_ru_64 ...   # only these
"""

import sys
import time

import _common  # noqa: F401

from src import data

MMLU_DEV = "global_mmlu_dev"
RUBLIMP_SUB = "rublimp_sub200"


def main() -> None:
    names = sys.argv[1:] or list(data.SPECS) + [MMLU_DEV, RUBLIMP_SUB]
    # dependencies first (calib_ru excludes the training docs)
    names.sort(key=lambda n: 0 if n == "train_ru_1024" else 1)
    for name in names:
        path = data.SPLITS / f"{name}.json"
        if path.exists():
            print(f"{name}: exists, kept")
            continue
        t0 = time.time()
        if name == MMLU_DEV:
            import json

            rec = data.make_mmlu_dev()
            data.SPLITS.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=0))
            print(f"{name}: {rec['n']} of {rec['n_total_ru']} items ({time.time() - t0:.0f}s)")
            continue
        if name == RUBLIMP_SUB:
            import json

            rec = data.make_rublimp_sub()
            path.write_text(json.dumps(rec, indent=0))
            print(f"{name}: {len(rec['rows'])} phenomena ({time.time() - t0:.0f}s)")
            continue
        docs = data.build_split(name)
        data.write_split(name, docs, data.SPECS[name])
        print(f"{name}: {len(docs)} docs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
