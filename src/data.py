"""Data: LCD-style document streams, hash-selected clean splits, fixed split files.

Two selection methods, both deterministic:
  * `lcd`  - exactly LCD's `load_texts` (ref/lcd/code/evaluate.py): `load_dataset(..., streaming=True)
             .shuffle(seed, buffer_size=50_000)`, keep docs with >= 30 whitespace words, first 2000 words, whitespace
             collapsed. Used for everything that must match LCD (calibration, training docs, LCD-protocol ppl).
  * `hash` - scan a split in stored order, bucket every doc by sha1(salt + id) % n_buckets, keep the first n docs of
             one bucket (same filters). Independent of the datasets library's shuffling; disjoint across buckets.

Split files: data/splits/<name>.json (ids + sha1 of the text, committed) and data/texts/<name>.jsonl (texts, not
committed; shipped to Colab as the `data` artifact). Gazeta texts are never written (license): only ids.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .utils import REPO

SPLITS = REPO / "data" / "splits"
TEXTS = Path(os.environ.get("LCDRU_TEXTS", REPO / "data" / "texts"))
NO_TEXT_STORE = {"IlyaGusev/gazeta"}

# name -> source spec. `lcd` specs mirror LCD's seeds: calibration 43, training 45, ppl eval 42 (run_evals.py).
SPECS: dict[str, dict] = {
    "calib_en_128": dict(method="lcd", dataset="allenai/c4", config="en", split="train", seed=43, n=128),
    "calib_ru_128": dict(method="lcd", dataset="allenai/c4", config="ru", split="train", seed=43, n=128,
                         exclude=["train_ru_1024"]),  # fmt: skip
    "train_ru_1024": dict(method="lcd", dataset="allenai/c4", config="ru", split="train", seed=45, n=1024),
    "train_en_256": dict(method="lcd", dataset="allenai/c4", config="en", split="train", seed=45, n=256),
    "ppl_lcd_ru_256": dict(method="lcd", dataset="allenai/c4", config="ru", split="train", seed=42, n=256),
    "ppl_lcd_en_256": dict(method="lcd", dataset="allenai/c4", config="en", split="train", seed=42, n=256),
    "val_ru_64": dict(method="hash", dataset="allenai/c4", config="ru", split="validation", bucket=0, n=64),
    "ppl_ru_256": dict(method="hash", dataset="allenai/c4", config="ru", split="validation", bucket=1, n=256),
    "ppl_en_256": dict(method="hash", dataset="allenai/c4", config="en", split="validation", bucket=1, n=256),
    "ppl_wiki_ru_256": dict(method="hash", dataset="wikimedia/wikipedia", config="20231101.ru", split="train",
                            bucket=1, n=256),  # fmt: skip
    # gazeta: tag v2.0 still has a loading script (unsupported by datasets>=4); main = v2 as parquet, pinned by sha
    "ppl_gazeta_256": dict(method="hash", dataset="IlyaGusev/gazeta", config=None, split="test", bucket=1, n=256,
                           revision="c4d8e46f79e17c4a864a320e6773436dc28fddd8"),  # fmt: skip
}
DERIVED = {"train_ru_256": ("train_ru_1024", 256)}  # prefix of another split (LCD takes the first n of the stream)
N_BUCKETS = 8
SALT = "lcd-ru"


def clean(text: str, min_words: int = 30, max_words: int = 2000) -> str | None:
    words = text.strip().split()
    if len(words) < min_words:
        return None
    return " ".join(words[:max_words])


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _doc_id(rec: dict) -> str:
    for k in ("url", "id"):
        if rec.get(k):
            return str(rec[k])
    return sha1(rec["text"])


def _stream(spec: dict):
    from datasets import load_dataset

    kw = {"revision": spec["revision"]} if spec.get("revision") else {}
    args = (spec["dataset"], spec["config"]) if spec.get("config") else (spec["dataset"],)
    return load_dataset(*args, split=spec["split"], streaming=True, **kw)


def select_lcd(spec: dict, exclude_ids: set[str] = frozenset()) -> list[dict]:
    ds = _stream(spec).shuffle(seed=spec["seed"], buffer_size=50_000)
    out = []
    for rec in ds:
        text = clean(rec["text"])
        if text is None:
            continue
        did = _doc_id(rec)
        if did in exclude_ids:
            continue
        out.append({"id": did, "text": text})
        if len(out) >= spec["n"]:
            break
    return out


def select_hash(spec: dict, exclude_ids: set[str] = frozenset()) -> list[dict]:
    out = []
    for rec in _stream(spec):
        did = _doc_id(rec)
        if int(sha1(SALT + did), 16) % N_BUCKETS != spec["bucket"] or did in exclude_ids:
            continue
        text = clean(rec["text"])
        if text is None:
            continue
        out.append({"id": did, "text": text})
        if len(out) >= spec["n"]:
            break
    return out


def build_split(name: str) -> list[dict]:
    spec = SPECS[name]
    exclude = set()
    for other in spec.get("exclude", []):
        exclude |= set(read_split(other)["ids"])
    docs = (select_lcd if spec["method"] == "lcd" else select_hash)(spec, exclude)
    if len(docs) < spec["n"]:
        raise RuntimeError(f"{name}: only {len(docs)}/{spec['n']} docs")
    return docs


def write_split(name: str, docs: list[dict], spec: dict) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "spec": spec, "n": len(docs), "ids": [d["id"] for d in docs],
           "sha1": [sha1(d["text"]) for d in docs]}  # fmt: skip
    with open(SPLITS / f"{name}.json", "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=0)
    if spec["dataset"] not in NO_TEXT_STORE:
        TEXTS.mkdir(parents=True, exist_ok=True)
        with open(TEXTS / f"{name}.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_split(name: str) -> dict:
    if name in DERIVED:
        src, n = DERIVED[name]
        rec = read_split(src)
        return {**rec, "name": name, "n": n, "ids": rec["ids"][:n], "sha1": rec["sha1"][:n]}
    with open(SPLITS / f"{name}.json") as f:
        return json.load(f)


def load_texts(name: str, n: int | None = None) -> list[str]:
    """Texts of a fixed split, verified against the committed sha1 list."""
    rec = read_split(name)
    base = DERIVED.get(name, (name,))[0]
    spec = SPECS[base]
    p = TEXTS / f"{base}.jsonl"
    if p.exists():
        with open(p) as f:
            docs = [json.loads(line) for line in f]
    else:  # not cached (gazeta, or no data artifact): refetch by id
        want = set(rec["ids"])
        docs = [d for d in _refetch(spec, want)]
        order = {i: k for k, i in enumerate(rec["ids"])}
        docs.sort(key=lambda d: order[d["id"]])
    docs = docs[: rec["n"]]
    if [sha1(d["text"]) for d in docs] != rec["sha1"]:
        raise RuntimeError(f"split {name}: texts do not match the committed sha1 list")
    texts = [d["text"] for d in docs]
    return texts[:n] if n else texts


def _refetch(spec: dict, want: set[str]):
    left = set(want)
    for rec in _stream(spec):
        did = _doc_id(rec)
        if did in left:
            left.discard(did)
            yield {"id": did, "text": clean(rec["text"])}
            if not left:
                return


# ---------------------------------------------------------------------------- Global-MMLU / RuBLiMP splits
MMLU = "CohereLabs/Global-MMLU"


def load_mmlu(lang: str):
    from datasets import load_dataset

    return load_dataset(MMLU, lang, split="test")


def make_mmlu_dev(n_dev: int = 2000, seed: int = 0) -> dict:
    """Stratified (by subject, largest remainder) dev sample of sample_ids; final-test = all other items."""
    import numpy as np

    ds = load_mmlu("ru")
    by_subj: dict[str, list[str]] = {}
    for sid, subj in zip(ds["sample_id"], ds["subject"]):
        by_subj.setdefault(subj, []).append(sid)
    total = sum(len(v) for v in by_subj.values())
    quota = {s: n_dev * len(v) / total for s, v in by_subj.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: (-(quota[s] - take[s]), s))[: n_dev - sum(take.values())]:
        take[s] += 1
    rng = np.random.default_rng(seed)
    dev = []
    for s in sorted(by_subj):
        ids = sorted(by_subj[s])
        dev += [ids[i] for i in sorted(rng.choice(len(ids), size=take[s], replace=False))]
    return {"name": "global_mmlu_dev", "dataset": MMLU, "stratify": "subject", "seed": seed, "n": len(dev),
            "n_total_ru": total, "sample_ids": sorted(dev)}  # fmt: skip


def mmlu_dev_ids() -> set[str]:
    with open(SPLITS / "global_mmlu_dev.json") as f:
        return set(json.load(f)["sample_ids"])


RUBLIMP = "RussianNLP/rublimp"


def make_rublimp_sub(n_per: int = 200, seed: int = 0) -> dict:
    import numpy as np
    from datasets import get_dataset_config_names, load_dataset

    rng = np.random.default_rng(seed)
    out = {}
    for cfg in sorted(get_dataset_config_names(RUBLIMP)):
        ds = load_dataset(RUBLIMP, cfg, split="train")
        idx = sorted(int(i) for i in rng.choice(len(ds), size=min(n_per, len(ds)), replace=False))
        out[cfg] = idx
    return {"name": f"rublimp_sub{n_per}", "dataset": RUBLIMP, "seed": seed, "rows": out}
