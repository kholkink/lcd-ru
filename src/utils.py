"""Seeds, configs, run records (append-only jsonl), environment info."""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------- configs
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """YAML with optional `inherit: other.yaml` (relative to this file), then `a.b.c=value` overrides."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *head, last = key.split(".")
        for k in head:
            node = node.setdefault(k, {})
        node[last] = _parse_value(val)
    cfg.setdefault("name", path.stem)
    return cfg


# ---------------------------------------------------------------------------- provenance
def git_hash() -> str:
    """Commit of the code: GIT_HASH file written by kaggle/push_code.sh (Colab), else `git rev-parse`."""
    f = REPO / "GIT_HASH"
    if f.exists():
        return f.read_text().strip()
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
        return h.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def env_info() -> dict:
    import datasets
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import compressed_tensors

        info["compressed_tensors"] = compressed_tensors.__version__
    except ImportError:
        pass
    return info


def expand_env(path: str) -> str:
    """$LCDRU_RUNS etc. in config paths (the Colab worker sets them)."""
    return os.path.expandvars(path)


def now_utc() -> str:
    return f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    """Append one record; files are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def run_record(cfg: dict, kind: str, metrics: dict, seconds: float, **extra) -> dict:
    return {
        "time_utc": now_utc(),
        "run": cfg.get("name"),
        "kind": kind,
        "git": git_hash(),
        "seed": cfg.get("seed"),
        "config": cfg,
        "env": env_info(),
        "gpu_seconds": round(seconds, 1),
        "metrics": metrics,
        **extra,
    }


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.time() - self.t0
