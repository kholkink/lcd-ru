#!/usr/bin/env python
"""Train one language's corrections (cfg.train) on cfg.base; resumes from <run>/ckpt.pt. Evaluates afterwards when
cfg.eval is set (the corrected model, correction slot cfg.train.lang)."""

import _common

from src import pipeline

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    out = pipeline.train(cfg)
    if cfg.get("eval"):
        pipeline.evaluate(cfg, out["student"], cfg["name"], cfg["train"].get("lang", "ru"), out["teacher"])
