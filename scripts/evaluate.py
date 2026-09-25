#!/usr/bin/env python
"""Evaluate one condition: cfg.base (+ cfg.eval.corrections file, slot cfg.eval.lang) on cfg.eval.
The FP16 teacher is loaded for KL / e_l when cfg.eval.kl is set."""

import _common

from src import corrections as C
from src import pipeline
from src.models import load_hf
from src.utils import device

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    ev = cfg["eval"]
    model = pipeline.get_base(cfg["model"], cfg["base"])
    lang = None
    if ev.get("corrections"):
        cc = cfg.get("corrections", {})
        C.install(model, int(cc.get("rank", 2)), list(cc.get("langs", ["ru", "en"])), cc.get("placement", "all"))
        C.load(model, ev["corrections"], strict=True)
        lang = ev.get("lang", "ru")
    teacher = load_hf(cfg["model"], device()) if ev.get("kl") and cfg["base"].get("kind") != "fp16" else None
    if ev.get("kl") and teacher is None:
        teacher = model
    pipeline.evaluate(cfg, model, ev.get("cond", cfg["name"]), lang, teacher)
