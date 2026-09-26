#!/usr/bin/env python
"""Train one language's corrections (cfg.train) on cfg.base, resuming from <run>/ckpt.pt, then evaluate the corrected
model on cfg.eval (slot cfg.train.lang; per-entry `slot` overrides). Control (a) is checked on the way: with the
correction switched off the model must reproduce the base model's logits bit for bit."""

import _common
import torch

from src import corrections as C
from src import pipeline
from src.models import load_hf, load_tokenizer
from src.objectives import make
from src.utils import device

PROBE = "Контроль: при выключенной поправке выход совпадает с базовой моделью. Control (a), English too."

if __name__ == "__main__":
    cfg = _common.config_from_cli(__doc__)
    lang = cfg["train"].get("lang", "ru")
    student = pipeline.get_base(cfg["model"], pipeline.base_spec(cfg))
    probe = load_tokenizer(cfg["model"])(PROBE, return_tensors="pt").input_ids.to(device())
    with torch.no_grad():
        ref = student(input_ids=probe).logits
    needs_teacher = make(cfg["train"]["objective"], **cfg["train"].get("obj_kw", {})).needs_teacher
    teacher = load_hf(cfg["model"], device()) if needs_teacher or cfg.get("eval", {}).get("kl") else None
    out = pipeline.train(cfg, student=student, teacher=teacher)
    C.set_lang(student, None)
    with torch.no_grad():
        off_ok = bool(torch.equal(student(input_ids=probe).logits, ref))
    C.set_lang(student, lang)
    pipeline.record(cfg, "check", {"off_equals_base": off_ok}, 0.0, lang=lang)
    print(f"control (a): correction off == base: {off_ok}")
    if cfg.get("eval"):
        pipeline.evaluate(cfg, student, cfg.get("eval", {}).get("cond", cfg["name"]), lang, teacher)
    if not off_ok:
        raise SystemExit("control (a) failed")
