#!/usr/bin/env python
"""Phase 0 end-to-end run: quantize -> evaluate FP16 / INT3 / W4 -> train O1 and O3 -> evaluate each.

    python scripts/smoke.py --config configs/p0_smoke.yaml
"""

import copy
import time

import _common
import torch

from src import corrections as C
from src import pipeline
from src.models import load_hf
from src.utils import append_jsonl, device, set_seed


def sub(cfg: dict, suffix: str) -> dict:
    c = copy.deepcopy(cfg)
    c["name"] = f"{cfg['name']}_{suffix}"
    return c


def main() -> None:
    cfg = _common.config_from_cli(__doc__)
    t0 = time.time()
    stamps = {}
    teacher = load_hf(cfg["model"], device())
    pipeline.evaluate(sub(cfg, "fp16"), teacher, "fp16", None, teacher)
    stamps["fp16_eval"] = time.time() - t0

    w4 = pipeline.get_base(cfg["model"], cfg["w4_base"])
    w4cfg = sub(cfg, "w4")
    w4cfg["eval"] = {"batch_size": cfg["eval"]["batch_size"], "ppl": cfg["eval"]["ppl"][:3]}
    pipeline.evaluate(w4cfg, w4, "w4", None)
    del w4
    stamps["w4"] = time.time() - t0

    student = pipeline.get_base(cfg["model"], cfg["base"])
    stamps["int3_quantize"] = time.time() - t0
    pipeline.evaluate(sub(cfg, "int3"), student, "int3", None, teacher)
    stamps["int3_eval"] = time.time() - t0
    probe = pipeline.load_tokenizer(cfg["model"])(["Quantization control: the quick brown fox."], return_tensors="pt")
    with torch.no_grad():
        ref = student(input_ids=probe.input_ids.to(device())).logits
    checks = {}

    for obj, kw in cfg["objectives"].items():
        c = sub(cfg, obj.lower())
        c["train"] = {**cfg["train"], "objective": obj, "obj_kw": kw}
        C.remove(student) if C.modules(student) else None
        set_seed(int(cfg.get("seed", 0)))
        out = pipeline.train(c, student=student, teacher=teacher)
        stamps[f"{obj}_train"] = time.time() - t0
        pipeline.evaluate(c, out["student"], obj, c["train"]["lang"], teacher)
        # control (a): correction switched off must give INT3 back exactly
        C.set_lang(student, None)
        with torch.no_grad():
            checks[f"{obj}_off_equals_int3"] = bool(torch.equal(student(input_ids=probe.input_ids.to(device())).logits, ref))
        stamps[f"{obj}_eval"] = time.time() - t0

    total = time.time() - t0
    rec = {"run": cfg["name"], "total_minutes": round(total / 60, 2), "checks": checks,
           "stamps_s": {k: round(v) for k, v in stamps.items()}}
    append_jsonl(pipeline.RUNS / cfg["name"] / "timing.jsonl", rec)
    print(f"SMOKE DONE in {total / 60:.1f} min: {rec['stamps_s']} checks={checks}")
    if not all(checks.values()):
        raise SystemExit("control (a) failed: correction off != INT3")


if __name__ == "__main__":
    main()
