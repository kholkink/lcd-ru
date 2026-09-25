"""End-to-end training of one language's corrections (O1, O2, O3, O5), LCD's loop:

one document per step (truncated to max_len, no packing), AdamW, linear warmup then cosine to 0 (LambdaLR exactly as
LCD, so the first step has lr 0), grad clip on the correction params, fixed number of steps cycling over the docs.
`doc_order: fixed` cycles in stored order (LCD); `seeded` draws a new permutation per epoch from the seed.

Checkpoints (corrections + optimizer + scheduler + step + RNG) go to <out>/ckpt.pt every `ckpt_every` steps and on
SIGTERM (the Colab worker sends it on cancel/timeout); a rerun with the same out dir resumes from there.
Blockwise training (O4-alpha) is added in phase 3.
"""

from __future__ import annotations

import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import corrections as C
from .objectives import Objective, capture_blocks


class _Term:
    def __init__(self):
        self.hit = False
        self.prev = None

    def __enter__(self):
        try:
            self.prev = signal.signal(signal.SIGTERM, self._h)
        except ValueError:  # not in the main thread
            self.prev = None
        return self

    def _h(self, *_):
        self.hit = True

    def __exit__(self, *exc):
        if self.prev is not None:
            signal.signal(signal.SIGTERM, self.prev)


def lcd_lr_lambda(steps: int, warmup: int):
    def f(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return f


def doc_schedule(n_docs: int, steps: int, order: str, seed: int) -> list[int]:
    if order == "fixed":
        return [i % n_docs for i in range(steps)]
    rng = np.random.default_rng(seed)
    out: list[int] = []
    while len(out) < steps:
        out += rng.permutation(n_docs).tolist()
    return out[:steps]


def _save_ckpt(path: Path, model, lang, opt, sched, step, extra) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step, "corr": C.state(model, [lang]), "opt": opt.state_dict(), "sched": sched.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                else None, "py": random.getstate(), "np": np.random.get_state()}, "extra": extra,
    }, tmp)  # fmt: skip
    tmp.replace(path)


def _load_ckpt(path: Path, model, lang, opt, sched) -> tuple[int, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mods = C.modules(model)
    for key, t in ck["corr"]["tensors"].items():
        n, ab, lg = key.rsplit(".", 2)
        getattr(mods[n], ab)[lg].data.copy_(t)
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    torch.set_rng_state(ck["rng"]["torch"])
    if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    random.setstate(ck["rng"]["py"])
    np.random.set_state(ck["rng"]["np"])
    return ck["step"], ck.get("extra", {})


def train_e2e(student: nn.Module, tok, texts: list[str], objective: Objective, cfg: dict, lang: str, out_dir: str | Path,
              teacher: nn.Module | None = None, val_fn=None, log=print) -> dict:  # fmt: skip
    """cfg keys: steps, lr, weight_decay, warmup_ratio, grad_clip, max_len, seed, doc_order, ckpt_every, log_every,
    eval_every, grad_ckpt. `val_fn(student) -> dict` is called every eval_every steps (0 = never)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["steps"])
    dev = next(student.parameters()).device
    params = C.trainable(student, lang)
    C.set_lang(student, lang)
    if teacher is not None:
        C.set_lang(teacher, None) if C.modules(teacher) else None
        for p in teacher.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.01)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lcd_lr_lambda(steps, int(steps * float(cfg.get("warmup_ratio", 0.1)))))
    order = doc_schedule(len(texts), steps, cfg.get("doc_order", "fixed"), int(cfg.get("seed", 0)))
    ckpt = out / "ckpt.pt"
    start, hist = 0, {"loss": [], "val": []}
    if ckpt.exists():
        start, hist = _load_ckpt(ckpt, student, lang, opt, sched)
        log(f"  resumed from step {start}")
    if cfg.get("grad_ckpt"):
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    max_len = int(cfg.get("max_len", 512))
    log_every, ckpt_every, eval_every = (int(cfg.get(k, d)) for k, d in
                                         (("log_every", 25), ("ckpt_every", 100), ("eval_every", 0)))  # fmt: skip
    hs_t, hs_s, window = [], [], []
    t0 = time.time()
    with _Term() as term:
        step = start
        for step in range(start, steps):
            ids = tok(texts[order[step]], return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
            t_logits = None
            if objective.needs_teacher:
                with torch.no_grad(), capture_blocks(teacher, hs_t) if objective.needs_hidden else _null():
                    t_logits = teacher(input_ids=ids).logits
            with capture_blocks(student, hs_s) if objective.needs_hidden else _null():
                s_logits = student(input_ids=ids).logits
            loss, logs = objective(ids, None, s_logits, list(hs_s), t_logits, list(hs_t))
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(params, float(cfg.get("grad_clip", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            window.append({"loss": loss.item(), **logs, "gnorm": gnorm.item()})
            if (step + 1) % log_every == 0 or step + 1 == steps:
                avg = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
                hist["loss"].append({"step": step + 1, **avg, "lr": sched.get_last_lr()[0]})
                log(f"  step {step + 1}/{steps} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    + f" lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")  # fmt: skip
                window = []
            if eval_every and val_fn and ((step + 1) % eval_every == 0 or step + 1 == steps):
                student.eval()
                v = val_fn(student)
                student.train()
                hist["val"].append({"step": step + 1, **v})
                log(f"  val @ {step + 1}: " + " ".join(f"{k}={x:.4f}" for k, x in v.items() if isinstance(x, float)))
            if term.hit or (ckpt_every and (step + 1) % ckpt_every == 0):
                _save_ckpt(ckpt, student, lang, opt, sched, step + 1, hist)
            if term.hit:
                log(f"  SIGTERM: checkpoint at step {step + 1}, exiting")
                raise SystemExit(143)
    student.eval()
    if cfg.get("grad_ckpt"):
        student.gradient_checkpointing_disable()
    C.set_lang(student, lang)
    return {"steps": steps, "resumed_from": start, "seconds": time.time() - t0, "history": hist,
            "final_loss": hist["loss"][-1] if hist["loss"] else None}  # fmt: skip


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
