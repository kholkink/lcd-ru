"""Glue for the CLI scripts: build a base model from a config, train corrections, evaluate a condition.

Run directory: $LCDRU_RUNS/<name>/ with config.yaml, corrections.pt, ckpt.pt, results.jsonl (one record per
stage/condition, append-only) and mmlu_<cond>_<lang>.jsonl.gz (per-item A-D log-probs). Quantized INT3-ours bases live
in their own run dir ($LCDRU_RUNS/<base name>/quant.safetensors) and are reused when present.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import torch
import yaml

from . import corrections as C
from .data import load_texts
from .models import LCD_INT3, load_base, load_hf, load_tokenizer
from .utils import REPO, Timer, append_jsonl, device, run_record, set_seed

RUNS = Path(os.environ.get("LCDRU_RUNS", REPO / "runs"))


def run_dir(cfg: dict) -> Path:
    d = RUNS / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return d


def record(cfg: dict, kind: str, metrics: dict, seconds: float, **extra) -> dict:
    rec = run_record(cfg, kind, metrics, seconds, **extra)
    append_jsonl(RUNS / cfg["name"] / "results.jsonl", rec)
    return rec


# ---------------------------------------------------------------------------- bases
def quant_base_dir(model_id: str, base: dict) -> Path:
    name = base.get("name") or f"int3ours-{model_id.split('/')[-1].lower()}-{base.get('calib', 'calib_en_128')}"
    return RUNS / name


def ensure_quantized(model_id: str, base: dict, log=print) -> Path:
    """INT3-ours: quantize once (GPTQ on the calibration split) and store codes; reuse afterwards."""
    from .quant.gptq import QuantConfig, quantize_model, save_quant

    out = Path(base["path"]) if base.get("path") else quant_base_dir(model_id, base)
    if (out / "quant.safetensors").exists():
        return out
    qcfg = QuantConfig(**base.get("quant", {}))
    tok = load_tokenizer(model_id)
    model = load_hf(model_id, device())
    texts = load_texts(base.get("calib", "calib_en_128"), base.get("calib_n"))
    ml = int(base.get("calib_max_len", 512))
    calib = [tok(t, return_tensors="pt", truncation=True, max_length=ml).input_ids for t in texts]
    log(f"GPTQ {model_id}: {len(calib)} calibration docs x <= {ml} tokens, {qcfg}")
    with Timer() as t:
        state = quantize_model(model, calib, qcfg, log=log)
    save_quant(state, qcfg, out, meta={"model": model_id, "calib": base.get("calib"), "seconds": t.seconds})
    log(f"GPTQ done in {t.seconds:.0f}s -> {out}")
    del model
    return out


def base_spec(cfg: dict) -> dict:
    """cfg.base, completed from cfg.base_ours when it is an INT3-ours base given only by kind."""
    base = dict(cfg["base"])
    if base.get("kind") == "gptq-ours" and "quant" not in base:
        base = {**cfg.get("base_ours", {}), **base}
    return base


def get_base(model_id: str, base: dict, log=print) -> torch.nn.Module:
    base = dict(base)
    if base.get("kind") == "gptq-ours":
        base["path"] = str(ensure_quantized(model_id, base, log))
    if base.get("kind") == "lcd-int3" and not base.get("repo"):
        base["repo"] = LCD_INT3[model_id]
    return load_base(model_id, base, device())


# ---------------------------------------------------------------------------- training
def train(cfg: dict, student=None, teacher=None, log=print) -> dict:
    from .eval.kl import teacher_agreement
    from .objectives import make
    from .train import train_e2e

    d = run_dir(cfg)
    tr = cfg["train"]
    set_seed(int(cfg.get("seed", 0)))
    tok = load_tokenizer(cfg["model"])
    if student is None:
        student = get_base(cfg["model"], cfg["base"], log)
    cc = cfg.get("corrections", {})
    if not C.modules(student):
        C.install(student, int(cc.get("rank", 2)), list(cc.get("langs", ["ru", "en"])), cc.get("placement", "all"),
                  float(cc.get("init_std", 0.01)))  # fmt: skip
    obj = make(tr["objective"], **tr.get("obj_kw", {}))
    if (obj.needs_teacher or tr.get("eval_every")) and teacher is None:
        teacher = load_hf(cfg["model"], device())
    texts = load_texts(tr["data"], tr.get("n_docs"))
    val_fn = None
    if tr.get("eval_every"):
        val = load_texts(tr.get("val", "val_ru_64"), tr.get("val_n"))
        val_fn = lambda s: {k: v for k, v in teacher_agreement(s, teacher, tok, val, int(tr.get("max_len", 512)))
                            .items() if k in ("kl", "top1", "e_mean")}  # fmt: skip
    lang = tr.get("lang", "ru")
    with Timer() as t:
        res = train_e2e(student, tok, texts, obj, {**tr, "seed": cfg.get("seed", 0)}, lang, d, teacher, val_fn, log)
    C.save(student, d / "corrections.pt", [lang], meta={"run": cfg["name"], "objective": tr["objective"]})
    metrics = {"n_params": C.n_params(student, lang), "final": res["final_loss"], "history": res["history"],
               "resumed_from": res["resumed_from"]}  # fmt: skip
    record(cfg, "train", metrics, t.seconds, lang=lang)
    return {"student": student, "teacher": teacher, "train": res}


# ---------------------------------------------------------------------------- evaluation
def _slot(entry: dict, lang: str | None) -> str | None:
    """Correction slot for one eval entry: `slot: default` (the condition's language), `off`, or a language."""
    slot = entry.get("slot", "default")
    return lang if slot == "default" else None if slot == "off" else slot


def _tag(entry: dict) -> str:
    """Key suffix for entries that set their slot explicitly, so the same split can be evaluated with several slots."""
    return f"@{entry['slot']}" if entry.get("slot", "default") != "default" else ""


def _as_list(x) -> list:
    return x if isinstance(x, list) else [x] if x else []


def evaluate(cfg: dict, model, cond: str, lang: str | None, teacher=None, log=print) -> dict:
    """Evaluate one condition on cfg['eval']: the base model with correction slot `lang` active (None = off).

    Every entry may set `slot: default | off | <lang>`; the slot used is stored with each metric. KL / e_l entries
    need a teacher (FP16) and are skipped when the evaluated model is the teacher itself."""
    from .eval import kl as KL
    from .eval import mmlu as MM
    from .eval import ppl as PP
    from .eval import rublimp as RB

    d = run_dir(cfg)
    ev = cfg.get("eval", {})
    bs = int(ev.get("batch_size", 8))
    tok = load_tokenizer(cfg["model"])
    corrected = bool(C.modules(model))

    def use(entry):
        slot = _slot(entry, lang) if corrected else None
        if corrected:
            C.set_lang(model, slot)
        return slot

    out: dict = {}
    with Timer() as t:
        for p in ev.get("ppl", []):
            slot = use(p)
            r = PP.perplexity(model, tok, load_texts(p["split"], p.get("n")), int(p.get("max_len", 512)), bs)
            out[f"ppl/{p['split']}{_tag(p)}"] = {k: v for k, v in r.items() if k != "per_doc"} | {"slot": slot}
            log(f"  [{cond}|{slot}] ppl {p['split']}: {r['ppl']:.3f} ({r['tokens']} tok)")
        for k in _as_list(ev.get("kl")):
            if teacher is None or teacher is model:
                continue
            slot = use(k)
            r = KL.teacher_agreement(model, teacher, tok, load_texts(k["split"], k.get("n")), int(k.get("max_len", 512)),
                                     batch_size=max(1, bs // 2), linears=bool(k.get("linears")))  # fmt: skip
            out[f"kl/{k['split']}{_tag(k)}"] = r | {"slot": slot}
            log(f"  [{cond}|{slot}] KL {k['split']}: {r['kl']:.4f}, top1 {r['top1']:.4f}, e_mean {r['e_mean']:.4f}")
        for m in ev.get("mmlu", []):
            slot = use(m)
            split = m.get("split", "dev")
            items = MM.build_items(m["lang"], split, m.get("n"))
            recs = MM.score(model, tok, items, int(m.get("batch_size", 16)))
            s = MM.summarize(recs)
            out[f"mmlu/{m['lang']}/{split}{_tag(m)}"] = s | {"slot": slot}
            with gzip.open(d / f"mmlu_{cond}_{m['lang']}_{split}{_tag(m).replace('@', '_')}.jsonl.gz", "wt") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            log(f"  [{cond}|{slot}] Global-MMLU {m['lang']}/{split}: acc {s['acc']:.4f} (n={s['n']})")
        if ev.get("rublimp"):
            rb = ev["rublimp"]
            slot = use(rb)
            r = RB.evaluate(model, tok, rb.get("subsample", "rublimp_sub200"), int(rb.get("batch_size", 32)),
                            rb.get("n_per"))  # fmt: skip
            out[f"rublimp{_tag(rb)}"] = r | {"slot": slot}
            log(f"  [{cond}|{slot}] RuBLiMP ({rb.get('subsample')}, n_per={rb.get('n_per')}): {r['acc']:.4f}")
    if corrected:
        C.set_lang(model, lang)
    record(cfg, "eval", out, t.seconds, cond=cond, lang=lang)
    return out
