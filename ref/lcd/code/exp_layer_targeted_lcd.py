#!/usr/bin/env python3
"""
Experiment: Layer-Targeted LCD on Llama-3.2-3B
===============================================
Tests whether concentrating higher-rank corrections on Llama's worst-error
depth band (layers 7-14, identified from the layerwise heatmap) improves
recovery beyond uniform rank-2 correction.

Three configurations, matched approximately on parameter budget:
  1. uniform_r2   — rank-2 on ALL layers         (baseline)
  2. targeted_r4  — rank-4 on layers 7-14 ONLY   (depth-targeted)
  3. targeted_r2  — rank-2 on layers 7-14 ONLY   (controls for rank vs targeting)

If targeted_r4 > uniform_r2, the paper's §5.6 claim upgrades from
"we diagnose the depth asymmetry" to "we also exploit it."

Loads the pre-saved INT3 quantized model (skip re-quantization).

Usage:
    python exp_layer_targeted_lcd.py --device cuda:0 \
        --load-quantized-dir lcd_results/meta-llama_Llama-3.2-3B/quantized
"""
import argparse
import gc
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pipeline
from pipeline import LCD, load_texts, perplexity, BASELINE

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-3B"
QUANT_REPO = "nrmlthms/llama3.2-3b-int3-gptq"

# Llama worst-error depth band from layerwise heatmap analysis
TARGET_LAYERS = list(range(7, 15))   # layers 7–14 inclusive

EXPERIMENT_LANGUAGES = ["arabic", "japanese", "chinese", "korean",
                        "hindi", "russian", "french", "spanish"]

CONFIGS = [
    # (name,          rank, target_layers)
    ("uniform_r2",   2,    None),
    ("targeted_r4",  4,    TARGET_LAYERS),
    ("targeted_r2",  2,    TARGET_LAYERS),
]


# ─────────────────────────────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env"):
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
_load_dotenv()


def resolve_quant_dir(args) -> str:
    if args.load_quantized_dir:
        print(f"  [quant] Using specified dir: {args.load_quantized_dir}")
        return args.load_quantized_dir
    slug = MODEL_ID.replace("/", "_")
    local = f"lcd_results/{slug}/quantized"
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        print(f"  [quant] Using local cache: {local}")
        return local
    from huggingface_hub import snapshot_download
    print(f"  [quant] Downloading {QUANT_REPO} ...")
    os.makedirs(local, exist_ok=True)
    snapshot_download(repo_id=QUANT_REPO, local_dir=local,
                      local_dir_use_symlinks=False,
                      token=os.environ.get("HF_TOKEN"))
    return local


def run_config(quant_model, tokenizer, name, rank, target_indices,
               train_texts, eval_texts, fp16_ppls, int3_ppls, args):
    label = f"rank={rank} target={'all' if target_indices is None else f'layers {target_indices[0]}-{target_indices[-1]}'}"
    print(f"\n  {'='*68}\n  CONFIG: {name}  ({label})\n  {'='*68}")

    lcd = LCD(quant_model, rank=rank,
              layer_strategy="all" if target_indices is None else "all",
              target_layer_indices=target_indices)

    corrected_ppls = {}
    for lang in EXPERIMENT_LANGUAGES:
        lcd.train_lang(
            lang, train_texts[lang], tokenizer,
            args.device, args.steps, args.lr, args.max_seq_len,
            weight_decay=0.01, warmup_ratio=0.1, grad_accum=1,
        )
        ppl = lcd.ppl(lang, eval_texts[lang], tokenizer,
                      args.device, args.max_seq_len)
        corrected_ppls[lang] = ppl
        fp, q = fp16_ppls[lang], int3_ppls[lang]
        rec = (1 - (ppl - fp) / max(q - fp, 1e-8)) * 100
        print(f"    {lang:<10} corrected={ppl:.3f}  recovery={rec:.1f}%")

    lcd.remove()
    del lcd
    gc.collect()
    torch.cuda.empty_cache()

    recovery = {
        lang: (1 - (corrected_ppls[lang] - fp16_ppls[lang]) /
               max(int3_ppls[lang] - fp16_ppls[lang], 1e-8)) * 100
        for lang in EXPERIMENT_LANGUAGES
    }
    avg = float(np.mean(list(recovery.values())))
    print(f"\n  >> {name}: avg recovery = {avg:.1f}%  "
          f"(per-lang: " + " ".join(f"{l[:2]}={recovery[l]:.1f}%" for l in EXPERIMENT_LANGUAGES) + ")")
    return {"corrected_ppl": corrected_ppls, "recovery_pct": recovery, "avg_recovery": avg}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device",            default="cuda:0")
    p.add_argument("--load-quantized-dir", default=None)
    p.add_argument("--rank",              default=2,    type=int,
                   help="Rank for uniform_r2 config (overridden per-config for others)")
    p.add_argument("--steps",             default=500,  type=int)
    p.add_argument("--lr",                default=5e-4, type=float)
    p.add_argument("--max-seq-len",       default=512,  type=int)
    p.add_argument("--num-train-samples", default=256,  type=int)
    p.add_argument("--num-fp16-samples",  default=32,   type=int)
    p.add_argument("--num-eval-samples",  default=32,   type=int)
    p.add_argument("--seed",              default=42,   type=int)
    p.add_argument("--cache-dir",         default=None)
    p.add_argument("--target-layers",     default="7-14",
                   help="Layer index range for targeted configs, e.g. '7-14'")
    p.add_argument("--output",            default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    # Parse target layers from CLI (allows overriding the default 7-14)
    lo, hi = map(int, args.target_layers.split("-"))
    target_layers = list(range(lo, hi + 1))
    configs = [
        ("uniform_r2",  2, None),
        ("targeted_r4", 4, target_layers),
        ("targeted_r2", 2, target_layers),
    ]

    slug = MODEL_ID.replace("/", "_")
    out_path = Path(args.output) if args.output \
               else Path(f"lcd_results/{slug}/exp_layer_targeted_lcd.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Layer-Targeted LCD — {MODEL_ID}")
    print(f"{'='*70}")
    print(f"  Target layers : {target_layers[0]}–{target_layers[-1]}")
    print(f"  Configs       : {[c[0] for c in configs]}")
    print(f"  Languages     : {EXPERIMENT_LANGUAGES}")
    print(f"  Device        : {args.device}")
    print(f"  Output        : {out_path}\n")

    pipeline.LANGUAGES = ["english"] + EXPERIMENT_LANGUAGES

    print("  [1/4] Loading tokenizer + FP16 model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto",
        device_map=args.device, trust_remote_code=True,
    ).eval()

    print("\n  [2/4] Loading data ...", flush=True)
    kw = {"seed": args.seed, "cache_dir": args.cache_dir}
    fp16_texts  = {l: load_texts(l, args.num_fp16_samples, **kw)
                   for l in pipeline.LANGUAGES}
    train_texts = {l: load_texts(l, args.num_train_samples, seed=args.seed + 3,
                                  cache_dir=args.cache_dir)
                   for l in pipeline.LANGUAGES}
    eval_texts  = {l: load_texts(l, args.num_eval_samples,  seed=args.seed + 4,
                                  cache_dir=args.cache_dir)
                   for l in pipeline.LANGUAGES}

    print("\n  [3/4] Measuring FP16 baselines ...", flush=True)
    fp16_ppls = {}
    for lang in pipeline.LANGUAGES:
        fp16_ppls[lang] = perplexity(
            fp16_model, tokenizer, fp16_texts[lang],
            args.device, args.max_seq_len, desc=f"fp16 {lang}",
        )
        print(f"    {lang:<10} fp16_ppl = {fp16_ppls[lang]:.3f}")

    del fp16_model
    gc.collect()
    torch.cuda.empty_cache()

    quant_dir = resolve_quant_dir(args)
    print(f"\n  Loading INT3 quantized model from {quant_dir} ...", flush=True)
    quant_model = AutoModelForCausalLM.from_pretrained(
        quant_dir, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True,
    ).eval()

    print("\n  Measuring INT3 baselines ...", flush=True)
    int3_ppls = {}
    for lang in pipeline.LANGUAGES:
        int3_ppls[lang] = perplexity(
            quant_model, tokenizer, fp16_texts[lang],
            args.device, args.max_seq_len, desc=f"int3 {lang}",
        )
        print(f"    {lang:<10} int3_ppl = {int3_ppls[lang]:.3f}  "
              f"degradation = {int3_ppls[lang]/fp16_ppls[lang]:.2f}x")

    print(f"\n  [4/4] Running {len(configs)} targeted configs ...")
    results = {}
    for name, rank, tgt in configs:
        cfg_t0 = time.time()
        res = run_config(quant_model, tokenizer, name, rank, tgt,
                         train_texts, eval_texts, fp16_ppls, int3_ppls, args)
        res["wall_time_sec"] = time.time() - cfg_t0
        results[name] = res

    # ── Comparison table ────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  COMPARISON (perplexity recovery %)\n{'='*70}")
    col_langs = EXPERIMENT_LANGUAGES
    header = f"  {'config':<14} {'avg':>6}  " + "  ".join(f"{l[:3]:>6}" for l in col_langs)
    print(header)
    for name, _, _ in configs:
        rec = results[name]["recovery_pct"]
        avg = results[name]["avg_recovery"]
        row = f"  {name:<14} {avg:>5.1f}%  " + "  ".join(f"{rec[l]:>5.1f}%" for l in col_langs)
        print(row)

    winner = max(results, key=lambda n: results[n]["avg_recovery"])
    print(f"\n  Winner: {winner}  (avg = {results[winner]['avg_recovery']:.1f}%)")

    out = {
        "model":          MODEL_ID,
        "target_layers":  target_layers,
        "languages":      EXPERIMENT_LANGUAGES,
        "fp16_ppl":       fp16_ppls,
        "int3_ppl":       int3_ppls,
        "by_config":      results,
        "winner":         winner,
        "wall_time_min":  (time.time() - t0) / 60,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results → {out_path}")
    print(f"  Wall time: {(time.time() - t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
