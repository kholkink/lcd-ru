#!/usr/bin/env python3
"""
Experiment: Language-Agnostic Baseline
=======================================
Tests whether language-conditioning is actually necessary by training
a single rank-2 LoRA correction on *mixed* multilingual data and
comparing it against the per-language LCD corrections.

If the mixed correction underperforms per-language corrections,
language-conditioning is justified. If it matches, we have a simpler
method (still a good result).

Runs on: Qwen2.5-3B (INT3 GPTQ), 4 languages (Arabic, Japanese, Korean, French).
Requires: quantized model from the main pipeline at lcd_results/qwen2.5-3b/quantized/

Usage:
    python exp_agnostic_baseline.py
    python exp_agnostic_baseline.py --device cuda --steps 500
    python exp_agnostic_baseline.py --fast   # quick smoke test
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import shared infrastructure from main pipeline
import pipeline
from pipeline import (
    LCD, load_texts, perplexity, SEP,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_LANGUAGES = ["arabic", "japanese", "korean", "french"]
MODEL_NAME = "qwen2.5-3b"
MODEL_HF_ID = "Qwen/Qwen2.5-3B"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--rank",              default=2,    type=int)
    p.add_argument("--steps",             default=500,  type=int)
    p.add_argument("--lr",                default=5e-4, type=float)
    p.add_argument("--max-seq-len",       default=512,  type=int)
    p.add_argument("--num-train-samples", default=256,  type=int,
                   help="Samples per language for mixed training set")
    p.add_argument("--num-eval-samples",  default=32,   type=int)
    p.add_argument("--seed",              default=42,   type=int)
    p.add_argument("--cache-dir",         default=None)
    p.add_argument("--quantized-dir",
                   default=f"lcd_results/{MODEL_NAME}/quantized",
                   help="Path to pre-quantized INT3 model")
    p.add_argument("--lcd-results",
                   default=f"lcd_results/{MODEL_NAME}/lcd_results.json",
                   help="Per-language LCD results for comparison")
    p.add_argument("--output",
                   default=f"lcd_results/{MODEL_NAME}/exp_agnostic_baseline.json")
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke test: fewer samples and steps")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    if args.fast:
        args.num_train_samples = 16
        args.num_eval_samples  = 8
        args.steps             = 50
        print("  [--fast] smoke-test mode\n")

    # ── Load per-language LCD results for comparison ─────────────────────────
    lcd_results_path = Path(args.lcd_results)
    per_lang_results = {}
    if lcd_results_path.exists():
        lcd_data = json.loads(lcd_results_path.read_text())
        per_lang_results = lcd_data.get("results", {})
        print(f"  Loaded per-language LCD results from {lcd_results_path}")
    else:
        print(f"  WARNING: No per-language LCD results at {lcd_results_path}")
        print(f"  Comparison column will show n/a. Run main pipeline first.")

    # ── Load quantized model ─────────────────────────────────────────────────
    qdir = Path(args.quantized_dir)
    if not qdir.exists():
        print(f"  ERROR: Quantized model not found at {qdir}")
        print(f"  Run the main pipeline first: python run_lcd.py --models {MODEL_NAME}")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Loading quantized model from {qdir}")
    print(f"{'─'*60}")
    tokenizer = AutoTokenizer.from_pretrained(str(qdir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(qdir), torch_dtype="auto",
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Loading data for {', '.join(EXPERIMENT_LANGUAGES)}")
    print(f"{'─'*60}")

    # Use distinct seeds to avoid overlap with main pipeline data
    train_texts, eval_texts = {}, {}
    for lang in EXPERIMENT_LANGUAGES:
        train_texts[lang] = load_texts(lang, args.num_train_samples,
                                       seed=args.seed + 10, cache_dir=args.cache_dir)
        eval_texts[lang]  = load_texts(lang, args.num_eval_samples,
                                       seed=args.seed + 11, cache_dir=args.cache_dir)

    # ── INT3 baselines on eval set ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Measuring INT3 perplexity baselines")
    print(f"{'─'*60}")
    int3_ppls = {}
    for lang in EXPERIMENT_LANGUAGES:
        int3_ppls[lang] = perplexity(model, tokenizer, eval_texts[lang],
                                     args.device, args.max_seq_len,
                                     desc=f"int3 {lang}")
        print(f"    {lang:<12} INT3 ppl = {int3_ppls[lang]:.2f}")

    # FP16 baselines from existing results (avoids reloading FP16 model)
    fp16_ppls = {}
    for lang in EXPERIMENT_LANGUAGES:
        if lang in per_lang_results:
            fp16_ppls[lang] = per_lang_results[lang]["fp16_ppl"]
        else:
            print(f"  WARNING: No FP16 ppl for {lang} — recovery will be approximate")
            fp16_ppls[lang] = int3_ppls[lang]

    # ══════════════════════════════════════════════════════════════════════════
    # AGNOSTIC BASELINE: single correction on mixed data
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Language-Agnostic Baseline")
    print(f"  Single rank-{args.rank} correction on mixed data")
    print(f"  Languages: {', '.join(EXPERIMENT_LANGUAGES)}")
    print(f"  Steps: {args.steps}  |  LR: {args.lr}")
    print(f"{'='*60}")

    # Build mixed training set (equal samples per language, shuffled)
    mixed_train = []
    for lang in EXPERIMENT_LANGUAGES:
        mixed_train.extend(train_texts[lang])
    random.shuffle(mixed_train)
    print(f"\n  Mixed training set: {len(mixed_train)} samples "
          f"({args.num_train_samples} per language × {len(EXPERIMENT_LANGUAGES)} languages)")

    # Temporarily set LANGUAGES to a single slot for the LCD constructor
    saved_languages = pipeline.LANGUAGES[:]
    pipeline.LANGUAGES = ["universal"]

    lcd = LCD(model, rank=args.rank)
    lcd.train_lang("universal", mixed_train, tokenizer,
                   args.device, args.steps, args.lr, args.max_seq_len)

    # Evaluate per-language using the single universal correction
    agnostic_ppls = {}
    lcd.lang_idx = 0
    lcd.freeze_all()
    for lang in EXPERIMENT_LANGUAGES:
        agnostic_ppls[lang] = perplexity(model, tokenizer, eval_texts[lang],
                                         args.device, args.max_seq_len,
                                         desc=f"agnostic {lang}")

    lcd.remove()
    pipeline.LANGUAGES = saved_languages

    # ══════════════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  COMPARISON — Per-Language LCD vs Language-Agnostic Baseline")
    print(f"  Model: {MODEL_HF_ID} (INT3 GPTQ)  |  Rank: {args.rank}")
    print(SEP)
    print(f"\n  {'Language':<12} {'FP16':>7} {'INT3':>7} {'Agnostic':>10} {'Per-Lang':>10} "
          f"{'Agn.Rec%':>9} {'PL.Rec%':>9} {'Delta':>7}")
    print("  " + "─" * 78)

    results = {}
    agn_recs, pl_recs = [], []

    for lang in EXPERIMENT_LANGUAGES:
        fp  = fp16_ppls[lang]
        q   = int3_ppls[lang]
        agn = agnostic_ppls[lang]
        gap = q - fp

        agn_rec = (1 - (agn - fp) / max(gap, 1e-8)) * 100 if gap > 0 else 0.0
        agn_recs.append(agn_rec)

        pl_ppl = per_lang_results.get(lang, {}).get("corrected_ppl")
        pl_rec = per_lang_results.get(lang, {}).get("recovery_pct")
        if pl_rec is not None:
            pl_recs.append(pl_rec)

        pl_str  = f"{pl_ppl:>10.2f}" if pl_ppl else f"{'n/a':>10}"
        plr_str = f"{pl_rec:>8.1f}%" if pl_rec is not None else f"{'n/a':>9}"
        delta   = f"{agn_rec - pl_rec:>+6.1f}%" if pl_rec is not None else f"{'—':>7}"

        print(f"  {lang:<12} {fp:>7.2f} {q:>7.2f} {agn:>10.2f} {pl_str} "
              f"{agn_rec:>8.1f}% {plr_str} {delta}")

        results[lang] = {
            "fp16_ppl": fp,
            "int3_ppl": q,
            "agnostic_ppl": agn,
            "agnostic_recovery_pct": round(agn_rec, 2),
            "per_lang_ppl": pl_ppl,
            "per_lang_recovery_pct": pl_rec,
        }

    avg_agn = np.mean(agn_recs)
    avg_pl  = np.mean(pl_recs) if pl_recs else float("nan")
    diff    = avg_agn - avg_pl if pl_recs else float("nan")

    print(f"\n  {'Average':<12} {'':>7} {'':>7} {'':>10} {'':>10} "
          f"{avg_agn:>8.1f}%", end="")
    if pl_recs:
        print(f" {avg_pl:>8.1f}% {diff:>+6.1f}%")
    else:
        print(f" {'n/a':>9} {'—':>7}")

    print(f"\n{SEP}")
    if not math.isnan(diff):
        if diff < -5:
            print(f"  CONDITIONING MATTERS — per-language corrections outperform "
                  f"agnostic by {abs(diff):.1f}pp")
            print(f"  The language-conditional design is justified.")
        elif diff > 5:
            print(f"  SURPRISE — agnostic baseline outperforms per-language by {diff:.1f}pp!")
            print(f"  Consider whether a simpler universal correction suffices.")
        else:
            print(f"  CLOSE CALL — difference is {diff:+.1f}pp (within 5pp)")
            print(f"  Per-language conditioning gives marginal benefit at this scale.")
    else:
        print(f"  Agnostic recovery: {avg_agn:.1f}%")
        print(f"  Run the main pipeline for per-language comparison.")
    print(SEP)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "agnostic_baseline",
        "model": MODEL_HF_ID,
        "rank": args.rank,
        "steps": args.steps,
        "lr": args.lr,
        "languages": EXPERIMENT_LANGUAGES,
        "results": results,
        "summary": {
            "avg_agnostic_recovery_pct": round(avg_agn, 2),
            "avg_per_lang_recovery_pct": round(avg_pl, 2) if pl_recs else None,
            "difference_pp": round(diff, 2) if not math.isnan(diff) else None,
        },
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved → {output_path}")

    elapsed = time.time() - t0
    print(f"  Wall time: {elapsed/60:.1f} min\n")


if __name__ == "__main__":
    main()
