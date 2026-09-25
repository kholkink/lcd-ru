#!/usr/bin/env python3
"""
Experiment: Rank Ablation
==========================
Tests rank ∈ {1, 2, 4} to justify the rank=2 default and characterize
the capacity–performance trade-off.

Expected story:
  - rank=1: underperforms (not enough capacity for distant languages)
  - rank=2: sweet spot
  - rank=4: diminishing returns (marginal improvement, 2× params)

Runs on: Qwen2.5-3B (INT3 GPTQ), 2 languages (Arabic, Japanese).

Usage:
    python exp_rank_ablation.py
    python exp_rank_ablation.py --device cuda
    python exp_rank_ablation.py --fast   # quick smoke test
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import shared infrastructure from main pipeline
import pipeline
from pipeline import (
    LCD, load_texts, perplexity, BASELINE, SEP,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_LANGUAGES = ["arabic", "japanese"]
RANKS = [1, 2, 4]
MODEL_NAME = "qwen2.5-3b"
MODEL_HF_ID = "Qwen/Qwen2.5-3B"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--steps",             default=500,  type=int)
    p.add_argument("--lr",                default=5e-4, type=float)
    p.add_argument("--max-seq-len",       default=512,  type=int)
    p.add_argument("--num-train-samples", default=256,  type=int)
    p.add_argument("--num-eval-samples",  default=32,   type=int)
    p.add_argument("--seed",              default=42,   type=int)
    p.add_argument("--cache-dir",         default=None)
    p.add_argument("--quantized-dir",
                   default=f"lcd_results/{MODEL_NAME}/quantized",
                   help="Path to pre-quantized INT3 model")
    p.add_argument("--lcd-results",
                   default=f"lcd_results/{MODEL_NAME}/lcd_results.json",
                   help="Main LCD results for FP16 baselines")
    p.add_argument("--output",
                   default=f"lcd_results/{MODEL_NAME}/exp_rank_ablation.json")
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

    # ── Load FP16 baselines from existing results ────────────────────────────
    lcd_results_path = Path(args.lcd_results)
    per_lang_results = {}
    if lcd_results_path.exists():
        lcd_data = json.loads(lcd_results_path.read_text())
        per_lang_results = lcd_data.get("results", {})
        print(f"  Loaded baselines from {lcd_results_path}")
    else:
        print(f"  WARNING: No main LCD results at {lcd_results_path}")

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
    # Include english in LANGUAGES so LCD indices work correctly, but only
    # train/evaluate on the experiment languages
    lcd_languages = [BASELINE] + EXPERIMENT_LANGUAGES

    print(f"\n{'─'*60}")
    print(f"  Loading data for {', '.join(EXPERIMENT_LANGUAGES)}")
    print(f"{'─'*60}")
    train_texts, eval_texts = {}, {}
    for lang in EXPERIMENT_LANGUAGES:
        train_texts[lang] = load_texts(lang, args.num_train_samples,
                                       seed=args.seed + 20, cache_dir=args.cache_dir)
        eval_texts[lang]  = load_texts(lang, args.num_eval_samples,
                                       seed=args.seed + 21, cache_dir=args.cache_dir)

    # ── INT3 baselines ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Measuring INT3 perplexity baselines")
    print(f"{'─'*60}")
    int3_ppls = {}
    for lang in EXPERIMENT_LANGUAGES:
        int3_ppls[lang] = perplexity(model, tokenizer, eval_texts[lang],
                                     args.device, args.max_seq_len,
                                     desc=f"int3 {lang}")
        print(f"    {lang:<12} INT3 ppl = {int3_ppls[lang]:.2f}")

    # FP16 baselines from existing results
    fp16_ppls = {}
    for lang in EXPERIMENT_LANGUAGES:
        if lang in per_lang_results:
            fp16_ppls[lang] = per_lang_results[lang]["fp16_ppl"]
        else:
            print(f"  WARNING: No FP16 ppl for {lang}")
            fp16_ppls[lang] = int3_ppls[lang]

    # ══════════════════════════════════════════════════════════════════════════
    # RANK ABLATION: train per-language corrections at each rank
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Rank Ablation")
    print(f"  Ranks: {RANKS}")
    print(f"  Languages: {', '.join(EXPERIMENT_LANGUAGES)}")
    print(f"  Steps: {args.steps}  |  LR: {args.lr}")
    print(f"{'='*60}")

    # Set LANGUAGES for LCD constructor
    saved_languages = pipeline.LANGUAGES[:]
    pipeline.LANGUAGES = lcd_languages

    rank_results = {}   # rank → {lang → {ppl, recovery}}
    param_counts = {}   # rank → params_per_language

    for rank in RANKS:
        print(f"\n  {'─'*50}")
        print(f"  RANK = {rank}")
        print(f"  {'─'*50}")

        lcd = LCD(model, rank=rank)

        # Count params per language for this rank
        n_per_lang = sum(
            sum(p.numel() for p in c.A) // len(lcd_languages) +
            sum(p.numel() for p in c.B) // len(lcd_languages)
            for c in lcd.corrections
        )
        param_counts[rank] = n_per_lang

        rank_results[rank] = {}
        for lang in EXPERIMENT_LANGUAGES:
            print(f"\n    ── {lang.upper()} (rank={rank}) ──")
            lcd.train_lang(lang, train_texts[lang], tokenizer,
                           args.device, args.steps, args.lr, args.max_seq_len)
            ppl = lcd.ppl(lang, eval_texts[lang], tokenizer,
                          args.device, args.max_seq_len)

            fp  = fp16_ppls[lang]
            q   = int3_ppls[lang]
            gap = q - fp
            rec = (1 - (ppl - fp) / max(gap, 1e-8)) * 100 if gap > 0 else 0.0

            rank_results[rank][lang] = {"ppl": ppl, "recovery_pct": rec}
            print(f"    INT3={q:.2f}  LCD={ppl:.2f}  FP16={fp:.2f}  recovery={rec:.1f}%")

        lcd.remove()
        del lcd
        gc.collect()
        torch.cuda.empty_cache()

    pipeline.LANGUAGES = saved_languages

    # ══════════════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  RANK ABLATION — Per-Language Recovery (%)")
    print(f"  Model: {MODEL_HF_ID} (INT3 GPTQ)")
    print(f"  Steps: {args.steps}  |  LR: {args.lr}")
    print(SEP)

    # Header
    header = f"  {'Rank':>6} {'Params':>10}"
    for lang in EXPERIMENT_LANGUAGES:
        header += f"  {lang:>12}"
    header += f"  {'Average':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    # Rows
    best_rank, best_avg = None, -float("inf")
    for rank in RANKS:
        recs = []
        row = f"  {rank:>6} {param_counts[rank]:>10,}"
        for lang in EXPERIMENT_LANGUAGES:
            rec = rank_results[rank][lang]["recovery_pct"]
            recs.append(rec)
            row += f"  {rec:>11.1f}%"
        avg = np.mean(recs)
        row += f"  {avg:>9.1f}%"
        if avg > best_avg:
            best_avg = avg
            best_rank = rank
        print(row)

    # Perplexity detail table
    print(f"\n  {'Rank':>6}", end="")
    for lang in EXPERIMENT_LANGUAGES:
        print(f"  {lang + ' ppl':>12}", end="")
    print()
    print("  " + "─" * (8 + 14 * len(EXPERIMENT_LANGUAGES)))
    print(f"  {'FP16':>6}", end="")
    for lang in EXPERIMENT_LANGUAGES:
        print(f"  {fp16_ppls[lang]:>12.2f}", end="")
    print()
    print(f"  {'INT3':>6}", end="")
    for lang in EXPERIMENT_LANGUAGES:
        print(f"  {int3_ppls[lang]:>12.2f}", end="")
    print()
    for rank in RANKS:
        print(f"  {f'r={rank}':>6}", end="")
        for lang in EXPERIMENT_LANGUAGES:
            print(f"  {rank_results[rank][lang]['ppl']:>12.2f}", end="")
        print()

    print(f"\n{SEP}")
    print(f"  BEST: rank={best_rank} ({best_avg:.1f}% average recovery)")

    # Interpret the story
    recs_by_rank = {r: np.mean([rank_results[r][l]["recovery_pct"]
                                for l in EXPERIMENT_LANGUAGES]) for r in RANKS}
    r1, r2, r4 = recs_by_rank[1], recs_by_rank[2], recs_by_rank[4]
    if r1 < r2 and r4 - r2 < (r2 - r1) * 0.5:
        print(f"  Rank=2 is the sweet spot: rank=1 underperforms by {r2-r1:.1f}pp, "
              f"rank=4 gains only {r4-r2:+.1f}pp at 2x params.")
    elif r4 > r2 + 5:
        print(f"  Higher rank helps substantially (+{r4-r2:.1f}pp at rank=4). "
              f"Consider rank=4 for production.")
    else:
        print(f"  Rank=1: {r1:.1f}%  |  Rank=2: {r2:.1f}%  |  Rank=4: {r4:.1f}%")
    print(SEP)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "rank_ablation",
        "model": MODEL_HF_ID,
        "steps": args.steps,
        "lr": args.lr,
        "languages": EXPERIMENT_LANGUAGES,
        "ranks": RANKS,
        "param_counts_per_language": {str(r): param_counts[r] for r in RANKS},
        "results": {
            str(rank): {
                lang: {
                    "corrected_ppl": rank_results[rank][lang]["ppl"],
                    "recovery_pct": round(rank_results[rank][lang]["recovery_pct"], 2),
                    "fp16_ppl": fp16_ppls[lang],
                    "int3_ppl": int3_ppls[lang],
                }
                for lang in EXPERIMENT_LANGUAGES
            }
            for rank in RANKS
        },
        "summary": {
            str(rank): round(np.mean([rank_results[rank][l]["recovery_pct"]
                                      for l in EXPERIMENT_LANGUAGES]), 2)
            for rank in RANKS
        },
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved → {output_path}")

    elapsed = time.time() - t0
    print(f"  Wall time: {elapsed/60:.1f} min\n")


if __name__ == "__main__":
    main()
