#!/usr/bin/env python3
"""
Experiment: Layer-Strategy Validation of the Depth-Asymmetry Hypothesis
========================================================================
Tests the central claim of §5.6 ("Diagnosing the Disconnect"):

    Per-layer LCD corrections placed at the model's worst-error depth band
    should recover more than the same parameter budget placed elsewhere.

For each model, we compare three LCD configurations under identical rank,
training data, and step count:
  1. all          (baseline — every linear layer corrected)
  2. bottom_half  (only layers in the first half of the network)
  3. top_half     (only layers in the second half)

Predictions from the layerwise heatmaps:
  - Qwen2.5-3B  worst layers cluster at indices 18–30  →  top_half should win.
  - Llama-3.2-3B worst layers cluster at indices 7–14  →  bottom_half should win.

If both predictions hold, the depth-asymmetry hypothesis is confirmed with
crossed evidence across two model families.

Quantization is performed ONCE per model invocation, then each strategy is
trained and evaluated on the cached quantized weights — avoiding 3×
redundant GPTQ runs.

Usage:
    # Run both models in parallel on separate GPUs
    python exp_layer_strategy.py --model Qwen/Qwen2.5-3B   --device cuda:0
    python exp_layer_strategy.py --model meta-llama/Llama-3.2-3B --device cuda:1
"""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import pipeline
from pipeline import LCD, load_texts, perplexity, BASELINE


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = ["all", "bottom_half", "top_half"]

# Two languages where the disconnect is most informative:
#  - Arabic: largest non-EN error in both models
#  - Japanese: second-largest, distinct script family
EXPERIMENT_LANGUAGES = ["arabic", "japanese", "chinese", "korean"]

QUANT_REPOS = {
    "Qwen/Qwen2.5-3B":         "nrmlthms/qwen2.5-3b-int3-gptq",
    "meta-llama/Llama-3.2-3B": "nrmlthms/llama3.2-3b-int3-gptq",
}


# ─────────────────────────────────────────────────────────────────────────────
# .env loader (minimal — same as exp_layerwise_error.py)
# ─────────────────────────────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env"):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
_load_dotenv()


def resolve_quant_dir(model: str) -> str:
    slug = model.replace("/", "_")
    local_path = f"lcd_results/{slug}/quantized"
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        print(f"  [quant] Using local cache: {local_path}")
        return local_path
    if model not in QUANT_REPOS:
        raise ValueError(f"No HF repo registered for {model}")
    repo = QUANT_REPOS[model]
    print(f"  [quant] Downloading from HF: {repo}")
    os.makedirs(local_path, exist_ok=True)
    snapshot_download(
        repo_id=repo, local_dir=local_path,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  [quant] Downloaded → {local_path}")
    return local_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy LCD train + evaluate
# ─────────────────────────────────────────────────────────────────────────────
def run_strategy(model, tokenizer, strategy: str,
                 train_texts, eval_texts, args, non_en):
    """Train LCD with the given layer_strategy and return perplexity dict."""
    print(f"\n  {'='*70}\n  STRATEGY: {strategy}\n  {'='*70}")
    lcd = LCD(model, rank=args.rank, layer_strategy=strategy)

    ppls = {}
    for lang in non_en:
        lcd.train_lang(
            lang, train_texts[lang], tokenizer,
            args.device, args.steps, args.lr, args.max_seq_len,
            weight_decay=0.01, warmup_ratio=0.1, grad_accum=1,
        )
        ppls[lang] = lcd.ppl(lang, eval_texts[lang], tokenizer,
                             args.device, args.max_seq_len)
        print(f"    [{strategy}] {lang}: corrected_ppl = {ppls[lang]:.3f}")

    lcd.remove()
    del lcd
    gc.collect()
    torch.cuda.empty_cache()
    return ppls


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",           default="Qwen/Qwen2.5-3B")
    p.add_argument("--device",          default="cuda:0")
    p.add_argument("--rank",            default=2,    type=int)
    p.add_argument("--steps",           default=500,  type=int)
    p.add_argument("--lr",              default=5e-4, type=float)
    p.add_argument("--max-seq-len",     default=512,  type=int)
    p.add_argument("--num-train-samples", default=256, type=int)
    p.add_argument("--num-fp16-samples",  default=32,  type=int)
    p.add_argument("--num-eval-samples",  default=32,  type=int)
    p.add_argument("--seed",            default=42,   type=int)
    p.add_argument("--cache-dir",       default=None)
    p.add_argument("--output",          default=None,
                   help="Output JSON path (default: lcd_results/<slug>/exp_layer_strategy.json)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    slug = args.model.replace("/", "_")
    out_path = Path(args.output) if args.output \
               else Path(f"lcd_results/{slug}/exp_layer_strategy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Layer-Strategy Experiment — {args.model}")
    print(f"{'='*70}")
    print(f"  Device     : {args.device}")
    print(f"  Strategies : {STRATEGIES}")
    print(f"  Languages  : {EXPERIMENT_LANGUAGES}")
    print(f"  Rank       : {args.rank},  Steps : {args.steps}")
    print(f"  Output     : {out_path}\n")

    # Override pipeline's global LANGUAGES so that LCD's per-language adapter slots
    # match the languages we evaluate.
    pipeline.LANGUAGES = ["english"] + EXPERIMENT_LANGUAGES

    quant_dir = resolve_quant_dir(args.model)

    print("\n  [1/4] Loading tokenizer + FP16 model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map=args.device, trust_remote_code=True,
    ).eval()

    # FP16 baselines (per language)
    kw = {"seed": args.seed, "cache_dir": args.cache_dir}
    print("\n  [2/4] Loading data ...", flush=True)
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

    # Free FP16 model — we need VRAM for the quantized model + LCD adapters
    del fp16_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load INT3 quantized model
    print(f"\n  Loading INT3 quantized model from {quant_dir} ...", flush=True)
    quant_model = AutoModelForCausalLM.from_pretrained(
        quant_dir, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True,
    ).eval()

    # INT3 baseline (uncorrected)
    print("\n  Measuring INT3 baselines (uncorrected) ...", flush=True)
    int3_ppls = {}
    for lang in pipeline.LANGUAGES:
        int3_ppls[lang] = perplexity(
            quant_model, tokenizer, fp16_texts[lang],
            args.device, args.max_seq_len, desc=f"int3 {lang}",
        )
        print(f"    {lang:<10} int3_ppl = {int3_ppls[lang]:.3f}  "
              f"degradation = {int3_ppls[lang]/fp16_ppls[lang]:.2f}x")

    # Run each strategy
    print(f"\n  [4/4] Running LCD with each layer strategy ...")
    results = {}
    for strategy in STRATEGIES:
        strategy_t0 = time.time()
        ppls = run_strategy(
            quant_model, tokenizer, strategy,
            train_texts, eval_texts, args, EXPERIMENT_LANGUAGES,
        )
        results[strategy] = {
            "perplexity_corrected": ppls,
            "recovery_pct": {
                lang: (1 - (ppls[lang] - fp16_ppls[lang]) /
                       max(int3_ppls[lang] - fp16_ppls[lang], 1e-8)) * 100
                for lang in EXPERIMENT_LANGUAGES
            },
            "wall_time_sec": time.time() - strategy_t0,
        }
        avg_rec = np.mean(list(results[strategy]["recovery_pct"].values()))
        print(f"\n  >> {strategy}: avg recovery = {avg_rec:.1f}% "
              f"(per-lang: " +
              " ".join(f"{l}={results[strategy]['recovery_pct'][l]:.1f}%"
                       for l in EXPERIMENT_LANGUAGES) + ")")

    # ── Final comparison table ──────────────────────────────────────────
    print(f"\n{'='*70}\n  COMPARISON\n{'='*70}")
    print(f"  {'strategy':<14} {'avg':>8}  " +
          "  ".join(f"{l[:3]:>8}" for l in EXPERIMENT_LANGUAGES))
    for strategy in STRATEGIES:
        rec = results[strategy]["recovery_pct"]
        avg = np.mean(list(rec.values()))
        print(f"  {strategy:<14} {avg:>7.1f}%  " +
              "  ".join(f"{rec[l]:>7.1f}%" for l in EXPERIMENT_LANGUAGES))

    # Identify the winner
    avg_recoveries = {
        s: np.mean(list(results[s]["recovery_pct"].values())) for s in STRATEGIES
    }
    winner = max(avg_recoveries, key=avg_recoveries.get)
    print(f"\n  Winner: {winner}  (avg = {avg_recoveries[winner]:.1f}%)")

    out = {
        "model": args.model,
        "rank": args.rank,
        "steps": args.steps,
        "languages": EXPERIMENT_LANGUAGES,
        "fp16_ppl": fp16_ppls,
        "int3_ppl": int3_ppls,
        "by_strategy": results,
        "winner": winner,
        "winner_avg_recovery_pct": avg_recoveries[winner],
        "wall_time_min": (time.time() - t0) / 60,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results written → {out_path}")
    print(f"  Wall time: {(time.time() - t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
