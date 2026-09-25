#!/usr/bin/env python3
"""
LCD Evaluation Runner
======================
Run perplexity + GlobalMMLU evaluations across FP16 / INT3 / LCD conditions.
Models are loaded from HuggingFace Hub.

Usage:
    # Both models, convention-based IDs
    python run_evals.py --hf-user your-username

    # Single model, explicit IDs
    python run_evals.py \\
        --fp16-model Qwen/Qwen2.5-3B \\
        --int3-model your-username/qwen2.5-3b-int3-gptq \\
        --lcd-model  your-username/qwen2.5-3b-int3-gptq-lcd

    # Select models / conditions
    python run_evals.py --hf-user your-username --models qwen2.5-3b
    python run_evals.py --hf-user your-username --conditions fp16,int3

    # Subsample MMLU for quick test (default: full dataset)
    python run_evals.py --hf-user your-username --num-mmlu-samples 100
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import (
    evaluate_global_mmlu,
    evaluate_perplexity,
    load_lcd_corrections,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "qwen2.5-3b": {
        "fp16": "Qwen/Qwen2.5-3B",
        "int3_suffix": "qwen2.5-3b-int3-gptq",
        "lcd_suffix":  "qwen2.5-3b-int3-gptq-lcd",
    },
    "llama3.2-3b": {
        "fp16": "meta-llama/Llama-3.2-3B",
        "int3_suffix": "llama3.2-3b-int3-gptq",
        "lcd_suffix":  "llama3.2-3b-int3-gptq-lcd",
    },
}

EVAL_LANGUAGES = [
    "english", "arabic", "japanese", "chinese", "hindi",
    "russian", "french", "spanish", "korean",
]

ALL_CONDITIONS = ["fp16", "int3", "lcd"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_id, device):
    print(f"\n  Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto",
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    actual_device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.2f}B params, dtype={next(model.parameters()).dtype}, "
          f"device={actual_device}")
    return model, tokenizer


def cleanup_gpu():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download_lcd_corrections(lcd_model: str) -> str:
    """Find lcd_corrections.pt — local path first, then HF download."""
    local_pt = Path(lcd_model) / "lcd_corrections.pt"
    if local_pt.exists():
        print(f"  LCD corrections found at {local_pt}")
        return str(local_pt)
    # Fall back to HF Hub download
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=lcd_model, filename="lcd_corrections.pt")
    print(f"  LCD corrections downloaded from {lcd_model}")
    return path


def resolve_model_ids(model_name, hf_user):
    """Resolve HF model IDs for a given model name."""
    reg = MODELS[model_name]
    fp16_id = reg["fp16"]
    int3_id = f"{hf_user}/{reg['int3_suffix']}"
    lcd_id  = f"{hf_user}/{reg['lcd_suffix']}"
    return fp16_id, int3_id, lcd_id


def _save_condition(out_dir, cond_name, model_id, languages, ppl, mmlu):
    """Save a single condition's results to JSON."""
    data = {
        "model": model_id,
        "languages": languages,
        "perplexity": ppl,
        "global_mmlu": mmlu,
    }
    path = out_dir / f"eval_{cond_name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_name, fp16_id, int3_id, lcd_id,
                   conditions, languages, device,
                   num_mmlu_samples, num_ppl_samples, ppl_max_seq_len,
                   output_dir):
    """Run perplexity + GlobalMMLU for all requested conditions. Returns results dict."""
    results = {"model": model_name, "conditions": {}}
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    sep = "=" * 64

    # ── FP16 ──────────────────────────────────────────────────────────────
    if "fp16" in conditions:
        print(f"\n{sep}")
        print(f"  [{model_name}] FP16 evaluation")
        print(f"  Model: {fp16_id}")
        print(sep)
        model, tokenizer = load_model(fp16_id, device)

        # Perplexity
        print(f"\n  ── Perplexity (mC4) ──")
        ppl = evaluate_perplexity(model, tokenizer, languages, device,
                                  num_ppl_samples, ppl_max_seq_len)

        # GlobalMMLU
        print(f"\n  ── GlobalMMLU ──")
        mmlu = evaluate_global_mmlu(model, tokenizer, languages, device,
                                    num_mmlu_samples)

        _save_condition(out, "fp16", fp16_id, languages, ppl, mmlu)
        results["conditions"]["fp16"] = {
            "model_id": fp16_id, "perplexity": ppl, "global_mmlu": mmlu,
        }
        del model, tokenizer
        cleanup_gpu()

    # ── INT3 + LCD (share same base model) ────────────────────────────────
    need_int3 = "int3" in conditions
    need_lcd  = "lcd"  in conditions

    if need_int3 or need_lcd:
        print(f"\n{sep}")
        print(f"  [{model_name}] Loading INT3 model")
        print(f"  Model: {int3_id}")
        print(sep)
        model, tokenizer = load_model(int3_id, device)

        # ── INT3 eval ─────────────────────────────────────────────────────
        if need_int3:
            print(f"\n{sep}")
            print(f"  [{model_name}] INT3 evaluation")
            print(sep)

            print(f"\n  ── Perplexity (mC4) ──")
            ppl = evaluate_perplexity(model, tokenizer, languages, device,
                                      num_ppl_samples, ppl_max_seq_len)

            print(f"\n  ── GlobalMMLU ──")
            mmlu = evaluate_global_mmlu(model, tokenizer, languages, device,
                                        num_mmlu_samples)

            _save_condition(out, "int3", int3_id, languages, ppl, mmlu)
            results["conditions"]["int3"] = {
                "model_id": int3_id, "perplexity": ppl, "global_mmlu": mmlu,
            }

        # ── LCD eval (attach corrections to same INT3 model) ─────────────
        if need_lcd:
            print(f"\n{sep}")
            print(f"  [{model_name}] LCD evaluation")
            print(f"  Corrections from: {lcd_id}")
            print(sep)

            corrections_path = download_lcd_corrections(lcd_id)
            lcd_state = load_lcd_corrections(model, corrections_path, device)

            print(f"\n  ── Perplexity (mC4) ──")
            ppl = evaluate_perplexity(model, tokenizer, languages, device,
                                      num_ppl_samples, ppl_max_seq_len,
                                      lcd_state=lcd_state)

            print(f"\n  ── GlobalMMLU ──")
            mmlu = evaluate_global_mmlu(model, tokenizer, languages, device,
                                        num_mmlu_samples, lcd_state=lcd_state)

            _save_condition(out, "lcd", lcd_id, languages, ppl, mmlu)
            results["conditions"]["lcd"] = {
                "model_id": lcd_id, "perplexity": ppl, "global_mmlu": mmlu,
            }

            # Clean up LCD corrections, hooks, and all references
            corrections, hooks, lang_idx = lcd_state
            for h in hooks:
                h.remove()
            del lcd_state, corrections, hooks, lang_idx

        del model, tokenizer
        cleanup_gpu()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ─────────────────────────────────────────────────────────────────────────────
def print_tables(all_results, languages):
    sep = "=" * 100

    for model_result in all_results:
        model_name = model_result["model"]
        conds = model_result["conditions"]

        # ── Perplexity table ──────────────────────────────────────────────
        has_ppl = any(conds.get(c, {}).get("perplexity") for c in ALL_CONDITIONS)
        if has_ppl:
            print(f"\n{sep}")
            print(f"  Perplexity (mC4) — {model_name}")
            print(sep)

            header = f"  {'Condition':<10}" + "".join(f"{l[:6]:>9}" for l in languages) + f"{'avg':>9}"
            print(header)
            print("  " + "─" * (len(header) - 2))

            for cond in ALL_CONDITIONS:
                if cond not in conds:
                    continue
                ppl_data = conds[cond].get("perplexity", {})
                if not ppl_data:
                    continue
                row = f"  {cond.upper():<10}"
                ppls = []
                for lang in languages:
                    p = ppl_data.get(lang)
                    if p is not None:
                        ppls.append(p)
                        row += f"{p:>9.2f}"
                    else:
                        row += f"{'n/a':>9}"
                avg = sum(ppls) / len(ppls) if ppls else 0
                row += f"{avg:>9.2f}"
                print(row)

        # ── Perplexity recovery table ─────────────────────────────────────
        fp16_ppl = conds.get("fp16", {}).get("perplexity", {})
        int3_ppl = conds.get("int3", {}).get("perplexity", {})
        lcd_ppl  = conds.get("lcd",  {}).get("perplexity", {})

        if fp16_ppl and int3_ppl and lcd_ppl:
            non_en = [l for l in languages if l != "english"]
            print(f"\n{sep}")
            print(f"  Perplexity Recovery (%) — {model_name}")
            print(sep)
            header = f"  " + "".join(f"{l[:6]:>9}" for l in non_en) + f"{'avg':>9}"
            print(header)
            print("  " + "─" * (len(header) - 2))

            row = "  "
            recs = []
            for lang in non_en:
                fp = fp16_ppl.get(lang, 0)
                q  = int3_ppl.get(lang, 0)
                c  = lcd_ppl.get(lang, 0)
                gap = q - fp
                if gap > 0.01:
                    rec = (1 - (c - fp) / gap) * 100
                    recs.append(rec)
                    row += f"{rec:>8.1f}%"
                else:
                    row += f"{'n/a':>9}"
            avg_rec = sum(recs) / len(recs) if recs else 0
            row += f"{avg_rec:>8.1f}%"
            print(row)

        # ── GlobalMMLU accuracy table (macro ± std) ──────────────────────
        print(f"\n{sep}")
        print(f"  GlobalMMLU Accuracy — {model_name}")
        print(f"  (macro-averaged across subjects ± std)")
        print(sep)

        header = (f"  {'Cond':<6}"
                  + "".join(f"{l[:6]:>13}" for l in languages)
                  + f"{'avg':>10}")
        print(header)
        print("  " + "─" * (len(header) - 2))

        cond_accs = {}
        for cond in ALL_CONDITIONS:
            if cond not in conds:
                continue
            mmlu = conds[cond].get("global_mmlu", {})
            row = f"  {cond.upper():<6}"
            macro_accs = []
            for lang in languages:
                ld = mmlu.get(lang, {})
                ma = ld.get("macro_accuracy", ld.get("accuracy", 0))
                ms = ld.get("macro_std", 0)
                macro_accs.append(ma)
                row += f"{ma*100:>7.1f}±{ms*100:4.1f}%"
            avg = np.mean(macro_accs) if macro_accs else 0
            row += f"{avg*100:>9.1f}%"
            cond_accs[cond] = {
                lang: mmlu.get(lang, {}).get("macro_accuracy",
                       mmlu.get(lang, {}).get("accuracy", 0))
                for lang in languages
            }
            print(row)

        # ── MMLU accuracy recovery table ──────────────────────────────────
        if "fp16" in cond_accs and "int3" in cond_accs and "lcd" in cond_accs:
            non_en = [l for l in languages if l != "english"]
            print(f"\n{sep}")
            print(f"  GlobalMMLU Recovery (%) — {model_name}")
            print(sep)
            header = f"  " + "".join(f"{l[:6]:>9}" for l in non_en) + f"{'avg':>9}"
            print(header)
            print("  " + "─" * (len(header) - 2))

            row = "  "
            recs = []
            for lang in non_en:
                fp  = cond_accs["fp16"].get(lang, 0)
                q   = cond_accs["int3"].get(lang, 0)
                lcd = cond_accs["lcd"].get(lang, 0)
                gap = fp - q
                if gap > 0.001:
                    rec = (lcd - q) / gap * 100
                    recs.append(rec)
                    row += f"{rec:>8.1f}%"
                else:
                    row += f"{'n/a':>9}"
            avg_rec = sum(recs) / len(recs) if recs else 0
            row += f"{avg_rec:>8.1f}%"
            print(row)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Convention-based mode
    p.add_argument("--hf-user", default=None,
                   help="HF username — derives INT3/LCD model IDs from naming convention")
    p.add_argument("--models", default=None,
                   help="Comma-separated model names to evaluate "
                        f"(default: all). Available: {', '.join(MODELS.keys())}")

    # Explicit model IDs (override convention for single-model runs)
    p.add_argument("--fp16-model", default=None,
                   help="Explicit HF ID for the FP16 model")
    p.add_argument("--int3-model", default=None,
                   help="Explicit HF ID for the INT3 quantized model")
    p.add_argument("--lcd-model", default=None,
                   help="Explicit HF ID for the LCD bundle (contains corrections .pt)")

    # Eval config
    p.add_argument("--conditions", default="fp16,int3,lcd",
                   help="Comma-separated conditions to evaluate (default: fp16,int3,lcd)")
    p.add_argument("--languages", default=",".join(EVAL_LANGUAGES),
                   help="Comma-separated languages")
    p.add_argument("--num-mmlu-samples", default=None, type=int,
                   help="Max samples per language for GlobalMMLU (default: None = full dataset)")
    p.add_argument("--num-ppl-samples", default=256, type=int,
                   help="Samples per language for perplexity measurement (default: 256)")
    p.add_argument("--ppl-max-seq-len", default=512, type=int,
                   help="Max sequence length for perplexity evaluation (default: 512)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="lcd_results",
                   help="Root output directory (default: lcd_results)")

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    conditions = [c.strip() for c in args.conditions.split(",")]
    languages  = [l.strip() for l in args.languages.split(",")]

    # ── Determine what to run ─────────────────────────────────────────────
    runs = []  # list of (model_name, fp16_id, int3_id, lcd_id)

    if args.fp16_model or args.int3_model or args.lcd_model:
        # Explicit single-model mode
        name = "custom"
        if args.fp16_model:
            for mname, mreg in MODELS.items():
                if mreg["fp16"] == args.fp16_model:
                    name = mname
                    break
        runs.append((
            name,
            args.fp16_model or "",
            args.int3_model or "",
            args.lcd_model or "",
        ))
    elif args.hf_user:
        # Convention-based mode
        model_names = list(MODELS.keys())
        if args.models:
            model_names = [m.strip() for m in args.models.split(",")]
        for mname in model_names:
            if mname not in MODELS:
                print(f"  Unknown model: {mname}. Available: {list(MODELS.keys())}")
                sys.exit(1)
            fp16_id, int3_id, lcd_id = resolve_model_ids(mname, args.hf_user)
            runs.append((mname, fp16_id, int3_id, lcd_id))
    else:
        print("  ERROR: Provide --hf-user for convention-based IDs, or explicit model IDs.")
        print("  Examples:")
        print("    python run_evals.py --hf-user your-username")
        print("    python run_evals.py --fp16-model Qwen/Qwen2.5-3B "
              "--int3-model user/qwen2.5-3b-int3-gptq "
              "--lcd-model user/qwen2.5-3b-int3-gptq-lcd")
        sys.exit(1)

    # ── Print plan ────────────────────────────────────────────────────────
    mmlu_desc = f"{args.num_mmlu_samples}" if args.num_mmlu_samples else "full dataset"
    print(f"\n{'='*64}")
    print(f"  LCD Evaluation Runner")
    print(f"  Conditions  : {', '.join(conditions)}")
    print(f"  Languages   : {', '.join(languages)}")
    print(f"  GlobalMMLU  : {mmlu_desc}")
    print(f"  Perplexity  : {args.num_ppl_samples} samples/lang, "
          f"max_len={args.ppl_max_seq_len}")
    print(f"  Device      : {args.device}")
    print(f"  Output      : {args.output_dir}/")
    for mname, fp16_id, int3_id, lcd_id in runs:
        print(f"\n  Model: {mname}")
        if "fp16" in conditions:
            print(f"    FP16 : {fp16_id}")
        if "int3" in conditions:
            print(f"    INT3 : {int3_id}")
        if "lcd" in conditions:
            print(f"    LCD  : {lcd_id}")
    print(f"{'='*64}\n")

    # Validate: don't try conditions with empty model IDs
    for mname, fp16_id, int3_id, lcd_id in runs:
        if "fp16" in conditions and not fp16_id:
            print(f"  ERROR: FP16 model ID required for {mname}")
            sys.exit(1)
        if "int3" in conditions and not int3_id:
            print(f"  ERROR: INT3 model ID required for {mname}")
            sys.exit(1)
        if "lcd" in conditions and not lcd_id:
            print(f"  ERROR: LCD model ID required for {mname}")
            sys.exit(1)

    # ── Run evaluations ───────────────────────────────────────────────────
    all_results = []
    for mname, fp16_id, int3_id, lcd_id in runs:
        result = evaluate_model(
            mname, fp16_id, int3_id, lcd_id,
            conditions, languages, args.device,
            args.num_mmlu_samples, args.num_ppl_samples,
            args.ppl_max_seq_len, args.output_dir,
        )
        all_results.append(result)

        # Free GPU between models
        cleanup_gpu()

    # ── Comparison tables ─────────────────────────────────────────────────
    print_tables(all_results, languages)

    # ── Save combined results ─────────────────────────────────────────────
    combined_path = Path(args.output_dir) / "eval_combined.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"  Combined results → {combined_path}")

    elapsed = time.time() - t0
    print(f"  Total wall time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)\n")


if __name__ == "__main__":
    main()
