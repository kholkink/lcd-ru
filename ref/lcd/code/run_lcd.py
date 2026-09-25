#!/usr/bin/env python3
"""
LCD Experiment — Full Pipeline
================================
Runs the complete Language-Conditional Dequantization (LCD) experiment for:
  - Qwen/Qwen2.5-3B
  - meta-llama/Llama-3.2-3B

Steps per model:
  1. LCD pipeline  : FP16 ppl → GPTQ INT3 → per-language LoRA corrections
  2. Benchmark eval: GlobalMMLU accuracy for FP16 / INT3 / LCD
  3. HF push       : INT3 + LCD-corrected model to HuggingFace (optional)

Results are saved to lcd_results/<model-name>/

Usage:
    # Full run (pipeline + eval)
    python run_lcd.py

    # With HF push
    python run_lcd.py --hf-user your-username

    # Quick test (fewer samples)
    python run_lcd.py --fast --num-eval-samples 100

    # Skip pipeline, only re-run benchmarks
    python run_lcd.py --skip-pipeline

    # Skip push
    python run_lcd.py --hf-user your-username --skip-push
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "qwen2.5-3b",
        "hf_id": "Qwen/Qwen2.5-3B",
    },
    {
        "name": "llama3.2-3b",
        "hf_id": "meta-llama/Llama-3.2-3B",
    },
]

# All languages for LCD training (perplexity-based)
LCD_LANGUAGES = "english,arabic,japanese,chinese,hindi,russian,french,spanish,korean,code"

# Languages for benchmark evaluation
EVAL_LANGUAGES = "english,arabic,japanese,chinese,hindi,russian,french,spanish,korean"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def run(cmd, desc=""):
    """Run a command, stream output, return exit code."""
    sep = "=" * 64
    print(f"\n{sep}")
    if desc:
        print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{sep}\n", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n  [WARN] command exited with code {rc}")
    return rc


def push_to_hub(local_dir: str, repo_id: str, hf_user: str):
    """Push a saved model directory to HuggingFace Hub."""
    print(f"\n  Pushing → {repo_id} ...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message="LCD INT3 GPTQ quantized model",
        )
        print(f"  Pushed: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  HF push failed: {e}")
        return False


def load_json(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-model pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_model(model, args):
    name = model["name"]
    hf_id = model["hf_id"]
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    quantized_dir = str(out / "quantized")
    corrections_pt = str(out / "lcd_corrections.pt")
    pipeline_json  = str(out / "lcd_results.json")
    eval_fp16_json = str(out / "eval_fp16.json")
    eval_int3_json = str(out / "eval_int3.json")
    eval_lcd_json  = str(out / "eval_lcd.json")

    print(f"\n{'#'*64}")
    print(f"  MODEL : {name}")
    print(f"  HF ID : {hf_id}")
    print(f"  OutDir: {out}")
    print(f"{'#'*64}")

    # ── 1. LCD pipeline ───────────────────────────────────────────────────
    if not args.skip_pipeline:
        pipeline_cmd = [
            sys.executable, "pipeline.py",
            "--model", hf_id,
            "--device", args.device,
            "--languages", LCD_LANGUAGES,
            "--save-quantized-dir", quantized_dir,
            "--output", pipeline_json,
            "--save-corrections", corrections_pt,
            "--rank", "2",
            "--steps", "500",
            "--lr", "5e-4",
            "--num-calibration-samples", "128",
            "--num-train-samples", "256",
            "--num-fp16-samples", "32",
            "--num-quant-samples", "32",
            "--num-eval-samples", "32",
            "--grid-search",
        ]
        if args.fast:
            pipeline_cmd.append("--fast")

        rc = run(pipeline_cmd, desc=f"LCD pipeline — {name}")
        if rc != 0:
            print(f"\n  PIPELINE FAILED for {name}. Skipping evaluation.")
            return {"name": name, "status": "pipeline_failed"}

    # ── 2. Benchmark evaluation ───────────────────────────────────────────
    if not args.skip_eval:
        base_eval = [
            sys.executable, "evaluate.py",
            "--device", args.device,
            "--benchmarks", "global_mmlu",
            "--languages", EVAL_LANGUAGES,
        ]
        if args.num_eval_samples:
            base_eval += ["--num-samples", str(args.num_eval_samples)]

        # FP16
        run(base_eval + ["--model", hf_id, "--output", eval_fp16_json],
            desc=f"GlobalMMLU eval — {name} FP16")

        # INT3
        if Path(quantized_dir).exists():
            run(base_eval + ["--model", quantized_dir, "--output", eval_int3_json],
                desc=f"GlobalMMLU eval — {name} INT3")
        else:
            print(f"  [SKIP] INT3 eval — quantized model not found at {quantized_dir}")

        # LCD (INT3 + corrections)
        if Path(quantized_dir).exists() and Path(corrections_pt).exists():
            run(base_eval + [
                    "--model", quantized_dir,
                    "--lcd-corrections", corrections_pt,
                    "--output", eval_lcd_json,
                ],
                desc=f"GlobalMMLU eval — {name} LCD")
        else:
            print(f"  [SKIP] LCD eval — missing quantized model or corrections")

    # ── 3. HuggingFace push ───────────────────────────────────────────────
    if not args.skip_push and args.hf_user and Path(quantized_dir).exists():
        # Push raw INT3 model
        push_to_hub(quantized_dir, f"{args.hf_user}/{name}-int3-gptq", args.hf_user)

        # Push LCD-corrected bundle (INT3 weights + corrections file)
        if Path(corrections_pt).exists():
            lcd_bundle = str(out / "lcd_corrected")
            lcd_bundle_p = Path(lcd_bundle)
            lcd_bundle_p.mkdir(parents=True, exist_ok=True)
            for f in Path(quantized_dir).iterdir():
                shutil.copy2(f, lcd_bundle_p / f.name)
            shutil.copy2(corrections_pt, lcd_bundle_p / "lcd_corrections.pt")
            push_to_hub(lcd_bundle, f"{args.hf_user}/{name}-int3-gptq-lcd", args.hf_user)

    return {"name": name, "status": "completed"}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ─────────────────────────────────────────────────────────────────────────────
def print_tables(output_dir: str):
    out = Path(output_dir)
    langs = EVAL_LANGUAGES.split(",")
    models = [m["name"] for m in MODELS]
    conditions = [("FP16", "eval_fp16.json"),
                  ("INT3", "eval_int3.json"),
                  ("LCD",  "eval_lcd.json")]

    # ── Perplexity recovery table ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  LCD Recovery — Perplexity Gap Closed (%)")
    print(f"{'='*90}")
    ppl_langs = [l for l in LCD_LANGUAGES.split(",") if l != "english"]
    header = f"  {'Model':<16}" + "".join(f"{l[:7]:>10}" for l in ppl_langs) + f"{'avg':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for mname in models:
        res_path = out / mname / "lcd_results.json"
        if not res_path.exists():
            continue
        data = load_json(res_path).get("results", {})
        row = f"  {mname:<16}"
        recs = []
        for lang in ppl_langs:
            rec = data.get(lang, {}).get("recovery_pct", None)
            if rec is not None:
                row += f"{rec:>9.1f}%"
                recs.append(rec)
            else:
                row += f"{'n/a':>10}"
        avg = sum(recs) / len(recs) if recs else 0
        row += f"{avg:>7.1f}%"
        print(row)

    # ── GlobalMMLU accuracy table ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GlobalMMLU Accuracy — FP16 / INT3 / LCD")
    print(f"{'='*90}")
    header = f"  {'Model':<16} {'Cond':<6}" + "".join(f"{l[:6]:>8}" for l in langs)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for mname in models:
        for cond, fname in conditions:
            fpath = out / mname / fname
            if not fpath.exists():
                continue
            mmlu = load_json(fpath).get("global_mmlu", {})
            row = f"  {mname:<16} {cond:<6}"
            for lang in langs:
                acc = mmlu.get(lang, {}).get("accuracy", 0)
                row += f"{acc*100:>7.1f}%"
            print(row)
        print()

    # ── LCD accuracy recovery table ───────────────────────────────────────
    print(f"{'='*90}")
    print(f"  GlobalMMLU LCD — Accuracy Gap Recovered (%)")
    print(f"{'='*90}")
    non_en = [l for l in langs if l != "english"]
    header = f"  {'Model':<16}" + "".join(f"{l[:6]:>8}" for l in non_en) + f"{'avg':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for mname in models:
        fp16_d = load_json(out / mname / "eval_fp16.json").get("global_mmlu", {})
        int3_d = load_json(out / mname / "eval_int3.json").get("global_mmlu", {})
        lcd_d  = load_json(out / mname / "eval_lcd.json").get("global_mmlu",  {})
        if not fp16_d or not int3_d or not lcd_d:
            continue
        row = f"  {mname:<16}"
        recs = []
        for lang in non_en:
            fp  = fp16_d.get(lang, {}).get("accuracy", 0)
            q   = int3_d.get(lang, {}).get("accuracy", 0)
            c   = lcd_d.get(lang,  {}).get("accuracy", 0)
            gap = fp - q
            if gap > 0.001:
                rec = (c - q) / gap * 100
                recs.append(rec)
                row += f"{rec:>7.1f}%"
            else:
                row += f"{'n/a':>8}"
        avg = sum(recs) / len(recs) if recs else 0
        row += f"{avg:>7.1f}%"
        print(row)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device",  default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--hf-user", default=None,
                   help="HuggingFace username — enables model push after pipeline")
    p.add_argument("--fast",    action="store_true",
                   help="Use fewer samples for quick testing")
    p.add_argument("--num-eval-samples", default=200, type=int,
                   help="Samples per language for GlobalMMLU eval (default: 200)")
    p.add_argument("--output-dir", default="lcd_results",
                   help="Root output directory (default: lcd_results)")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip LCD pipeline (use existing quantized models + corrections)")
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip benchmark evaluation")
    p.add_argument("--skip-push",     action="store_true",
                   help="Skip HuggingFace push")
    p.add_argument("--models", default=None,
                   help="Comma-separated model names to run, e.g. qwen2.5-3b,llama3.2-3b")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Filter models if requested
    models = MODELS
    if args.models:
        names = {n.strip() for n in args.models.split(",")}
        models = [m for m in MODELS if m["name"] in names]
        if not models:
            print(f"Unknown model(s). Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  LCD Experiment")
    print(f"  Models  : {', '.join(m['name'] for m in models)}")
    print(f"  Device  : {args.device}")
    print(f"  OutDir  : {args.output_dir}")
    print(f"  HF User : {args.hf_user or '(none — push disabled)'}")
    print(f"{'='*64}\n")

    for model in models:
        run_model(model, args)
        # Free GPU memory between models
        subprocess.run([sys.executable, "-c",
                        "import torch; torch.cuda.empty_cache()"],
                       capture_output=True)

    print_tables(args.output_dir)

    elapsed = time.time() - t0
    print(f"\n  Results → {args.output_dir}/")
    print(f"  Total wall time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)\n")


if __name__ == "__main__":
    main()
