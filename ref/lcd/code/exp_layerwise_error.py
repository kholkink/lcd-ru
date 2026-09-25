#!/usr/bin/env python3
"""
Layer-wise Quantization Error Analysis
=======================================
For each linear layer in the model, measures how much INT3 GPTQ changes the
layer's output activations relative to FP16, per language.

Produces:
  - lcd_results/<model>/layerwise_error.json   (raw per-layer per-language ratios)
  - lcd_results/<model>/layerwise_heatmap.png  (layer × language heat map)

Setup: FP16 model on cuda:0, quantized model on cuda:1. Same token IDs → both
models → compare layer outputs.

Usage:
    python exp_layerwise_error.py --model Qwen/Qwen2.5-3B
    python exp_layerwise_error.py --model meta-llama/Llama-3.2-3B
    python exp_layerwise_error.py --model Qwen/Qwen2.5-3B --n-samples 64
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import LANGUAGE_SOURCES, IGNORE, load_texts


# ─────────────────────────────────────────────────────────────────────────────
# Load .env file if present (for HF_TOKEN and other secrets)
# ─────────────────────────────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no external dependency.
    Reads KEY=VALUE lines, ignores comments and blank lines."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Known HuggingFace repos for INT3-quantized base models
# Auto-downloaded when --quant-dir is "auto" and no local copy exists.
# ─────────────────────────────────────────────────────────────────────────────
QUANT_REPOS = {
    "Qwen/Qwen2.5-3B":       "nrmlthms/qwen2.5-3b-int3-gptq",
    "meta-llama/Llama-3.2-3B": "nrmlthms/llama3.2-3b-int3-gptq",
}


def resolve_quant_dir(model: str, quant_dir_arg: str) -> str:
    """Resolve --quant-dir to a local path, downloading from HF if needed."""
    slug = model.replace("/", "_")
    local_path = f"lcd_results/{slug}/quantized"

    # Explicit user override
    if quant_dir_arg != "auto":
        return quant_dir_arg

    # Local copy exists → use it
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        print(f"  [quant] Using local: {local_path}")
        return local_path

    # Otherwise, download from known HF repo
    if model not in QUANT_REPOS:
        raise ValueError(
            f"No local quantized model at '{local_path}' and no known HF repo "
            f"for '{model}'. Either quantize first (python pipeline.py --model {model}) "
            f"or pass --quant-dir <hf-repo-or-path>."
        )

    repo_id = QUANT_REPOS[model]
    print(f"  [quant] Local path '{local_path}' not found.")
    print(f"  [quant] Downloading INT3 model from HF: {repo_id} ...")
    os.makedirs(local_path, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),  # falls back to cached login if None
    )
    print(f"  [quant] Downloaded → {local_path}")
    return local_path


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification
# ─────────────────────────────────────────────────────────────────────────────

ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KEYS  = ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")

def layer_type(name: str) -> str:
    if any(k in name for k in ATTN_KEYS):
        return "attn"
    if any(k in name for k in MLP_KEYS):
        return "mlp"
    return "other"

def layer_index(name: str) -> int:
    """Parse the transformer block index from a layer name."""
    m = re.search(r"layers?[\.\[](\d+)", name)
    return int(m.group(1)) if m else -1


# ─────────────────────────────────────────────────────────────────────────────
# Hook management
# ─────────────────────────────────────────────────────────────────────────────

class ActivationCapture:
    """Registers output hooks on all linear layers and stores the last forward pass."""

    def __init__(self, model: nn.Module):
        self.outputs: Dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if any(ign in name for ign in IGNORE):
                continue
            self._hooks.append(
                mod.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(mod, inp, out):
            self.outputs[name] = out.detach().float()
        return hook

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Error metric  ← YOUR CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_layer_error(y_fp16: torch.Tensor, y_quant: torch.Tensor) -> float:
    """
    Compute a scalar error between the FP16 and INT3 outputs of one linear layer.

    Both tensors have shape (seq_len, out_features) — we've already removed the
    batch dimension by the time this is called.

    Your task: implement the error metric that best answers the question
    "how much does INT3 quantization distort this layer's output for this input?"

    Two strong options to choose between:

    Option A — Relative Frobenius norm (measures magnitude of distortion):
        error = ||y_fp16 - y_quant||_F / (||y_fp16||_F + 1e-8)
    This says: "what fraction of the original signal is distorted?"
    Best for showing that some layers are catastrophically wrong.

    Option B — Mean cosine distance (measures directional change):
        sims = F.cosine_similarity(y_fp16, y_quant, dim=-1)   # (seq_len,)
        error = (1 - sims).mean().item()
    This says: "how much does the direction of each token's representation change?"
    Best for showing that quantization rotates representations away from FP16.
    Scale-invariant — not affected by layers with large absolute activations.

    Both are valid research choices. Option A will likely show larger absolute
    differences in early/embedding-adjacent layers. Option B will show more
    uniform-looking distortion. For the paper's disconnect diagnosis,
    Option A is probably more interpretable.

    Implementation: relative Frobenius norm (Option A).
    """
    diff = y_fp16 - y_quant
    return (diff.norm() / (y_fp16.norm() + 1e-8)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    fp16_model, quant_model, tokenizer,
    languages: List[str],
    n_samples: int,
    max_len: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
        {layer_name: {language: mean_error, ...}, ...}
    """
    cap_fp16  = ActivationCapture(fp16_model)
    cap_quant = ActivationCapture(quant_model)

    dev_fp16  = next(fp16_model.parameters()).device
    dev_quant = next(quant_model.parameters()).device

    # layer_name → language → list of per-sample errors
    errors: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    t_global = time.time()
    lang_bar = tqdm(languages, desc="languages", unit="lang", dynamic_ncols=True)
    for lang in lang_bar:
        lang_bar.set_postfix(current=lang)
        t0 = time.time()
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] [{lang}] loading {n_samples} samples ...")
        texts = load_texts(lang, n_samples, seed=seed)

        sample_bar = tqdm(texts, desc=f"  {lang:<10}", unit="sample",
                          leave=False, dynamic_ncols=True, mininterval=2.0)
        for text in sample_bar:
            tokens = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_len,
            )
            ids_fp16  = tokens.input_ids.to(dev_fp16)
            ids_quant = tokens.input_ids.to(dev_quant)

            cap_fp16.clear()
            cap_quant.clear()

            with torch.no_grad():
                fp16_model(ids_fp16)
                quant_model(ids_quant)

            # Compare outputs for every hooked layer
            for name in cap_fp16.outputs:
                if name not in cap_quant.outputs:
                    continue
                y_fp = cap_fp16.outputs[name].squeeze(0)   # (seq_len, out_f)
                y_q  = cap_quant.outputs[name].to(y_fp.device).squeeze(0)
                err  = compute_layer_error(y_fp, y_q)
                errors[name][lang].append(err)

        # Per-language summary: mean error across all hooked layers and samples
        all_errs = [v for lang_errs in errors.values()
                    for v in lang_errs.get(lang, [])]
        mean_err = float(np.mean(all_errs)) if all_errs else 0.0
        dt = time.time() - t0
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] [{lang}] done in {dt:.1f}s  "
                   f"layers={len(errors)}  mean_err={mean_err:.4f}")

    tqdm.write(f"[{time.strftime('%H:%M:%S')}] total elapsed: "
               f"{(time.time()-t_global)/60:.1f} min")

    cap_fp16.remove()
    cap_quant.remove()

    # Aggregate: mean over samples
    return {
        name: {lang: float(np.mean(vals)) for lang, vals in lang_errs.items()}
        for name, lang_errs in errors.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    results: Dict[str, Dict[str, float]],
    languages: List[str],
    out_path: Path,
    model_name: str,
):
    # Sort layers by (type, block_index, sublayer_name)
    def sort_key(name):
        return (layer_type(name), layer_index(name), name)

    layers_sorted = sorted(results.keys(), key=sort_key)

    # Build matrix: rows = languages, cols = layers
    matrix = np.array([
        [results[l].get(lang, 0.0) for l in layers_sorted]
        for lang in languages
    ])

    fig, ax = plt.subplots(figsize=(max(12, len(layers_sorted) * 0.25), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Relative error (lower = better)")

    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages)
    ax.set_xticks([])
    ax.set_xlabel("Transformer layers (attn → mlp, shallow → deep)")
    ax.set_title(f"Per-layer INT3 quantization error by language — {model_name}")

    # Add vertical separators between attention and MLP blocks
    prev_type = layer_type(layers_sorted[0])
    for i, name in enumerate(layers_sorted):
        t = layer_type(name)
        if t != prev_type:
            ax.axvline(i - 0.5, color="white", linewidth=1.5, alpha=0.6)
        prev_type = t

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default="Qwen/Qwen2.5-3B")
    p.add_argument("--quant-dir",  default="auto",
                   help="Path to saved quantized model ('auto' = lcd_results/<model>/quantized)")
    p.add_argument("--languages",  default="english,arabic,japanese,chinese,french,spanish,korean",
                   help="Comma-separated languages to analyse")
    p.add_argument("--n-samples",  default=32, type=int,
                   help="Samples per language (32 is sufficient; matches ppl eval budget)")
    p.add_argument("--max-len",    default=512, type=int)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--output-dir", default="auto")
    p.add_argument("--fp16-device",  default="cuda:0",
                   help="Device for FP16 model (use same as --quant-device for single-GPU mode)")
    p.add_argument("--quant-device", default="cuda:1",
                   help="Device for quantized model")
    return p.parse_args()


def main():
    args = parse_args()

    languages = [l.strip() for l in args.languages.split(",")]

    slug = args.model.replace("/", "_")
    out_dir   = Path(args.output_dir if args.output_dir != "auto" else f"lcd_results/{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model     : {args.model}")
    quant_dir = resolve_quant_dir(args.model, args.quant_dir)
    print(f"  Quant dir : {quant_dir}")
    print(f"  Languages : {', '.join(languages)}")
    print(f"  Samples   : {args.n_samples} per language\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading FP16 model on {args.fp16_device} ...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.fp16_device, trust_remote_code=True,
    ).eval()

    print(f"  Loading INT3 quantized model on {args.quant_device} ...")
    quant_model = AutoModelForCausalLM.from_pretrained(
        quant_dir, torch_dtype=torch.float16,
        device_map=args.quant_device, trust_remote_code=True,
    ).eval()

    print("\n  Running layer-wise error analysis ...")
    results = analyse(
        fp16_model, quant_model, tokenizer,
        languages=languages,
        n_samples=args.n_samples,
        max_len=args.max_len,
        seed=args.seed,
    )

    # Enrich results with metadata
    enriched = {
        name: {
            "layer_type":  layer_type(name),
            "layer_index": layer_index(name),
            "errors":      lang_errs,
        }
        for name, lang_errs in results.items()
    }

    json_path = out_dir / "layerwise_error.json"
    json_path.write_text(json.dumps(enriched, indent=2))
    print(f"  Results → {json_path}")

    plot_heatmap(results, languages, out_dir / "layerwise_heatmap.png", args.model)

    # Print a quick summary: top-5 most impacted layers per language
    print("\n  Top-5 most impacted layers per language:")
    for lang in languages:
        ranked = sorted(results.items(), key=lambda kv: kv[1].get(lang, 0), reverse=True)[:5]
        top = ", ".join(f"{name.split('.')[-1]}(L{layer_index(name)})={v[lang]:.3f}"
                        for name, v in ranked)
        print(f"    {lang:<12}: {top}")


if __name__ == "__main__":
    main()
