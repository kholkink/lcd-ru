#!/usr/bin/env python3
"""
LCD Full Pipeline — Diagnose + Train + Evaluate in one run
==========================================================
Steps (shared quantization, no redundant GPTQ):

  1. Load model in fp16
  2. Measure fp16 perplexity per language         [DIAGNOSTIC phase 1]
  3. GPTQ W3A16 on English C4 (once)
  4. Measure INT3 perplexity per language          [DIAGNOSTIC phase 2]
  5. Print diagnostic table + hypothesis verdict
  6. Attach per-language LoRA corrections (init = 0)
  7. For each language: train correction on mC4    [TRAINING]
  8. Evaluate corrected perplexity per language    [EVALUATION]
  9. Print final recovery table
  10. Save corrections + results

Usage:
    python pipeline.py
    python pipeline.py --model Qwen/Qwen2.5-3B --device cuda
    python pipeline.py --model meta-llama/Llama-3.2-3B --device cuda:1
    python pipeline.py --rank 4 --steps 500 --fast   # --fast uses fewer samples
"""

import argparse
import itertools
import json
import math
import random
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Shim: torch/torchvision version mismatch breaks transformers' lazy import of
# PreTrainedModel (via image_transforms → torchvision).  Fix: uninstall the
# broken torchvision so transformers skips the vision path, or patch around it.
import sys, importlib
# Block the broken torchvision import so transformers doesn't choke on it
if "torchvision" not in sys.modules:
    try:
        import torchvision  # noqa: F401
    except (RuntimeError, ImportError, Exception):
        # Full recursive stub so any torchvision.* import resolves silently
        import types as _types
        class _StubModule(_types.ModuleType):
            def __getattr__(self, name):
                fullname = f"{self.__name__}.{name}"
                if fullname in sys.modules:
                    return sys.modules[fullname]
                sub = _StubModule(fullname)
                sys.modules[fullname] = sub
                super().__setattr__(name, sub)
                return sub
            def __call__(self, *a, **kw):
                return self
        _tv = _StubModule("torchvision")
        sys.modules["torchvision"] = _tv

import transformers  # noqa: E402
# Now force PreTrainedModel into the transformers namespace
try:
    transformers.PreTrainedModel
except Exception:
    from transformers.modeling_utils import PreTrainedModel
    transformers.PreTrainedModel = PreTrainedModel

try:
    from llmcompressor.modifiers.quantization import GPTQModifier
    from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES
    from compressed_tensors.quantization import QuantizationArgs
except ImportError as e:
    print(f"ERROR: llm-compressor import failed: {e}")
    sys.exit(1)

try:
    from llmcompressor.transformers import oneshot
except ImportError:
    from llmcompressor import oneshot

if "W3A16" not in PRESET_SCHEMES:
    PRESET_SCHEMES["W3A16"] = {
        "weights": QuantizationArgs(
            num_bits=3, type="int", symmetric=True,
            group_size=128, strategy="group",
        )
    }

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALL_LANGUAGES = [
    "english", "arabic", "japanese", "chinese", "hindi",
    "russian", "french", "spanish", "korean", "code",
]
LANGUAGES = ALL_LANGUAGES          # overridden by --languages CLI
BASELINE  = "english"
IGNORE    = ("lm_head", "embed_tokens")

LANGUAGE_SOURCES = {
    "english":  ("allenai/c4",                   "en",      "text"),
    "arabic":   ("allenai/c4",                   "ar",      "text"),
    "japanese": ("allenai/c4",                   "ja",      "text"),
    "chinese":  ("allenai/c4",                   "zh",      "text"),
    "hindi":    ("allenai/c4",                   "hi",      "text"),
    "russian":  ("allenai/c4",                   "ru",      "text"),
    "french":   ("allenai/c4",                   "fr",      "text"),
    "spanish":  ("allenai/c4",                   "es",      "text"),
    "korean":   ("allenai/c4",                   "ko",      "text"),
    "code":     ("codeparrot/github-code-clean", "default", "code"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_texts(
    lang: str,
    n: int,
    seed: int = 42,
    min_words: int = 30,
    max_chars: int = 2000,
    cache_dir: Optional[str] = None,
) -> List[str]:
    dataset_name, config, text_field = LANGUAGE_SOURCES[lang]
    print(f"    [{lang:<9}] {dataset_name}/{config}  n={n} ...", end=" ", flush=True)
    try:
        ds = load_dataset(dataset_name, config, split="train",
                          streaming=True,
                          **({"cache_dir": cache_dir} if cache_dir else {}))
        ds = ds.shuffle(seed=seed, buffer_size=50_000)
    except RuntimeError as e:
        if "scripts are no longer supported" in str(e):
            ds = load_dataset(dataset_name, config, split="train",
                              streaming=True, revision="refs/convert/parquet",
                              **({"cache_dir": cache_dir} if cache_dir else {}))
            ds = ds.shuffle(seed=seed, buffer_size=50_000)
        else:
            raise

    texts = []
    with tqdm(total=n, desc=f"  {lang:<9}", unit="doc", leave=False) as pbar:
        for sample in ds:
            text = sample[text_field].strip()
            if len(text.split()) < min_words:
                continue
            texts.append(" ".join(text.split()[:max_chars]))
            pbar.update(1)
            if len(texts) >= n:
                break

    if len(texts) < n:
        print(f"  Warning: only got {len(texts)}/{n} samples for {lang}")
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────
def perplexity(model, tokenizer, texts: List[str], device: str, max_len: int,
               desc: str = "ppl") -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for text in tqdm(texts, desc=f"  {desc}", unit="doc", leave=False):
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(ids, labels=ids)
            n   = ids.shape[1] - 1
            if n > 0:
                total_nll += out.loss.item() * n
                total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


# ─────────────────────────────────────────────────────────────────────────────
# GPTQ
# ─────────────────────────────────────────────────────────────────────────────
def detect_decoder_layer(model) -> Optional[List[str]]:
    for m in model.modules():
        if "DecoderLayer" in type(m).__name__:
            return [type(m).__name__]
    return None


def run_gptq(model, tokenizer, cal_texts: List[str], args) -> None:
    cal_ds      = Dataset.from_dict({"text": cal_texts})
    seq_targets = detect_decoder_layer(model)
    recipe_kwargs = dict(
        targets="Linear",
        scheme="W3A16",
        ignore=["lm_head"],
    )
    if seq_targets:
        recipe_kwargs["sequential_targets"] = seq_targets
    recipe = GPTQModifier(**recipe_kwargs)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=cal_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calibration_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LCD correction module
# ─────────────────────────────────────────────────────────────────────────────
class LangCorrection(nn.Module):
    """Rank-r additive correction for one linear layer, one language slot each."""
    def __init__(self, in_f: int, out_f: int, n_langs: int, rank: int):
        super().__init__()
        self.scale = 1.0 / rank
        # A=0, B~N(0,0.01) → correction starts at exactly zero
        self.A = nn.ParameterList([nn.Parameter(torch.zeros(out_f, rank))
                                   for _ in range(n_langs)])
        self.B = nn.ParameterList([nn.Parameter(torch.empty(rank, in_f).normal_(0, 0.01))
                                   for _ in range(n_langs)])

    def forward(self, x: torch.Tensor, lang_idx: int) -> torch.Tensor:
        x = x.to(self.B[lang_idx].dtype)
        return (x @ self.B[lang_idx].T) @ self.A[lang_idx].T * self.scale


# Layer-selection strategies
LAYER_STRATEGIES = {
    "all":         lambda name: True,
    "attn_only":   lambda name: any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")),
    "mlp_only":    lambda name: any(k in name for k in ("gate_proj", "up_proj", "down_proj")),
    "top_half":    None,  # needs n_layers, set dynamically
    "bottom_half": None,
}


class LCD:
    """Hooks per-language LoRA corrections onto a frozen quantized model."""

    def __init__(self, model: nn.Module, rank: int, layer_strategy: str = "all",
                 target_layer_indices: Optional[List[int]] = None):
        self.model    = model
        self.rank     = rank
        self.layer_strategy = layer_strategy
        self.lang_idx = 0
        self.corrections: nn.ModuleList = nn.ModuleList()
        self.layer_names: List[str]     = []
        self._hooks: list               = []

        for p in model.parameters():
            p.requires_grad_(False)

        # Collect all candidate linear layers
        candidates = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if any(ign in name for ign in IGNORE):
                continue
            candidates.append((name, mod))

        # Apply layer-selection strategy
        import re as _re
        if target_layer_indices is not None:
            # Exact depth-band targeting: only correct the specified layer indices
            _target_set = set(target_layer_indices)
            keep = lambda name: any(
                bool(_re.search(rf"layers\.{i}\.", name)) for i in _target_set
            )
        elif layer_strategy in ("top_half", "bottom_half"):
            layer_indices = set()
            for name, _ in candidates:
                m = _re.search(r"layers\.(\d+)\.", name)
                if m:
                    layer_indices.add(int(m.group(1)))
            if layer_indices:
                mid = max(layer_indices) // 2
                if layer_strategy == "bottom_half":
                    keep = lambda name: any(f"layers.{i}." in name for i in layer_indices if i <= mid)
                else:
                    keep = lambda name: any(f"layers.{i}." in name for i in layer_indices if i > mid)
            else:
                keep = lambda name: True
        else:
            keep = LAYER_STRATEGIES[layer_strategy]

        dev = next(model.parameters()).device
        for name, mod in candidates:
            if not keep(name):
                continue
            corr = LangCorrection(mod.in_features, mod.out_features,
                                  n_langs=len(LANGUAGES), rank=rank).to(dev)
            self.corrections.append(corr)
            self.layer_names.append(name)
            self._hooks.append(mod.register_forward_hook(self._hook(corr)))

        n_per_lang = sum(
            sum(p.numel() for p in c.A) // len(LANGUAGES) +
            sum(p.numel() for p in c.B) // len(LANGUAGES)
            for c in self.corrections
        )
        base = sum(p.numel() for p in model.parameters())
        print(f"  LCD: {len(self.corrections)} layers ({layer_strategy})  "
              f"{n_per_lang:,} params/language  "
              f"({n_per_lang/base*100:.3f}% of base)")

    def _hook(self, corr: LangCorrection):
        lcd = self
        def h(mod, inp, out):
            return out + corr(inp[0], lcd.lang_idx).to(out.dtype)
        return h

    def set_lang(self, lang: str | int):
        self.lang_idx = lang if isinstance(lang, int) else LANGUAGES.index(lang)

    def enable_lang(self, lang_idx: int) -> List[nn.Parameter]:
        for corr in self.corrections:
            for p in corr.parameters():
                p.requires_grad_(False)
        params = []
        for corr in self.corrections:
            for p in [corr.A[lang_idx], corr.B[lang_idx]]:
                p.requires_grad_(True)
                params.append(p)
        return params

    def freeze_all(self):
        for corr in self.corrections:
            for p in corr.parameters():
                p.requires_grad_(False)

    def ppl(self, lang: str, texts: List[str], tokenizer, device: str, max_len: int) -> float:
        self.set_lang(lang)
        self.freeze_all()
        return perplexity(self.model, tokenizer, texts, device, max_len,
                          desc=f"eval {lang}")

    def train_lang(self, lang: str, texts: List[str], tokenizer,
                   device: str, steps: int, lr: float,
                   max_len: int, log_every: int = 50,
                   weight_decay: float = 0.01,
                   warmup_ratio: float = 0.0,
                   grad_accum: int = 1) -> None:
        lang_idx = LANGUAGES.index(lang)
        params   = self.enable_lang(lang_idx)
        self.set_lang(lang_idx)

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        warmup_steps = int(steps * warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        self.model.train()
        losses: List[float] = []

        pbar = tqdm(
            itertools.islice(itertools.cycle(texts), steps * grad_accum),
            total=steps * grad_accum,
            desc=f"  train {lang}",
            unit="sample",
            dynamic_ncols=True,
        )
        opt.zero_grad()
        for sample_i, text in enumerate(pbar, 1):
            ids  = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=max_len).input_ids.to(device)
            loss = self.model(ids, labels=ids).loss / grad_accum
            loss.backward()
            losses.append(loss.item() * grad_accum)

            if sample_i % grad_accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step = sample_i // grad_accum
                if step % log_every == 0 or step == steps:
                    avg = np.mean(losses[-log_every * grad_accum:])
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        pbar.close()
        self.model.eval()
        del opt, sched
        torch.cuda.empty_cache()

    def remove(self):
        """Detach all hooks and delete correction modules."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        del self.corrections
        self.corrections = nn.ModuleList()
        self.layer_names.clear()
        torch.cuda.empty_cache()

    def save(self, path: Path):
        torch.save({"rank": self.rank, "layer_names": self.layer_names,
                    "corrections": self.corrections.state_dict()}, path)
        print(f"  Corrections saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 82

def print_diagnostic(fp16: Dict, quant: Dict, args):
    en_deg = quant[BASELINE] / max(fp16[BASELINE], 1e-8)
    print(f"\n{SEP}")
    print("  DIAGNOSTIC — Perplexity Degradation at INT3 GPTQ")
    print(f"  Model: {args.model}  |  Group: {args.group_size}  |  Calib: C4 EN ×{args.num_calibration_samples}")
    print(SEP)
    print(f"\n  {'Language':<12} {'FP16':>8} {'INT3':>8} {'INT3/FP16':>11} {'vs EN degrad':>14}")
    print("  " + "─" * 58)
    for lang in LANGUAGES:
        deg   = quant[lang] / max(fp16[lang], 1e-8)
        ratio = deg / max(en_deg, 1e-8)
        mark  = "  * ← disproportionate" if lang != BASELINE and ratio >= 1.5 else ""
        vs    = "(baseline)" if lang == BASELINE else f"{ratio:.3f}x{mark}"
        print(f"  {lang:<12} {fp16[lang]:>8.2f} {quant[lang]:>8.2f} {deg:>10.3f}x  {vs}")

    hits = [l for l in LANGUAGES if l != BASELINE
            and quant[l]/max(fp16[l],1e-8) / max(en_deg,1e-8) >= 1.5]
    print(f"\n{SEP}")
    if hits:
        print(f"\n  HYPOTHESIS SUPPORTED — disproportionate INT3 harm on: {', '.join(hits)}")
        print(f"  Proceeding to correction training ...\n")
    else:
        print(f"\n  Ratios below 1.5x threshold — effect may be model-dependent.")
        print(f"  Continuing to train corrections anyway ...\n")


def print_recovery(fp16: Dict, quant: Dict, corrected: Dict, args):
    print(f"\n{SEP}")
    print("  RECOVERY TABLE — LCD Correction (rank={}, {} steps/lang)".format(
        args.rank, args.steps))
    print(f"  Model: {args.model}")
    print(SEP)
    print(f"\n  {'Language':<12} {'FP16':>7} {'INT3':>7} {'LCD':>7}  "
          f"{'INT3/FP16':>10} {'LCD/FP16':>10} {'Recovery':>10}")
    print("  " + "─" * 72)

    for lang in LANGUAGES:
        fp = fp16[lang];  q = quant[lang];  c = corrected[lang]
        gap = q - fp
        rec = (1 - (c - fp) / max(gap, 1e-8)) * 100 if gap > 0 else 0.0
        mark = "  ✓" if rec >= 40 else ""
        print(f"  {lang:<12} {fp:>7.2f} {q:>7.2f} {c:>7.2f}  "
              f"{q/max(fp,1e-8):>9.3f}x {c/max(fp,1e-8):>9.3f}x {rec:>8.1f}%{mark}")

    avg_rec = np.mean([
        (1 - (corrected[l] - fp16[l]) / max(quant[l] - fp16[l], 1e-8)) * 100
        for l in LANGUAGES if l != BASELINE and quant[l] > fp16[l]
    ])
    print(f"\n  Average non-English recovery: {avg_rec:.1f}%")
    print(f"\n{SEP}")
    if avg_rec >= 50:
        print("  LCD WORKS — >50% of quantization gap recovered with <0.1% extra params.")
    elif avg_rec >= 25:
        print("  PARTIAL — >25% recovery. Try --rank 4 --steps 500 for more.")
    else:
        print("  LOW RECOVERY — try higher rank or more training steps.")
    print(f"{SEP}\n")


def save_results(fp16, quant, corrected, args, path: Path):
    out = {
        "model": args.model, "rank": args.rank, "steps": args.steps,
        "group_size": args.group_size,
        "results": {
            lang: {
                "fp16_ppl":      fp16[lang],
                "quant_ppl":     quant[lang],
                "corrected_ppl": corrected[lang],
                "recovery_pct":  (
                    (1 - (corrected[lang] - fp16[lang])
                     / max(quant[lang] - fp16[lang], 1e-8)) * 100
                    if quant[lang] > fp16[lang] else 0.0
                ),
            }
            for lang in LANGUAGES
        },
    }
    path.write_text(json.dumps(out, indent=2))
    print(f"  Results → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",                   default="Qwen/Qwen2.5-3B")
    p.add_argument("--device",                  default="cuda")
    p.add_argument("--group-size",              default=128,   type=int)

    # Data sample counts
    p.add_argument("--num-calibration-samples", default=128,   type=int,
                   help="C4 samples for GPTQ Hessian")
    p.add_argument("--num-fp16-samples",        default=32,    type=int,
                   help="mC4 samples for fp16 perplexity measurement")
    p.add_argument("--num-quant-samples",       default=32,    type=int,
                   help="mC4 samples for INT3 perplexity measurement")
    p.add_argument("--num-train-samples",       default=256,   type=int,
                   help="mC4 samples for correction training per language")
    p.add_argument("--num-eval-samples",        default=32,    type=int,
                   help="mC4 samples for corrected perplexity evaluation")

    # Correction training
    p.add_argument("--rank",   default=2,    type=int,   help="LoRA rank")
    p.add_argument("--steps",  default=500,  type=int,   help="Training steps per language")
    p.add_argument("--lr",     default=5e-4, type=float)
    p.add_argument("--max-seq-len", default=512, type=int)
    p.add_argument("--seed",   default=42,   type=int)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--layer-strategy", default="all",
                   choices=["all", "attn_only", "mlp_only", "top_half", "bottom_half"],
                   help="Which linear layers to attach LCD corrections to.")

    # Presets
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke test: fewer samples and steps")
    p.add_argument("--grid-search", action="store_true",
                   help="Grid search over rank/lr/steps/wd (runs diagnostic once)")

    # Languages
    p.add_argument("--languages", default=None,
                   help="Comma-separated languages (default: all). E.g. english,arabic,japanese,code")

    # Output
    p.add_argument("--output",           default="lcd_pipeline_results.json")
    p.add_argument("--save-corrections", default="lcd_corrections.pt")
    p.add_argument("--save-quantized-dir", default="auto",
                   help="Save quantized model dir ('auto' = lcd_results/<model>/quantized, 'none' = skip)")
    p.add_argument("--load-quantized-dir", default=None,
                   help="Skip GPTQ and load a pre-saved quantized model from this directory.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────
def _run_trial(model, tokenizer, train_texts, eval_texts, fp16_ppls, quant_ppls,
               args, *, rank, lr, steps, wd, warmup, accum, layers, non_en):
    """Run one trial, return dict with results."""
    import gc
    lcd = LCD(model, rank=rank, layer_strategy=layers)
    trial_ppls = {}
    for lang in non_en:
        lcd.train_lang(lang, train_texts[lang], tokenizer,
                       args.device, steps, lr, args.max_seq_len,
                       weight_decay=wd, warmup_ratio=warmup, grad_accum=accum)
        trial_ppls[lang] = lcd.ppl(lang, eval_texts[lang], tokenizer,
                                   args.device, args.max_seq_len)
    trial_ppls[BASELINE] = quant_ppls[BASELINE]

    recoveries = {}
    for lang in non_en:
        gap = quant_ppls[lang] - fp16_ppls[lang]
        recoveries[lang] = (
            (1 - (trial_ppls[lang] - fp16_ppls[lang]) / max(gap, 1e-8)) * 100
            if gap > 0 else 0.0
        )
    avg_rec = np.mean(list(recoveries.values()))

    lcd.remove()
    del lcd
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "rank": rank, "lr": lr, "steps": steps, "wd": wd,
        "warmup": warmup, "accum": accum, "layers": layers,
        "avg_recovery": avg_rec, "ppls": dict(trial_ppls),
        "per_lang_recovery": recoveries,
    }


def grid_search(model, tokenizer, fp16_ppls, quant_ppls,
                train_texts, eval_texts, args):
    """Two-phase grid search: coarse sweep → fine-tune around best."""
    non_en = [l for l in LANGUAGES if l != BASELINE]

    # ── PHASE 1: Coarse sweep (short training) across structural dimensions ──
    phase1_grid = [
        # (rank, lr, steps, wd, warmup, accum, layers)
        # Layer strategies (most impactful structural decision)
        (2, 5e-4, 200, 0.01, 0.1, 1, "all"),
        (2, 5e-4, 200, 0.01, 0.1, 1, "attn_only"),
        (2, 5e-4, 200, 0.01, 0.1, 1, "mlp_only"),
        (2, 5e-4, 200, 0.01, 0.1, 1, "top_half"),
        (2, 5e-4, 200, 0.01, 0.1, 1, "bottom_half"),
        # Rank sweep
        (1, 5e-4, 200, 0.01, 0.1, 1, "all"),
        (4, 5e-4, 200, 0.01, 0.1, 1, "all"),
        (8, 5e-4, 200, 0.01, 0.1, 1, "all"),
        # LR sweep
        (2, 1e-5, 200, 0.01, 0.1, 1, "all"),
        (2, 5e-5, 200, 0.01, 0.1, 1, "all"),
        (2, 1e-4, 200, 0.01, 0.1, 1, "all"),
        (2, 1e-3, 200, 0.01, 0.1, 1, "all"),
        # Warmup vs no warmup
        (2, 5e-4, 200, 0.01, 0.0, 1, "all"),
        # Grad accumulation
        (2, 5e-4, 200, 0.01, 0.1, 4, "all"),
        # Weight decay sweep
        (2, 5e-4, 200, 0.0,  0.1, 1, "all"),
        (2, 5e-4, 200, 0.1,  0.1, 1, "all"),
    ]

    print(f"\n{'='*82}")
    print(f"  PHASE 1 — Coarse sweep ({len(phase1_grid)} trials, 200 steps each)")
    print(f"{'='*82}\n")

    phase1_results = []
    phase1_t0 = time.time()
    for i, (rank, lr, steps, wd, warmup, accum, layers) in enumerate(phase1_grid, 1):
        print(f"  ── P1 Trial {i}/{len(phase1_grid)}:  rank={rank}  lr={lr:.0e}  "
              f"steps={steps}  wd={wd}  warmup={warmup}  accum={accum}  layers={layers} ──")
        trial_t0 = time.time()
        r = _run_trial(model, tokenizer, train_texts, eval_texts, fp16_ppls, quant_ppls,
                       args, rank=rank, lr=lr, steps=steps, wd=wd,
                       warmup=warmup, accum=accum, layers=layers, non_en=non_en)
        trial_dt = time.time() - trial_t0
        phase1_results.append(r)
        per = "  ".join(f"{l}={r['per_lang_recovery'][l]:+.1f}%" for l in non_en)
        elapsed = time.time() - phase1_t0
        eta = elapsed / i * (len(phase1_grid) - i)
        print(f"    avg: {r['avg_recovery']:+.1f}%  ({per})  "
              f"[{trial_dt/60:.1f}min, ETA {eta/60:.0f}min]\n")

    phase1_results.sort(key=lambda x: x["avg_recovery"], reverse=True)

    print(f"\n{'='*82}")
    print(f"  PHASE 1 LEADERBOARD")
    print(f"{'='*82}")
    print(f"  {'#':>3} {'rank':>5} {'lr':>9} {'wd':>6} {'warm':>5} {'acc':>4} {'layers':>12} {'avg':>8}  per-lang")
    print(f"  {'─'*82}")
    for i, r in enumerate(phase1_results, 1):
        per = "  ".join(f"{l}={r['per_lang_recovery'][l]:+.1f}%" for l in non_en)
        print(f"  {i:>3} {r['rank']:>5} {r['lr']:>9.0e} {r['wd']:>6} {r['warmup']:>5} "
              f"{r['accum']:>4} {r['layers']:>12} {r['avg_recovery']:>+7.1f}%  {per}")

    # ── PHASE 2: Fine-tune top-3 configs with longer training ────────────────
    top3 = phase1_results[:3]
    phase2_steps = [500, 1000]

    phase2_combos = []
    for base in top3:
        for steps in phase2_steps:
            # Also try +/- on lr around the best
            for lr_mult in [0.5, 1.0, 2.0]:
                phase2_combos.append((
                    base["rank"], base["lr"] * lr_mult, steps,
                    base["wd"], base["warmup"], base["accum"], base["layers"],
                ))
    # Deduplicate
    phase2_combos = list(dict.fromkeys(phase2_combos))

    print(f"\n{'='*82}")
    print(f"  PHASE 2 — Fine-tune top-3 configs ({len(phase2_combos)} trials, longer training)")
    print(f"{'='*82}\n")

    phase2_results = []
    phase2_t0 = time.time()
    for i, (rank, lr, steps, wd, warmup, accum, layers) in enumerate(phase2_combos, 1):
        print(f"  ── P2 Trial {i}/{len(phase2_combos)}:  rank={rank}  lr={lr:.0e}  "
              f"steps={steps}  wd={wd}  warmup={warmup}  accum={accum}  layers={layers} ──")
        trial_t0 = time.time()
        r = _run_trial(model, tokenizer, train_texts, eval_texts, fp16_ppls, quant_ppls,
                       args, rank=rank, lr=lr, steps=steps, wd=wd,
                       warmup=warmup, accum=accum, layers=layers, non_en=non_en)
        trial_dt = time.time() - trial_t0
        phase2_results.append(r)
        per = "  ".join(f"{l}={r['per_lang_recovery'][l]:+.1f}%" for l in non_en)
        elapsed = time.time() - phase2_t0
        eta = elapsed / i * (len(phase2_combos) - i)
        print(f"    avg: {r['avg_recovery']:+.1f}%  ({per})  "
              f"[{trial_dt/60:.1f}min, ETA {eta/60:.0f}min]\n")

    # ── Final ranking across all trials ──────────────────────────────────────
    all_results = phase1_results + phase2_results
    all_results.sort(key=lambda x: x["avg_recovery"], reverse=True)

    print(f"\n{'='*82}")
    print(f"  FINAL LEADERBOARD — all {len(all_results)} trials")
    print(f"{'='*82}")
    print(f"  {'#':>3} {'rank':>5} {'lr':>9} {'steps':>6} {'wd':>6} {'warm':>5} "
          f"{'acc':>4} {'layers':>12} {'avg':>8}  per-lang")
    print(f"  {'─'*86}")
    for i, r in enumerate(all_results[:15], 1):
        per = "  ".join(f"{l}={r['per_lang_recovery'][l]:+.1f}%" for l in non_en)
        marker = " ★" if i == 1 else ""
        print(f"  {i:>3} {r['rank']:>5} {r['lr']:>9.0e} {r['steps']:>6} {r['wd']:>6} "
              f"{r['warmup']:>5} {r['accum']:>4} {r['layers']:>12} "
              f"{r['avg_recovery']:>+7.1f}%  {per}{marker}")

    best = all_results[0]
    print(f"\n  ★ BEST: rank={best['rank']}  lr={best['lr']:.0e}  steps={best['steps']}  "
          f"wd={best['wd']}  warmup={best['warmup']}  accum={best['accum']}  "
          f"layers={best['layers']}  →  {best['avg_recovery']:+.1f}% recovery")
    print(f"{'='*82}\n")

    # Save all results
    grid_path = Path(args.output).with_suffix(".grid.json")
    grid_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"  Grid results → {grid_path}")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    # Set languages from CLI or default
    global LANGUAGES
    if args.languages:
        LANGUAGES = [l.strip() for l in args.languages.split(",")]
    print(f"  Languages: {', '.join(LANGUAGES)}")

    if args.fast:
        args.num_calibration_samples = 32
        args.num_fp16_samples        = 8
        args.num_quant_samples       = 8
        args.num_train_samples       = 16
        args.num_eval_samples        = 8
        args.steps                   = 50
        print("  [--fast] smoke-test mode: small sample counts, 50 steps/lang\n")

    kw = {"seed": args.seed, "cache_dir": args.cache_dir}

    # ── STEP 1: Load fp16 model ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STEP 1 / 5 — Loading {args.model} (fp16)")
    print(f"{'─'*60}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B params  "
          f"dtype={next(model.parameters()).dtype}")

    # ── STEP 2: fp16 perplexity per language ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STEP 2 / 5 — fp16 perplexity baseline")
    print(f"{'─'*60}\n  Loading data ...")
    fp16_texts = {lang: load_texts(lang, args.num_fp16_samples, **kw)
                  for lang in LANGUAGES}

    print("\n  Measuring fp16 perplexity ...")
    fp16_ppls: Dict[str, float] = {}
    for lang in tqdm(LANGUAGES, desc="  fp16 ppl", unit="lang"):
        ppl = perplexity(model, tokenizer, fp16_texts[lang],
                         args.device, args.max_seq_len, desc=f"fp16 {lang}")
        fp16_ppls[lang] = ppl
        tqdm.write(f"    {lang:<12} ppl={ppl:.2f}")

    # ── STEP 3: GPTQ W3A16 (or load pre-saved) ──────────────────────────────
    print(f"\n{'─'*60}")
    if args.load_quantized_dir:
        print(f"  STEP 3 / 5 — Loading pre-quantized model from {args.load_quantized_dir}")
        print(f"{'─'*60}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.load_quantized_dir, torch_dtype=torch.float16,
            device_map=args.device, trust_remote_code=True,
        ).eval()
        print(f"  Loaded.  dtype={next(model.parameters()).dtype}")
    else:
        print(f"  STEP 3 / 5 — GPTQ W3A16 quantization (English C4)")
        print(f"{'─'*60}\n  Loading calibration data ...")
        cal_texts = load_texts("english", args.num_calibration_samples,
                               seed=args.seed + 1, cache_dir=args.cache_dir)

        print(f"\n  Running GPTQ (group={args.group_size}, "
              f"{args.num_calibration_samples} samples) ...")
        run_gptq(model, tokenizer, cal_texts, args)
        print("  Quantization complete.")

        # Cast any residual BF16 tensors → F16 to avoid Char/BFloat16 mismatch
        # in compressed_tensors' dequantize-then-linear path (INT8/Char weights
        # are skipped because we only cast bfloat16 dtype).
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
        for b in model.buffers():
            if b.dtype == torch.bfloat16:
                b.data = b.data.to(torch.float16)

        # Resolve save dir
        save_qdir = args.save_quantized_dir
        if save_qdir == "auto":
            slug = args.model.replace("/", "_")
            save_qdir = f"lcd_results/{slug}/quantized"
        if save_qdir and save_qdir.lower() != "none":
            qdir = Path(save_qdir)
            qdir.mkdir(parents=True, exist_ok=True)
            print(f"  Saving quantized model → {qdir} ...")
            model.save_pretrained(qdir)
            tokenizer.save_pretrained(qdir)
            print(f"  Quantized model saved.")

    # ── STEP 4: INT3 perplexity + diagnostic ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STEP 4 / 5 — INT3 perplexity + hypothesis test")
    print(f"{'─'*60}\n  Loading measurement data ...")
    quant_texts = {lang: load_texts(lang, args.num_quant_samples,
                                    seed=args.seed + 2, cache_dir=args.cache_dir)
                   for lang in LANGUAGES}

    print("\n  Measuring INT3 perplexity ...")
    quant_ppls: Dict[str, float] = {}
    for lang in tqdm(LANGUAGES, desc="  int3 ppl", unit="lang"):
        ppl = perplexity(model, tokenizer, quant_texts[lang],
                         args.device, args.max_seq_len, desc=f"int3 {lang}")
        quant_ppls[lang] = ppl
        tqdm.write(f"    {lang:<12} ppl={ppl:.2f}  (fp16={fp16_ppls[lang]:.2f}  ratio={ppl/fp16_ppls[lang]:.2f}x)")

    print_diagnostic(fp16_ppls, quant_ppls, args)

    # ── Load train / eval data (shared by single run and grid search) ────────
    print("\n  Loading train / eval data ...")
    train_texts = {lang: load_texts(lang, args.num_train_samples,
                                    seed=args.seed + 3, cache_dir=args.cache_dir)
                   for lang in LANGUAGES}
    eval_texts  = {lang: load_texts(lang, args.num_eval_samples,
                                    seed=args.seed + 4, cache_dir=args.cache_dir)
                   for lang in LANGUAGES}

    # ── STEP 5: Grid search OR single run ────────────────────────────────────
    best = None
    if args.grid_search:
        best = grid_search(model, tokenizer, fp16_ppls, quant_ppls,
                           train_texts, eval_texts, args)
        print(f"\n  Re-running best config for final report ...")
        args.rank  = best["rank"]
        args.lr    = best["lr"]
        args.steps = best["steps"]

    _wd     = best["wd"]      if best else 0.01
    _warmup = best["warmup"]  if best else 0.1
    _accum  = best["accum"]   if best else 1
    _layers = best["layers"]  if best else args.layer_strategy

    print(f"\n{'─'*60}")
    print(f"  STEP 5 / 5 — LCD correction training (rank={args.rank}, layers={_layers})")
    print(f"{'─'*60}")

    print("\n  Attaching LCD hooks ...")
    lcd = LCD(model, rank=args.rank, layer_strategy=_layers)

    corrected_ppls: Dict[str, float] = {}
    non_en = [l for l in LANGUAGES if l != BASELINE]
    for i, lang in enumerate(non_en, 1):
        print(f"\n  ── [{i}/{len(non_en)}] {lang.upper()} ──")
        lcd.train_lang(lang, train_texts[lang], tokenizer,
                       args.device, args.steps, args.lr, args.max_seq_len,
                       weight_decay=_wd, warmup_ratio=_warmup, grad_accum=_accum)

        ppl = lcd.ppl(lang, eval_texts[lang], tokenizer, args.device, args.max_seq_len)
        corrected_ppls[lang] = ppl

        q   = quant_ppls[lang]
        fp  = fp16_ppls[lang]
        gap = q - fp
        rec = (1 - (ppl - fp) / max(gap, 1e-8)) * 100 if gap > 0 else 0.0
        print(f"  ✓ INT3={q:.2f}  LCD={ppl:.2f}  fp16={fp:.2f}  recovery={rec:.1f}%")

    # English = calibration baseline, no correction needed
    corrected_ppls[BASELINE] = quant_ppls[BASELINE]
    print(f"\n  ── {BASELINE.upper()} (skipped — calibration language) ──")
    print(f"  ✓ INT3={quant_ppls[BASELINE]:.2f}  (no correction applied)")

    # ── Final report + save ────────────────────────────────────────────────────
    print_recovery(fp16_ppls, quant_ppls, corrected_ppls, args)
    save_results(fp16_ppls, quant_ppls, corrected_ppls, args, Path(args.output))
    lcd.save(Path(args.save_corrections))

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")


if __name__ == "__main__":
    main()
