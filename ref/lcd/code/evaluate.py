#!/usr/bin/env python3
"""
LCD Benchmark Evaluation
========================
Evaluate fp16 / INT3-quantized / LCD-corrected models on multilingual benchmarks.

Benchmarks:
  - GlobalMMLU  : multiple-choice accuracy via log-likelihood (42 languages)
  - mArenaHard  : open-ended generation, saved for judge scoring (22 languages)
  - mDolly      : instruction-following, ROUGE-L (en/es/de/fr)

Usage:
    python evaluate.py --model Qwen/Qwen2.5-3B --benchmarks global_mmlu
    python evaluate.py --model ./quantized_model --benchmarks global_mmlu,m_dolly
    python evaluate.py --model ./quantized_model --lcd-corrections lcd_corrections.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Language mapping (benchmark dataset language codes)
# ─────────────────────────────────────────────────────────────────────────────
LANG_MAP = {
    "english":  "en", "arabic":  "ar", "japanese": "ja", "chinese": "zh",
    "hindi":    "hi", "russian": "ru", "french":   "fr", "spanish": "es",
    "korean":   "ko", "german":  "de",
}

# mDolly only has these languages
MDOLLY_LANGS = {"en", "es", "de", "fr"}

# Language order used by LCD corrections (must match pipeline.py ALL_LANGUAGES)
ALL_LANG_ORDER = [
    "english", "arabic", "japanese", "chinese", "hindi",
    "russian", "french", "spanish", "korean", "code",
]


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity measurement (mC4 / CodeParrot)
# ─────────────────────────────────────────────────────────────────────────────
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


def load_texts(lang, n, seed=42, min_words=30, max_chars=2000, cache_dir=None):
    """Load text samples from mC4 or CodeParrot for perplexity measurement."""
    from datasets import load_dataset as _load_dataset

    dataset_name, config, text_field = LANGUAGE_SOURCES[lang]
    print(f"    [{lang:<9}] {dataset_name}/{config}  n={n} ...", end=" ", flush=True)
    try:
        ds = _load_dataset(dataset_name, config, split="train",
                           streaming=True,
                           **({"cache_dir": cache_dir} if cache_dir else {}))
        ds = ds.shuffle(seed=seed, buffer_size=50_000)
    except RuntimeError as e:
        if "scripts are no longer supported" in str(e):
            ds = _load_dataset(dataset_name, config, split="train",
                               streaming=True, revision="refs/convert/parquet",
                               **({"cache_dir": cache_dir} if cache_dir else {}))
            ds = ds.shuffle(seed=seed, buffer_size=50_000)
        else:
            raise

    texts = []
    for sample in ds:
        text = sample[text_field].strip()
        if len(text.split()) < min_words:
            continue
        texts.append(" ".join(text.split()[:max_chars]))
        if len(texts) >= n:
            break

    if len(texts) < n:
        print(f"  Warning: only got {len(texts)}/{n} samples for {lang}")
    else:
        print("OK")
    return texts


def _perplexity(model, tokenizer, texts, device, max_len, desc="ppl"):
    """Compute perplexity over a list of text strings."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for text in tqdm(texts, desc=f"  {desc}", unit="doc", leave=False):
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            if n > 0:
                total_nll += out.loss.item() * n
                total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


def evaluate_perplexity(model, tokenizer, languages, device,
                        num_samples=256, max_seq_len=512, seed=42,
                        cache_dir=None, lcd_state=None):
    """Measure perplexity on mC4 data per language."""
    results = {}
    for lang_name in languages:
        if lang_name not in LANGUAGE_SOURCES:
            print(f"    [ppl] Skipping {lang_name} — no data source")
            continue

        # Set LCD language if corrections are loaded
        if lcd_state is not None:
            _, _, lang_idx = lcd_state
            if lang_name in ALL_LANG_ORDER:
                lang_idx[0] = ALL_LANG_ORDER.index(lang_name)

        texts = load_texts(lang_name, num_samples, seed=seed, cache_dir=cache_dir)
        ppl = _perplexity(model, tokenizer, texts, device, max_seq_len,
                          desc=f"ppl {lang_name}")
        results[lang_name] = round(ppl, 2)
        tqdm.write(f"    {lang_name:<12} ppl={ppl:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GlobalMMLU — log-likelihood multiple choice
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_global_mmlu(model, tokenizer, languages: List[str], device: str,
                         num_samples: Optional[int] = None,
                         num_few_shot: int = 0,
                         lcd_state=None) -> Dict:
    """Evaluate on CohereLabs/Global-MMLU using log-likelihood scoring."""
    from datasets import load_dataset

    results = {}
    option_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
                     for c in "ABCD"]

    for lang_name in languages:
        # Set LCD language index if corrections are loaded
        if lcd_state is not None:
            _, _, lang_idx = lcd_state
            if lang_name in ALL_LANG_ORDER:
                lang_idx[0] = ALL_LANG_ORDER.index(lang_name)
                print(f"    LCD lang_idx set to {lang_idx[0]} ({lang_name})")
        lang_code = LANG_MAP.get(lang_name)
        if not lang_code:
            print(f"    [GlobalMMLU] Skipping {lang_name} — no mapping")
            continue

        try:
            ds = load_dataset("CohereLabs/Global-MMLU", lang_code, split="test")
        except Exception as e:
            print(f"    [GlobalMMLU] Failed to load {lang_code}: {e}")
            continue

        if num_samples and num_samples < len(ds):
            ds = ds.shuffle(seed=42).select(range(num_samples))

        correct = 0
        total = 0
        per_subject = {}

        for item in tqdm(ds, desc=f"  MMLU {lang_name}", unit="q", leave=False):
            question = item["question"]

            # Handle different field formats
            if "choices" in item:
                choices = item["choices"]
            elif "options" in item:
                choices = item["options"]
            elif "option_a" in item:
                choices = [item["option_a"], item["option_b"],
                           item["option_c"], item["option_d"]]
            else:
                continue

            # Skip items with empty or missing options (lighteval filtering)
            if not all(c is not None and str(c).strip() for c in choices):
                continue

            raw_answer = item.get("answer", item.get("answer_index", 0))
            if isinstance(raw_answer, str):
                answer = ord(raw_answer.upper()) - ord("A")
            else:
                answer = int(raw_answer)
            subject = item.get("subject", "unknown")

            # Format prompt
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"

            ids = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=2048).input_ids.to(device)

            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]  # last token logits

            # Score each option token
            option_logprobs = torch.log_softmax(logits, dim=-1)
            scores = [option_logprobs[t].item() for t in option_tokens]
            pred = scores.index(max(scores))

            if pred == answer:
                correct += 1
            total += 1

            if subject not in per_subject:
                per_subject[subject] = {"correct": 0, "total": 0}
            per_subject[subject]["total"] += 1
            if pred == answer:
                per_subject[subject]["correct"] += 1

        acc = correct / max(total, 1)

        # Per-subject accuracy → macro-average and standard deviation
        subject_accs = [v["correct"] / max(v["total"], 1)
                        for v in per_subject.values() if v["total"] > 0]
        macro_acc = float(np.mean(subject_accs)) if subject_accs else 0.0
        macro_std = float(np.std(subject_accs)) if subject_accs else 0.0
        n_subjects = len(subject_accs)
        macro_sem = macro_std / math.sqrt(n_subjects) if n_subjects > 0 else 0.0

        results[lang_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "macro_accuracy": macro_acc,
            "macro_std": macro_std,
            "macro_sem": macro_sem,
            "n_subjects": n_subjects,
            "per_subject": {
                s: {"accuracy": v["correct"] / max(v["total"], 1),
                    "correct": v["correct"], "total": v["total"]}
                for s, v in per_subject.items()
            },
        }
        tqdm.write(f"    {lang_name:<12} acc={acc:.4f}  macro={macro_acc:.4f}±{macro_std:.4f}  "
                   f"({correct}/{total}, {n_subjects} subjects)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# mArenaHard — open-ended generation (save outputs for judge)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_m_arena_hard(model, tokenizer, languages: List[str], device: str,
                          num_samples: Optional[int] = None,
                          max_new_tokens: int = 512) -> Dict:
    """Generate responses for mArenaHard prompts. Returns outputs for judging."""
    from datasets import load_dataset

    results = {}
    try:
        ds = load_dataset("CohereLabs/m-ArenaHard-v2.0", split="train",
                          trust_remote_code=True)
    except Exception as e:
        print(f"    [mArenaHard] Failed to load dataset: {e}")
        return results

    for lang_name in languages:
        lang_code = LANG_MAP.get(lang_name)
        if not lang_code:
            continue

        lang_ds = ds.filter(lambda x: x.get("language", "") == lang_code)
        if len(lang_ds) == 0:
            print(f"    [mArenaHard] No prompts for {lang_name}")
            continue

        if num_samples and num_samples < len(lang_ds):
            lang_ds = lang_ds.shuffle(seed=42).select(range(num_samples))

        outputs = []
        for item in tqdm(lang_ds, desc=f"  Arena {lang_name}", unit="q", leave=False):
            prompt = item["prompt"]
            ids = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=2048).input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=1.0)
            response = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            outputs.append({
                "question_id": item.get("question_id", ""),
                "prompt": prompt,
                "response": response,
                "category": item.get("subcategory", ""),
            })

        results[lang_name] = {
            "num_responses": len(outputs),
            "responses": outputs,
        }
        tqdm.write(f"    {lang_name:<12} generated {len(outputs)} responses")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# mDolly — instruction-following with ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────
def _rouge_l(pred: str, ref: str) -> float:
    """Simple ROUGE-L F1 without external deps."""
    pred_toks = pred.lower().split()
    ref_toks = ref.lower().split()
    if not pred_toks or not ref_toks:
        return 0.0
    # LCS via DP
    m, n = len(pred_toks), len(ref_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_toks[i-1] == ref_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate_m_dolly(model, tokenizer, languages: List[str], device: str,
                     num_samples: Optional[int] = None,
                     max_new_tokens: int = 256) -> Dict:
    """Evaluate on mDolly using ROUGE-L against reference responses."""
    from datasets import load_dataset

    results = {}
    try:
        ds = load_dataset("argilla/databricks-dolly-15k-curated-multilingual",
                          split="train", trust_remote_code=True)
    except Exception as e:
        print(f"    [mDolly] Failed to load dataset: {e}")
        return results

    for lang_name in languages:
        lang_code = LANG_MAP.get(lang_name)
        if not lang_code or lang_code not in MDOLLY_LANGS:
            continue

        # Filter by language field or use translated columns
        lang_ds = ds.filter(lambda x: x.get("language", "en") == lang_code)
        if len(lang_ds) == 0:
            # Try using all samples with the target language instruction column
            lang_ds = ds
        if num_samples and num_samples < len(lang_ds):
            lang_ds = lang_ds.shuffle(seed=42).select(range(num_samples))

        scores = []
        for item in tqdm(lang_ds, desc=f"  Dolly {lang_name}", unit="q", leave=False):
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            reference = item.get("response", "")

            prompt = instruction
            if context:
                prompt = f"Context: {context}\n\nInstruction: {instruction}"

            ids = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=1024).input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=1.0)
            response = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            scores.append(_rouge_l(response, reference))

        avg_rouge = np.mean(scores) if scores else 0.0
        results[lang_name] = {
            "rouge_l": avg_rouge,
            "num_samples": len(scores),
        }
        tqdm.write(f"    {lang_name:<12} ROUGE-L={avg_rouge:.4f}  (n={len(scores)})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LCD correction loading
# ─────────────────────────────────────────────────────────────────────────────
def load_lcd_corrections(model, corrections_path: str, device: str):
    """Load saved LCD corrections and attach hooks to model."""
    import torch.nn as nn

    ckpt = torch.load(corrections_path, map_location=device, weights_only=False)
    rank = ckpt["rank"]
    layer_names = ckpt["layer_names"]

    # Import LangCorrection from pipeline
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import LangCorrection, IGNORE, ALL_LANGUAGES

    corrections = nn.ModuleList()
    hooks = []
    lang_idx = [0]  # mutable for closure

    name_to_mod = dict(model.named_modules())

    for name in layer_names:
        mod = name_to_mod[name]
        corr = LangCorrection(mod.in_features, mod.out_features,
                               n_langs=len(ALL_LANGUAGES), rank=rank).to(device)
        corrections.append(corr)

        def make_hook(c):
            def h(mod, inp, out):
                return out + c(inp[0], lang_idx[0]).to(out.dtype)
            return h
        hooks.append(mod.register_forward_hook(make_hook(corr)))

    corrections.load_state_dict(ckpt["corrections"])
    print(f"  LCD corrections loaded: {len(corrections)} layers, rank={rank}")
    return corrections, hooks, lang_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF model ID or local path to model directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--benchmarks", default="global_mmlu",
                   help="Comma-separated: global_mmlu, perplexity, m_arena_hard, m_dolly")
    p.add_argument("--languages",
                   default="english,arabic,japanese,chinese,hindi,russian,french,spanish,korean",
                   help="Comma-separated language names")
    p.add_argument("--num-samples", default=None, type=int,
                   help="Max samples per language per benchmark (None = all)")
    p.add_argument("--lcd-corrections", default=None,
                   help="Path to LCD corrections .pt file (optional)")
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--max-new-tokens", default=512, type=int)
    p.add_argument("--num-ppl-samples", default=256, type=int,
                   help="Samples per language for perplexity measurement")
    p.add_argument("--ppl-max-seq-len", default=512, type=int,
                   help="Max sequence length for perplexity evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    languages = [l.strip() for l in args.languages.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    print(f"\n  Model:      {args.model}")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Languages:  {', '.join(languages)}")

    # Load model
    print(f"\n  Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Optionally attach LCD corrections
    lcd_state = None
    if args.lcd_corrections:
        lcd_state = load_lcd_corrections(model, args.lcd_corrections, args.device)

    # Run benchmarks
    all_results = {"model": args.model, "languages": languages}

    if "perplexity" in benchmarks:
        print(f"\n{'─'*60}")
        print(f"  Perplexity evaluation (mC4)")
        print(f"{'─'*60}")
        all_results["perplexity"] = evaluate_perplexity(
            model, tokenizer, languages, args.device,
            args.num_ppl_samples, args.ppl_max_seq_len,
            lcd_state=lcd_state)

    if "global_mmlu" in benchmarks:
        print(f"\n{'─'*60}")
        print(f"  GlobalMMLU evaluation")
        print(f"{'─'*60}")
        all_results["global_mmlu"] = evaluate_global_mmlu(
            model, tokenizer, languages, args.device, args.num_samples,
            lcd_state=lcd_state)

    if "m_arena_hard" in benchmarks:
        print(f"\n{'─'*60}")
        print(f"  mArenaHard evaluation")
        print(f"{'─'*60}")
        all_results["m_arena_hard"] = evaluate_m_arena_hard(
            model, tokenizer, languages, args.device,
            args.num_samples, args.max_new_tokens)

    if "m_dolly" in benchmarks:
        print(f"\n{'─'*60}")
        print(f"  mDolly evaluation")
        print(f"{'─'*60}")
        all_results["m_dolly"] = evaluate_m_dolly(
            model, tokenizer, languages, args.device,
            args.num_samples, args.max_new_tokens)

    # Save
    Path(args.output).write_text(json.dumps(all_results, indent=2, default=str))
    elapsed = time.time() - t0
    print(f"\n  Results → {args.output}")
    print(f"  Wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
