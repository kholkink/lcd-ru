# LCD — Language-Conditional Dequantization

Per-language rank-2 LoRA corrections that recover most of the multilingual
quality gap introduced by English-calibrated INT3 GPTQ quantization.

> ⚠️ **Anonymous submission notice.** This repository contains code and results
> accompanying a paper under review. Author identity will be revealed on
> acceptance.

## Quick start

```bash
pip install -r requirements.txt

# Diagnose + train + evaluate on a single model
python pipeline.py --model Qwen/Qwen2.5-3B --device cuda:0
python pipeline.py --model meta-llama/Llama-3.2-3B --device cuda:0
```

This runs the full pipeline: load FP16 → measure per-language perplexity →
GPTQ W3A16 quantize → measure INT3 perplexity → attach per-language rank-2
LoRA corrections → train each correction (500 steps, 256 samples) →
evaluate corrected perplexity. Output is written to
`lcd_results/<model>/lcd_results.json` plus a `.pt` checkpoint of the
corrections.

## Repository layout

```
.
├── pipeline.py                  Main LCD pipeline (diagnose + train + evaluate)
├── evaluate.py                  GlobalMMLU evaluation harness
├── run_lcd.py                   Wrapper to load saved corrections & evaluate
├── run_evals.py                 Batch evaluation over saved checkpoints
├── run_all.sh                   End-to-end driver for both models
├── exp_agnostic_baseline.py     Per-lang vs shared LoRA comparison
├── exp_rank_ablation.py         Rank r ∈ {1, 2, 4} ablation
├── exp_layerwise_error.py       Per-layer L2 quantization error per language
├── exp_se_significance.py       Binomial SE + paired t-test on MMLU results
├── lcd_results/                 Per-model perplexity, MMLU, and ablation JSON
├── paper/                       LaTeX source for the paper
└── requirements.txt
```

## Reproducing the paper results

The numbers in the paper come from these commands:

```bash
# Main results (Tables 1, 2, 3)
python pipeline.py --model Qwen/Qwen2.5-3B
python pipeline.py --model meta-llama/Llama-3.2-3B

# Downstream MMLU eval
python evaluate.py --model Qwen/Qwen2.5-3B --benchmark global_mmlu
python evaluate.py --model meta-llama/Llama-3.2-3B --benchmark global_mmlu

# Per-language vs agnostic baseline (Table 4)
python exp_agnostic_baseline.py --model Qwen/Qwen2.5-3B

# Rank ablation (Section 5.4)
python exp_rank_ablation.py --model Qwen/Qwen2.5-3B

# Layer-wise quantization error analysis (Figure 1)
python exp_layerwise_error.py --model Qwen/Qwen2.5-3B
python exp_layerwise_error.py --model meta-llama/Llama-3.2-3B

# Statistical significance for Table 3
python exp_se_significance.py
```

## Hardware

All experiments run on a single GPU. Reproducing the full grid:

| Component | Wall-clock | Hardware |
|-----------|-----------|----------|
| Per-model pipeline (3B) | ~3 h | L4 24 GB or L40S 48 GB |
| GlobalMMLU eval per model | ~1 h | same |
| Layer-wise analysis per model | ~30 min | L40S (needs two models in VRAM) |

Total reported compute: ~42 GPU-hours.

## Data

- **C4 / mC4** (perplexity evaluation, calibration, adapter training) — public, AllenAI.
- **GlobalMMLU** (downstream accuracy) — public; 14k+ items per language across 57 subjects.
- **CodeParrot github-code-clean** (Code language, optional) — public.

No human subjects, no proprietary data, no PII.

## Citation

```
[Citation will be added on acceptance.]
```

## License

Apache License 2.0 — see `LICENSE`.
