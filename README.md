# lcd-ru

Language-conditional low-rank corrections for GPTQ-W3 LLMs (Russian): what do they repair, and why accuracy does not
come back with perplexity. Extends LCD (arXiv:2608.11786) with KD / hidden-state / blockwise objectives.

- Plan (RU, phase-gated): `PLAN.md`. Review that shaped it: `reports/plan_review.md`. Changes: `reports/CHANGELOG.md`.
- LCD paper + code for reference: `ref/lcd/` (not imported).

## Layout

| Path | What |
|---|---|
| `src/corrections.py` | `CorrectedLinear` (per-language rank-r A·B, fp32), install / switch / save / load |
| `src/quant/gptq.py` | fake-quant GPTQ (INT3-ours), LCD settings: sym, g128, static bf16 scales |
| `src/models.py` | base models: FP16 (bf16), INT3-LCD (author's checkpoints, decoded by hand), INT3-ours |
| `src/objectives.py`, `src/train.py` | O1 / O2 / O3 / O5 losses, LCD training loop with checkpoints + resume |
| `src/eval/` | perplexity, teacher KL / e_l / per-linear error, Global-MMLU (LCD scorer), RuBLiMP, stats |
| `src/data.py`, `data/splits/` | fixed splits (ids + sha1); texts in `data/texts/` (not committed; Kaggle artifact `data`) |
| `configs/` | one YAML per experiment |
| `scripts/` | `make_splits.py`, `quantize.py`, `train.py`, `evaluate.py`, `smoke.py` |
| `colab/`, `kaggle/` | Colab worker + job queue + artifacts on private Kaggle datasets (pattern from `~/casmi26`) |
| `results/` | append-only jsonl collected from runs |

## Local (CPU, tests only)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest              # tiny random models, ~40 s
.venv/bin/python -m pytest -m network   # LCD data loader == reference code (streams C4)
```

## Colab

1. Push the code: `bash kaggle/push_code.sh` (committed tree; writes `GIT_HASH`).
2. Open `colab/worker.ipynb` in Colab (L4/A100, secrets `KAGGLE_API_TOKEN`, `HF_TOKEN`), Run all.
3. Queue jobs locally, e.g. the phase 0 smoke run:

```bash
.venv/bin/python colab/jobs.py add --id p0-smoke --gpu --timeout-h 2 --save runs/p0_smoke \
    --results '$LCDRU_RUNS/p0_smoke*/*.jsonl' --cmd "python scripts/smoke.py --config configs/p0_smoke.yaml"
.venv/bin/python colab/jobs.py wait --id p0-smoke
```

## Licenses

Code: see `ref/lcd/code/LICENSE` (LCD, Apache-2.0) and the IST-DASLab GPTQ reference (Apache-2.0) for the parts that
follow them. Models: Qwen2.5-3B is under the Qwen Research License; Llama 3.2 under the Llama 3.2 Community License.
Gazeta texts are never stored (non-commercial, personal use only).
