#!/usr/bin/env bash
# Once per Colab session (run by colab/worker.py): deps, GPU check, `data` artifact (split texts) from Kaggle.
# Idempotent. Env from the worker: LCDRU_TEXTS, LCDRU_RUNS, HF_HOME (all local), HF_TOKEN from the notebook.
set -euo pipefail
cd "$(dirname "$0")/.."
t0=$(date +%s)
say() { echo "[setup +$(( $(date +%s) - t0 ))s] $*"; }

nvidia-smi || true
say "pip (torch stays Colab's CUDA build)"
grep -v '^#' requirements.txt | grep -v '^\s*$' > /tmp/req.txt
pip install -q -r /tmp/req.txt 2>&1 | grep -v "already satisfied" || true
python - <<'PY'
import torch, transformers, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
    assert torch.cuda.is_bf16_supported(), "this GPU has no bf16 (T4?): use L4 / A100"
PY
mkdir -p "$LCDRU_TEXTS" "$LCDRU_RUNS" "$HF_HOME"
say "artifacts from Kaggle"
python colab/artifacts.py restore ${LCDRU_SETUP_ARTIFACTS:-data}
say "ready: $(nproc) CPUs, $(free -g | awk '/Mem/{print $2}') GB RAM, $(df -h /content | awk 'NR==2{print $4}') free disk"
