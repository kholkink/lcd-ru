"""Base models: FP16 (= original bf16 weights), INT3-ours (our fake-quant GPTQ) and INT3-LCD (author's checkpoints).

All three end up as the same HF architecture in bf16, so corrections, hooks and evaluation code never need to know
which one they got. INT3-LCD checkpoints (llm-compressor, compressed-tensors "naive-quantized": int8 containers with
codes in [-4, 3] plus bf16 `weight_scale` per group of 128, no zero point) are decoded by hand:
W = codes * scale (one rounding to bf16, identical to compressed-tensors' dequantize).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

LCD_INT3 = {
    "Qwen/Qwen2.5-3B": "nrmlthms/qwen2.5-3b-int3-gptq",
    "meta-llama/Llama-3.2-3B": "nrmlthms/llama3.2-3b-int3-gptq",
    # Meta declined gated access (2026-09-26): the unsloth mirror has byte-identical weights (same safetensors sha256)
    # and identical tokenization (checked on 1176 docs + MMLU prompts); only pad_token_id / metadata differ.
    "unsloth/Llama-3.2-3B": "nrmlthms/llama3.2-3b-int3-gptq",
}


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def default_dtype() -> torch.dtype:
    """bf16 everywhere; LCDRU_DTYPE=float32 only for local CPU dry runs (bf16 backward is emulated on this CPU)."""
    return getattr(torch, os.environ.get("LCDRU_DTYPE", "bfloat16"))


def load_hf(model_id: str, device: str | torch.device, dtype: torch.dtype | None = None) -> nn.Module:
    from transformers import AutoModelForCausalLM

    dtype = dtype or default_dtype()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


@torch.no_grad()
def apply_lcd_int3(model: nn.Module, repo: str) -> dict:
    """Overwrite the decoder Linear weights with the decoded INT3 weights of an LCD checkpoint."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())["quantization_config"]
    w = cfg["config_groups"]["group_0"]["weights"]
    assert (w["num_bits"], w["group_size"], w["symmetric"], w["strategy"]) == (3, 128, True, "group"), w
    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors")]
    n = 0
    for fname in files:
        with safe_open(hf_hub_download(repo, fname), framework="pt") as f:
            keys = set(f.keys())
            for k in keys:
                if not k.endswith(".weight_scale"):
                    continue
                name = k[: -len(".weight_scale")]
                codes = f.get_tensor(name + ".weight").float()
                scale = f.get_tensor(k).float()
                lin = model.get_submodule(name)
                lin = getattr(lin, "base", lin)
                deq = codes * scale.repeat_interleave(w["group_size"], dim=1)[:, : codes.shape[1]]
                lin.weight.data.copy_(deq.to(lin.weight.dtype))
                n += 1
    return {"repo": repo, "n_linear": n, "scheme": w}


def load_base(model_id: str, base: dict, device: str | torch.device) -> nn.Module:
    """base = {kind: fp16} | {kind: lcd-int3[, repo]} | {kind: gptq-ours, path: <dir with quant.safetensors>}"""
    from .quant.gptq import apply_quant

    model = load_hf(model_id, device)
    kind = base.get("kind", "fp16")
    if kind == "fp16":
        pass
    elif kind == "lcd-int3":
        info = apply_lcd_int3(model, base.get("repo") or LCD_INT3[model_id])
        print(f"  INT3-LCD: {info['n_linear']} linear layers from {info['repo']}")
    elif kind == "gptq-ours":
        apply_quant(model, base["path"])
    else:
        raise ValueError(f"unknown base kind {kind!r}")
    return model
