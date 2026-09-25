"""Fake-quant GPTQ (weights only), block by block through the decoder.

Algorithm after the reference implementation of Frantar et al. (IST-DASLab/gptq, Apache-2.0: gptq.py `fasterquant`,
quant.py `Quantizer`); the decoder loop is rewritten for transformers 5.x (layer inputs, including
`position_embeddings`, are captured with a forward pre-hook). Weights are quantized and immediately dequantized, so
the result is an ordinary bf16 model; the integer codes are kept to store the model compactly.

Grid (per output row and group of `group_size` input columns, minmax observer):
  sym : scale = absmax / ((2^b - 1) / 2), zero = 2^(b-1)  -> values scale * {-2^(b-1) .. 2^(b-1)-1}
        (identical to compressed-tensors' symmetric int grid used by llm-compressor / LCD)
  asym: scale = (max - min) / (2^b - 1),   zero = round(-min / scale)
`group_params`: "static" computes all group scales from the original weights before the GPTQ sweep, "dynamic"
recomputes a group's scale from the already error-updated weights when the sweep reaches it (reference implementation).
The LCD checkpoints (llm-compressor) use static scales rounded to bf16: for Qwen2.5-3B layers 0 q_proj and 20 down_proj
every stored scale equals bf16(absmax(original group) / 3.5) (checked 2026-09-26 with range requests).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    bits: int = 3
    group_size: int = 128
    sym: bool = True
    percdamp: float = 0.01
    act_order: bool = False
    true_sequential: bool = False
    group_params: str = "static"  # static | dynamic
    scale_bf16: bool = True  # round scales to bf16 (as stored by compressed-tensors)
    blocksize: int = 128


def grid_params(w: torch.Tensor, bits: int, sym: bool, scale_bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row scale / zero for a [rows, cols] slice (minmax). Returns [rows] tensors."""
    maxq = 2**bits - 1
    xmin = torch.minimum(w.min(dim=1).values, torch.zeros(1, device=w.device))
    xmax = torch.maximum(w.max(dim=1).values, torch.zeros(1, device=w.device))
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        scale = xmax / (maxq / 2)
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / torch.where(scale == 0, torch.ones_like(scale), scale))
    if scale_bf16:
        scale = scale.to(torch.bfloat16).float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return scale, zero


def quantize_codes(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1)


def dequantize(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return scale * (codes - zero)


class GPTQ:
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.cols = layer.weight.shape[1]
        self.H = torch.zeros(self.cols, self.cols, device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        inp = inp.reshape(-1, inp.shape[-1]).t().float() if inp.dim() > 1 else inp[:, None].float()
        tmp = 1  # one calibration sequence per call, as in the reference code with batch size 1
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp @ inp.t()

    @torch.no_grad()
    def quantize(self, cfg: QuantConfig) -> dict:
        W = self.layer.weight.data.clone().float()
        rows, cols = W.shape
        gs = cfg.group_size if cfg.group_size > 0 else cols
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        perm = None
        if cfg.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W, H = W[:, perm], H[perm][:, perm]
        n_groups = math.ceil(cols / gs)
        scales = torch.zeros(rows, n_groups, device=W.device)
        zeros = torch.zeros(rows, n_groups, device=W.device)
        if cfg.group_params == "static":
            for g in range(n_groups):
                scales[:, g], zeros[:, g] = grid_params(W[:, g * gs : (g + 1) * gs], cfg.bits, cfg.sym, cfg.scale_bf16)
        elif cfg.group_params != "dynamic":
            raise ValueError(cfg.group_params)

        damp = cfg.percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(cols, device=W.device)
        H[idx, idx] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        codes = torch.zeros_like(W)
        for i1 in range(0, cols, cfg.blocksize):
            i2 = min(i1 + cfg.blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1, C1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1), torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                c = i1 + i
                g = c // gs
                if cfg.group_params == "dynamic" and c % gs == 0:
                    scales[:, g], zeros[:, g] = grid_params(W[:, c : c + gs], cfg.bits, cfg.sym, cfg.scale_bf16)
                w, d = W1[:, i], Hinv1[i, i]
                q_codes = quantize_codes(w, scales[:, g], zeros[:, g], cfg.bits)
                q = dequantize(q_codes, scales[:, g], zeros[:, g])
                Q1[:, i], C1[:, i] = q, q_codes
                err = (w - q) / d
                W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
                Err1[:, i] = err
            Q[:, i1:i2], codes[:, i1:i2] = Q1, C1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        if perm is not None:
            inv = torch.argsort(perm)
            Q, codes = Q[:, inv], codes[:, inv]
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        out = {"codes": codes.to(torch.uint8).cpu(), "scale": scales.cpu(), "zero": zeros.cpu()}
        if perm is not None:
            out["perm"] = perm.cpu()
        return out


class _Stop(Exception):
    pass


def _layer_out(o):
    return o[0] if isinstance(o, tuple) else o


def _groups(names: list[str], true_sequential: bool) -> list[list[str]]:
    if not true_sequential:
        return [names]
    order = [("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",)]
    out = [[n for n in names if n.endswith(suf)] for suf in order]
    rest = [n for n in names if not any(n in g for g in out)]
    return [g for g in out if g] + ([rest] if rest else [])


@torch.no_grad()
def quantize_model(model: nn.Module, calib: list[torch.Tensor], cfg: QuantConfig, log=print) -> dict:
    """Quantize every Linear inside the decoder layers (lm_head / embeddings stay as they are), in place.

    `calib`: list of [1, T] token id tensors (variable length, batch size 1 each). Returns {module name: codes/scale/zero}.
    """
    layers = model.model.layers
    dev = next(model.parameters()).device
    cache: list[tuple[torch.Tensor, dict]] = []

    def catch(_mod, args, kwargs):
        x = args[0] if args else kwargs.pop("hidden_states")
        cache.append((x, dict(kwargs)))
        raise _Stop

    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    for ids in calib:
        try:
            model(ids.to(dev), use_cache=False)
        except _Stop:
            pass
    h.remove()
    inps = [c[0] for c in cache]
    kws = [c[1] for c in cache]

    state = {}
    for li, layer in enumerate(layers):
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        for group in _groups(list(linears), cfg.true_sequential):
            gq = {n: GPTQ(linears[n]) for n in group}
            hooks = [linears[n].register_forward_hook(lambda m, i, o, q=gq[n]: q.add_batch(i[0].detach())) for n in group]
            for x, kw in zip(inps, kws):
                layer(x, **kw)
            for hk in hooks:
                hk.remove()
            for n in group:
                state[f"model.layers.{li}.{n}"] = gq[n].quantize(cfg)
        inps = [_layer_out(layer(x, **kw)) for x, kw in zip(inps, kws)]
        if log:
            log(f"  gptq: layer {li + 1}/{len(layers)} done")
    return state


# ---------------------------------------------------------------------------- compact storage
def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Two codes per byte when bits <= 4."""
    if bits > 4:
        return codes
    flat = codes.flatten()
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    return (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8)


def _unpack(packed: torch.Tensor, shape: tuple[int, ...], bits: int) -> torch.Tensor:
    if bits > 4:
        return packed.reshape(shape)
    lo, hi = packed & 0x0F, packed >> 4
    flat = torch.stack([lo, hi], dim=1).flatten()[: math.prod(shape)]
    return flat.reshape(shape)


def save_quant(state: dict, cfg: QuantConfig, path: str | Path, meta: dict | None = None) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors, shapes = {}, {}
    for name, s in state.items():
        shapes[name] = list(s["codes"].shape)
        tensors[f"{name}.codes"] = _pack(s["codes"], cfg.bits).contiguous()
        tensors[f"{name}.scale"] = s["scale"].contiguous()
        tensors[f"{name}.zero"] = s["zero"].contiguous()
        if "perm" in s:
            tensors[f"{name}.perm"] = s["perm"].contiguous()
    save_file(tensors, str(path / "quant.safetensors"))
    with open(path / "quant_config.json", "w") as f:
        json.dump({"quant": asdict(cfg), "shapes": shapes, "meta": meta or {}}, f, indent=1)


@torch.no_grad()
def apply_quant(model: nn.Module, path: str | Path) -> dict:
    """Overwrite the model's Linear weights with the dequantized weights stored by `save_quant`."""
    from safetensors.torch import load_file

    path = Path(path)
    info = json.loads((path / "quant_config.json").read_text())
    cfg = QuantConfig(**info["quant"])
    t = load_file(str(path / "quant.safetensors"))
    for name, shape in info["shapes"].items():
        lin = model.get_submodule(name)
        lin = getattr(lin, "base", lin)  # CorrectedLinear wraps the Linear
        codes = _unpack(t[f"{name}.codes"], tuple(shape), cfg.bits).float()
        gs = cfg.group_size if cfg.group_size > 0 else shape[1]
        perm = t.get(f"{name}.perm")
        scale = t[f"{name}.scale"].repeat_interleave(gs, dim=1)[:, : shape[1]]
        zero = t[f"{name}.zero"].repeat_interleave(gs, dim=1)[:, : shape[1]]
        if perm is not None:  # groups were formed in act-order
            inv = torch.argsort(perm)
            scale, zero = scale[:, inv], zero[:, inv]
        w = dequantize(codes, scale, zero)
        lin.weight.data.copy_(w.to(lin.weight.dtype))
    return info
