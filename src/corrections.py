"""Language-conditional low-rank corrections (LCD, arXiv:2608.11786, Eq. 1-2):

    y = W_q x + (1/r) A_lang B_lang x,   A = 0,  B ~ N(0, 0.01 std)

`CorrectedLinear` wraps an nn.Linear (not a hook). One (A, B) pair per language in ParameterDicts; `active_lang=None`
switches the correction off. As in the LCD code the correction runs in fp32 (input cast to fp32, result cast back to
the layer dtype) while the base model stays bf16.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn

IGNORE = ("lm_head", "embed_tokens")


class CorrectedLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, langs: list[str], init_std: float = 0.01):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = 1.0 / rank
        self.in_features, self.out_features = base.in_features, base.out_features
        dev = base.weight.device
        self.A = nn.ParameterDict(
            {lang: nn.Parameter(torch.zeros(self.out_features, rank, device=dev)) for lang in langs}
        )
        self.B = nn.ParameterDict(
            {lang: nn.Parameter(torch.empty(rank, self.in_features, device=dev).normal_(0.0, init_std)) for lang in langs}
        )
        self.active_lang: str | None = None

    @property
    def weight(self) -> torch.Tensor:  # code that inspects .weight (dtype, device) keeps working
        return self.base.weight

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        A, B = self.A[self.active_lang], self.B[self.active_lang]
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (x.to(B.dtype) @ B.t()) @ A.t() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.active_lang is None:
            return y
        return y + self.delta(x).to(y.dtype)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, langs={list(self.A.keys())}, active={self.active_lang}"


def block_index(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _placement_filter(placement: str, names: list[str]):
    """all | o_down (o_proj + down_proj) | bottom_half (LCD: blocks 0..max//2) | top_half | blocks:<i>,<j>,..."""
    if placement == "all":
        return lambda n: True
    if placement == "o_down":
        return lambda n: n.endswith(("o_proj", "down_proj"))
    idx = [i for i in (block_index(n) for n in names) if i is not None]
    mid = max(idx) // 2
    if placement == "bottom_half":
        return lambda n: block_index(n) is not None and block_index(n) <= mid
    if placement == "top_half":
        return lambda n: block_index(n) is not None and block_index(n) > mid
    if placement.startswith("blocks:"):
        keep = {int(i) for i in placement.split(":", 1)[1].split(",")}
        return lambda n: block_index(n) in keep
    raise ValueError(f"unknown placement {placement!r}")


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def install(model: nn.Module, rank: int, langs: list[str], placement: str = "all", init_std: float = 0.01) -> list[str]:
    """Freeze the model and wrap the selected Linear layers (never lm_head / embeddings). Returns wrapped names."""
    for p in model.parameters():
        p.requires_grad_(False)
    cands = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and not any(i in n for i in IGNORE)]
    keep = _placement_filter(placement, cands)
    names = [n for n in cands if keep(n)]
    for n in names:
        _set_module(model, n, CorrectedLinear(model.get_submodule(n), rank, langs, init_std))
    return names


def modules(model: nn.Module) -> dict[str, CorrectedLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, CorrectedLinear)}


def remove(model: nn.Module) -> None:
    """Put the original Linear layers back."""
    for n, m in list(modules(model).items()):
        _set_module(model, n, m.base)


def set_lang(model: nn.Module, lang: str | None) -> None:
    for m in modules(model).values():
        if lang is not None and lang not in m.A:
            raise KeyError(f"no correction slot {lang!r}")
        m.active_lang = lang


def trainable(model: nn.Module, lang: str, block: int | None = None) -> list[nn.Parameter]:
    """Enable grads for one language (optionally one block only); everything else frozen."""
    params = []
    for n, m in modules(model).items():
        for d in (m.A, m.B):
            for lg, p in d.items():
                on = lg == lang and (block is None or block_index(n) == block)
                p.requires_grad_(on)
                if on:
                    params.append(p)
    return params


def n_params(model: nn.Module, lang: str) -> int:
    return sum(m.A[lang].numel() + m.B[lang].numel() for m in modules(model).values())


def state(model: nn.Module, langs: list[str] | None = None) -> dict:
    mods = modules(model)
    first = next(iter(mods.values()))
    langs = langs or list(first.A.keys())
    return {
        "rank": first.rank,
        "names": list(mods),
        "tensors": {f"{n}.{ab}.{lg}": getattr(m, ab)[lg].detach().cpu() for n, m in mods.items()
                    for ab in ("A", "B") for lg in langs},  # fmt: skip
        "langs": langs,
    }


def save(model: nn.Module, path: str | Path, langs: list[str] | None = None, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({**state(model, langs), "meta": meta or {}}, tmp)
    tmp.replace(path)


def load(model: nn.Module, path: str | Path, strict: bool = True) -> dict:
    """Load corrections into an installed model (slots must exist). Returns the file's meta."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    mods = modules(model)
    if strict and set(ck["names"]) != set(mods):
        raise ValueError(f"layer mismatch: file has {len(ck['names'])} layers, model {len(mods)}")
    for key, t in ck["tensors"].items():
        n, ab, lg = key.rsplit(".", 2)
        getattr(mods[n], ab)[lg].data.copy_(t)
    return ck.get("meta", {})
