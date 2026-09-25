"""Training objectives O1-O3, O5 (end-to-end) and the blockwise target of O4-alpha; hidden-state capture.

Notation (PLAN.md §4): T teacher (bf16), S quantized student with corrections, h_l residual stream after block l,
e_l = mean_t ||h_l^S - h_l^T||^2 / (||h_l^T||^2 + eps) over positions t >= k that are not massive-activation outliers.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


@contextmanager
def capture_blocks(model: nn.Module, store: list):
    """Record the output (residual stream) of every decoder block, in order, during a forward pass."""
    store.clear()
    layers = model.model.layers
    hooks = [
        layer.register_forward_hook(lambda m, i, o, k=k: store.__setitem__(k, o[0] if isinstance(o, tuple) else o))
        for k, layer in enumerate(layers)
    ]
    store.extend([None] * len(layers))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def position_mask(h_teacher: list[torch.Tensor], attn_mask: torch.Tensor | None, skip_first: int = 1,
                  outlier_factor: float = 10.0) -> list[torch.Tensor]:  # fmt: skip
    """Per block [B, T] bool mask: real tokens, t >= skip_first, ||h^T(t)|| <= factor * median over the block."""
    out = []
    for h in h_teacher:
        B, T, _ = h.shape
        m = torch.ones(B, T, dtype=torch.bool, device=h.device) if attn_mask is None else attn_mask.bool().clone()
        m[:, :skip_first] = False
        if outlier_factor:
            norm = h.float().norm(dim=-1)
            med = norm[m].median() if m.any() else norm.median()
            m &= norm <= outlier_factor * med
        out.append(m)
    return out


def normalized_err(h_s: torch.Tensor, h_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """e_l for one block (mean over masked positions)."""
    d = (h_s.float() - h_t.float()).pow(2).sum(-1) / (h_t.float().pow(2).sum(-1) + EPS)
    return (d * mask).sum() / mask.sum().clamp(min=1)


def lm_loss(logits: torch.Tensor, ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean next-token CE (as HF `model(ids, labels=ids).loss`)."""
    lg = logits[:, :-1].float()
    tgt = ids[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
    m = torch.ones_like(tgt, dtype=torch.bool) if attn_mask is None else attn_mask[:, 1:].bool()
    return (ce * m).sum() / m.sum().clamp(min=1)


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, attn_mask: torch.Tensor | None = None,
            temperature: float = 1.0) -> torch.Tensor:  # fmt: skip
    """mean_t KL(p^T || p^S) over real positions (all positions predict a next token)."""
    ls = F.log_softmax(s_logits.float() / temperature, dim=-1)
    lt = F.log_softmax(t_logits.float() / temperature, dim=-1)
    kl = (lt.exp() * (lt - ls)).sum(-1)
    m = torch.ones_like(kl, dtype=torch.bool) if attn_mask is None else attn_mask.bool()
    return (kl * m).sum() / m.sum().clamp(min=1) * temperature**2


class Objective:
    """End-to-end objective. `needs_teacher` tells the trainer to run the teacher forward pass."""

    needs_teacher = False
    needs_hidden = False

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, ids, attn, s_logits, s_hidden, t_logits=None, t_hidden=None) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


class O1(Objective):
    def __call__(self, ids, attn, s_logits, s_hidden, t_logits=None, t_hidden=None):
        loss = lm_loss(s_logits, ids, attn)
        return loss, {"ce": loss.item()}


class O2(Objective):
    needs_teacher = True

    def __call__(self, ids, attn, s_logits, s_hidden, t_logits=None, t_hidden=None):
        loss = kd_loss(s_logits, t_logits, attn, self.kw.get("temperature", 1.0))
        return loss, {"kl": loss.item()}


class O3(Objective):
    """KL + lambda * (1/L) sum_l w_l e_l (+ mu * CE)."""

    needs_teacher = True
    needs_hidden = True

    def __call__(self, ids, attn, s_logits, s_hidden, t_logits=None, t_hidden=None):
        lam, mu = self.kw.get("lam", 1.0), self.kw.get("mu", 0.0)
        masks = position_mask(t_hidden, attn, self.kw.get("skip_first", 1), self.kw.get("outlier_factor", 10.0))
        w = self.kw.get("layer_weights") or [1.0] * len(t_hidden)
        errs = [normalized_err(hs, ht, m) for hs, ht, m in zip(s_hidden, t_hidden, masks)]
        hid = sum(wl * e for wl, e in zip(w, errs)) / len(errs)
        kl = kd_loss(s_logits, t_logits, attn)
        loss = kl + lam * hid
        logs = {"kl": kl.item(), "hid": hid.item()}
        if mu:
            ce = lm_loss(s_logits, ids, attn)
            loss = loss + mu * ce
            logs["ce"] = ce.item()
        return loss, logs


class O5(Objective):
    """RILQ-style: 0.5 * e_L (last block) + 0.5 * CE."""

    needs_teacher = True
    needs_hidden = True

    def __call__(self, ids, attn, s_logits, s_hidden, t_logits=None, t_hidden=None):
        m = position_mask(t_hidden[-1:], attn, self.kw.get("skip_first", 1), self.kw.get("outlier_factor", 10.0))[0]
        e = normalized_err(s_hidden[-1], t_hidden[-1], m)
        ce = lm_loss(s_logits, ids, attn)
        return 0.5 * e + 0.5 * ce, {"hid_last": e.item(), "ce": ce.item()}


OBJECTIVES = {"O1": O1, "O2": O2, "O3": O3, "O5": O5}


def make(name: str, **kw) -> Objective:
    return OBJECTIVES[name](**kw)


def blockwise_target(h_teacher_l: torch.Tensor, teacher_block_on_student_input: torch.Tensor, alpha: float):
    """O4-alpha target: alpha * h_l^T (propagated) + (1 - alpha) * Block_l^T(X_l) (local)."""
    return alpha * h_teacher_l + (1.0 - alpha) * teacher_block_on_student_input
