"""Teacher-student agreement on held-out text: KL(p^T || p^S), top-1 agreement, e_l per block, and the relative
output error of every corrected/quantized Linear (||y^T - y^S||_F / ||y^T||_F, as in LCD's layer-wise analysis;
end-to-end: each model runs its own forward pass)."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..objectives import capture_blocks, normalized_err, position_mask
from .ppl import batches, encode


@contextmanager
def capture_linears(model: nn.Module, store: dict):
    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and ".layers." in f".{n}" and not n.endswith(".base"):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: store.__setitem__(n, o)))
    for n, m in model.named_modules():  # CorrectedLinear: record the corrected output, not the inner base
        if type(m).__name__ == "CorrectedLinear":
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: store.__setitem__(n, o)))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def teacher_agreement(student, teacher, tok, texts: list[str], max_len: int = 512, batch_size: int = 4,
                      skip_first: int = 1, outlier_factor: float = 10.0, linears: bool = False) -> dict:  # fmt: skip
    dev = next(student.parameters()).device
    if linears:  # all Linear outputs of one batch are held at once
        batch_size = 1
    kl_sum = agree = n_pos = 0.0
    L = len(student.model.layers)
    e_sum = torch.zeros(L)
    e_cnt = 0
    lin_num: dict[str, float] = {}
    lin_den: dict[str, float] = {}
    hs_t, hs_s, ls_t, ls_s = [], [], {}, {}
    for chunk in batches(texts, batch_size):
        ids, mask = encode(tok, chunk, max_len, "right", dev)
        with capture_blocks(teacher, hs_t), capture_linears(teacher, ls_t) if linears else _null():
            t_logits = teacher(input_ids=ids, attention_mask=mask).logits
        with capture_blocks(student, hs_s), capture_linears(student, ls_s) if linears else _null():
            s_logits = student(input_ids=ids, attention_mask=mask).logits
        m = mask.bool()
        for b in range(ids.shape[0]):
            lt = F.log_softmax(t_logits[b].float(), -1)
            ls = F.log_softmax(s_logits[b].float(), -1)
            kl = (lt.exp() * (lt - ls)).sum(-1)
            kl_sum += kl[m[b]].sum().item()
            agree += (lt.argmax(-1) == ls.argmax(-1))[m[b]].sum().item()
            n_pos += m[b].sum().item()
        masks = position_mask(hs_t, mask, skip_first, outlier_factor)
        e_sum += torch.tensor([normalized_err(s, t, mk).item() for s, t, mk in zip(hs_s, hs_t, masks)])
        e_cnt += 1
        if linears:
            for n, yt in ls_t.items():
                ys = ls_s[n]
                mm = m[..., None]
                lin_num[n] = lin_num.get(n, 0.0) + ((ys.float() - yt.float()) * mm).pow(2).sum().item()
                lin_den[n] = lin_den.get(n, 0.0) + (yt.float() * mm).pow(2).sum().item()
    e = (e_sum / max(e_cnt, 1)).tolist()
    out = {"kl": kl_sum / n_pos, "top1": agree / n_pos, "positions": int(n_pos), "e_l": e, "e_mean": sum(e) / len(e)}
    if linears:
        out["linear_rel_err"] = {n: (lin_num[n] / lin_den[n]) ** 0.5 for n in lin_num}
    return out


@contextmanager
def _null():
    yield
