"""Phase 2 diagnostics: where the quantization error lives and how it travels (PLAN.md §7, phase 2).

Per decoder block l (normalized error e, positions t >= skip_first without massive-activation outliers):
  e_acc   = e(h_l^S, h_l^T)                              accumulated: both models run end to end
  e_locT  = e(Block_l^S(h_{l-1}^T), h_l^T)                local: the student's block on the teacher's input
  e_locS  = e(h_l^S, Block_l^T(h_{l-1}^S))                the block's own error on the (already corrupted) student input
The propagated share is what e_acc has on top of the local error.

Per Linear (relative Frobenius error over the same positions):
  e2e     = ||y^S - y^T|| / ||y^T||, each model on its own input (LCD's layer-wise metric)
  local   = ||Lin^S(x^T) - y^T|| / ||y^T||, the student's layer on the teacher's input (pure weight error)
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..objectives import normalized_err, position_mask


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@contextmanager
def _block_io(model: nn.Module, store: dict):
    """Record input hidden state + kwargs and output of every decoder block."""
    hooks = []
    for k, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, k=k):
            x = args[0] if args else kwargs["hidden_states"]
            kw = {n: v for n, v in kwargs.items() if n not in ("hidden_states", "past_key_values", "past_key_value")}
            store.setdefault("in", {})[k] = (x, kw)

        def post(mod, args, kwargs, out, k=k):
            store.setdefault("out", {})[k] = _out(out)

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.register_forward_hook(post, with_kwargs=True))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def _linears(model: nn.Module) -> dict[str, nn.Module]:
    """Decoder Linear layers by name; a CorrectedLinear stands for its wrapped base."""
    out = {}
    for n, m in model.named_modules():
        if ".layers." not in f".{n}" or n.endswith(".base"):
            continue
        if isinstance(m, nn.Linear) or type(m).__name__ == "CorrectedLinear":
            out[n] = m
    return out


@torch.no_grad()
def block_diagnostics(student: nn.Module, teacher: nn.Module, tok, texts: list[str], max_len: int = 512,
                      skip_first: int = 1, outlier_factor: float = 10.0, linears: bool = True) -> dict:  # fmt: skip
    dev = next(student.parameters()).device
    L = len(teacher.model.layers)
    acc, locT, locS = torch.zeros(L), torch.zeros(L), torch.zeros(L)
    s_lin, t_lin = _linears(student), _linears(teacher)
    num_e2e: dict[str, float] = {}
    num_loc: dict[str, float] = {}
    den: dict[str, float] = {}
    y_teacher: dict[str, torch.Tensor] = {}
    cur_mask: dict[str, torch.Tensor] = {}

    def t_hook(name):
        def h(mod, inp, out):
            m = cur_mask["m"][..., None]
            y_teacher[name] = out
            y_loc = s_lin[name](inp[0])
            den[name] = den.get(name, 0.0) + (out.float() * m).pow(2).sum().item()
            num_loc[name] = num_loc.get(name, 0.0) + ((y_loc.float() - out.float()) * m).pow(2).sum().item()
        return h

    def s_hook(name):
        def h(mod, inp, out):
            m = cur_mask["m"][..., None]
            num_e2e[name] = num_e2e.get(name, 0.0) + ((out.float() - y_teacher[name].float()) * m).pow(2).sum().item()
        return h

    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
        base = torch.ones(ids.shape, dtype=torch.bool, device=dev)
        base[:, :skip_first] = False
        cur_mask["m"] = base
        th = [t_lin[n].register_forward_hook(t_hook(n)) for n in t_lin] if linears else []
        T: dict = {}
        with _block_io(teacher, T):  # no KV cache: re-running a block must not see appended keys
            teacher(input_ids=ids, use_cache=False)
        for h in th:
            h.remove()
        sh = [s_lin[n].register_forward_hook(s_hook(n)) for n in s_lin] if linears else []
        S: dict = {}
        with _block_io(student, S):
            student(input_ids=ids, use_cache=False)
        for h in sh:
            h.remove()
        y_teacher.clear()
        t_out = [T["out"][k] for k in range(L)]
        masks = position_mask(t_out, None, skip_first, outlier_factor)
        for k in range(L):
            xt, kwt = T["in"][k]
            xs, kws = S["in"][k]
            y_st = _out(student.model.layers[k](xt, **kwt))
            y_ts = _out(teacher.model.layers[k](xs, **kws))
            acc[k] += normalized_err(S["out"][k], t_out[k], masks[k]).item()
            locT[k] += normalized_err(y_st, t_out[k], masks[k]).item()
            locS[k] += normalized_err(S["out"][k], y_ts, masks[k]).item()
    n = max(len(texts), 1)
    out = {"e_acc": (acc / n).tolist(), "e_locT": (locT / n).tolist(), "e_locS": (locS / n).tolist(), "n_docs": len(texts)}
    if linears:
        out["lin_e2e"] = {k: (num_e2e[k] / den[k]) ** 0.5 for k in den}
        out["lin_local"] = {k: (num_loc[k] / den[k]) ** 0.5 for k in den}
    return out
