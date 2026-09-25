"""Perplexity as in LCD (ref/lcd/code/evaluate.py `_perplexity`): each doc truncated to `max_len` tokens (tokenizer
default special tokens: Llama adds BOS, Qwen nothing), no sliding window, token-weighted mean NLL. Batched with right
padding; per-doc NLL sums are kept for bootstrap."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def batches(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def encode(tok, texts: list[str], max_len: int, side: str = "right", dev=None):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True, padding_side=side)
    return enc.input_ids.to(dev), enc.attention_mask.to(dev)


def token_nll(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, T-1] NLL of each next token (0 on padding), computed one sequence at a time in fp32."""
    out = torch.zeros(ids.shape[0], ids.shape[1] - 1, device=logits.device)
    for b in range(ids.shape[0]):
        out[b] = F.cross_entropy(logits[b, :-1].float(), ids[b, 1:], reduction="none")
    return out * mask[:, 1:]


@torch.no_grad()
def perplexity(model, tok, texts: list[str], max_len: int = 512, batch_size: int = 8) -> dict:
    dev = next(model.parameters()).device
    per_doc = []
    for chunk in batches(texts, batch_size):
        ids, mask = encode(tok, chunk, max_len, "right", dev)
        nll = token_nll(model(input_ids=ids, attention_mask=mask).logits, ids, mask)
        for b in range(ids.shape[0]):
            per_doc.append((nll[b].sum().item(), int(mask[b, 1:].sum().item())))
    total_nll = sum(d[0] for d in per_doc)
    total_tok = sum(d[1] for d in per_doc)
    return {"ppl": math.exp(total_nll / max(total_tok, 1)), "ce": total_nll / max(total_tok, 1),
            "tokens": total_tok, "n_docs": len(texts), "per_doc": per_doc}  # fmt: skip
