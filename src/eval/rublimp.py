"""RuBLiMP: a pair counts as correct when the grammatical sentence (`source_sentence`) gets a higher total log-prob
than the ungrammatical one (`target_sentence`). Sentences are scored after a single prefix token, as lm-eval does for an
empty context: BOS if the tokenizer has one (Llama), else EOS (Qwen)."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from ..data import RUBLIMP, SPLITS


def prefix_id(tok) -> int:
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


@torch.no_grad()
def sentence_logprobs(model, tok, sents: list[str], batch_size: int = 32) -> list[float]:
    dev = next(model.parameters()).device
    pre = prefix_id(tok)
    enc = [[pre] + tok.encode(s, add_special_tokens=False) for s in sents]
    out = [0.0] * len(sents)
    order = sorted(range(len(sents)), key=lambda i: len(enc[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        T = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, : len(enc[i])] = torch.tensor(enc[i])
            mask[r, : len(enc[i])] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for r, i in enumerate(idx):
            n = len(enc[i])
            lp = F.log_softmax(logits[r, : n - 1].float(), -1).gather(1, ids[r, 1:n, None]).sum()
            out[i] = lp.item()
    return out


def evaluate(model, tok, subsample: str | None = "rublimp_sub200", batch_size: int = 32, n_per: int | None = None) -> dict:
    """subsample: name of a data/splits file with row ids per phenomenon, or None for the full set.
    n_per: only the first n_per of those rows (smoke runs)."""
    from datasets import get_dataset_config_names, load_dataset

    rows = None
    if subsample:
        with open(SPLITS / f"{subsample}.json") as f:
            rows = json.load(f)["rows"]
    per = {}
    for cfg in sorted(rows) if rows else sorted(get_dataset_config_names(RUBLIMP)):
        ds = load_dataset(RUBLIMP, cfg, split="train")
        if rows:
            ds = ds.select(rows[cfg][:n_per] if n_per else rows[cfg])
        elif n_per:
            ds = ds.select(range(min(n_per, len(ds))))
        good = sentence_logprobs(model, tok, list(ds["source_sentence"]), batch_size)
        bad = sentence_logprobs(model, tok, list(ds["target_sentence"]), batch_size)
        per[cfg] = float(np.mean([g > b for g, b in zip(good, bad)]))
    return {"acc": float(np.mean(list(per.values()))), "per_phenomenon": per, "subsample": subsample}
