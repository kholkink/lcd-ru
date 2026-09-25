"""Global-MMLU with LCD's scorer (ref/lcd/code/evaluate.py `evaluate_global_mmlu`), batched.

Prompt (English template for every language, 0-shot):
    Question: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:
Tokenized with the tokenizer's default special tokens, truncated (right side) to 2048 tokens; the prediction is the
argmax of the next-token log-probs of " A" .. " D" (last token of `encode(" X")`). Items with an empty option are skipped
as in LCD (ru: 14,041 of 14,042). Batches are left-padded with explicit position ids. The four log-probs of every item
are kept: R_KL, flip rate, C3 and the bootstrap all work from them.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import load_mmlu, mmlu_dev_ids

LETTERS = "ABCD"


def build_items(lang: str, split: str = "full", n: int | None = None, seed: int = 0) -> list[dict]:
    """split: full | dev | test (= full minus dev). n: optional fixed random subsample of the split."""
    ds = load_mmlu(lang)
    dev = mmlu_dev_ids() if split in ("dev", "test") else None
    items = []
    for it in ds:
        choices = [it["option_a"], it["option_b"], it["option_c"], it["option_d"]]
        if not all(c is not None and str(c).strip() for c in choices):
            continue
        if split == "dev" and it["sample_id"] not in dev or split == "test" and it["sample_id"] in dev:
            continue
        prompt = f"Question: {it['question']}\n"
        for i, c in enumerate(choices):
            prompt += f"{LETTERS[i]}. {c}\n"
        prompt += "Answer:"
        ans = it["answer"]
        ans = ord(ans.upper()) - ord("A") if isinstance(ans, str) else int(ans)
        items.append({"sample_id": it["sample_id"], "subject": it["subject"], "answer": ans, "prompt": prompt})
    if n and n < len(items):
        keep = sorted(np.random.default_rng(seed).choice(len(items), size=n, replace=False))
        items = [items[i] for i in keep]
    return items


def option_ids(tok) -> list[int]:
    return [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in LETTERS]


@torch.no_grad()
def score(model, tok, items: list[dict], batch_size: int = 16, max_len: int = 2048) -> list[dict]:
    dev = next(model.parameters()).device
    opts = option_ids(tok)
    enc = [tok(it["prompt"], truncation=True, max_length=max_len).input_ids for it in items]
    order = sorted(range(len(items)), key=lambda i: len(enc[i]))
    pad = tok.pad_token_id
    out: list[dict | None] = [None] * len(items)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        T = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), T), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, T - len(enc[i]) :] = torch.tensor(enc[i])
            mask[r, T - len(enc[i]) :] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev)).logits[:, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)[:, opts].cpu()
        for r, i in enumerate(idx):
            it = items[i]
            out[i] = {"sample_id": it["sample_id"], "subject": it["subject"], "answer": it["answer"],
                      "lp": [round(v, 5) for v in lp[r].tolist()], "pred": int(lp[r].argmax())}  # fmt: skip
    return out


def summarize(records: list[dict]) -> dict:
    correct = [r["pred"] == r["answer"] for r in records]
    subj: dict[str, list[bool]] = {}
    for r, c in zip(records, correct):
        subj.setdefault(r["subject"], []).append(c)
    accs = [float(np.mean(v)) for v in subj.values()]
    return {"acc": float(np.mean(correct)), "n": len(records), "correct": int(sum(correct)),
            "macro_acc": float(np.mean(accs)), "n_subjects": len(accs),
            "per_subject": {s: float(np.mean(v)) for s, v in sorted(subj.items())}}  # fmt: skip
