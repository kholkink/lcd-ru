import math

import numpy as np
import torch

from src.eval import mmlu, ppl, rublimp, stats

DOCS = ["Первый документ, короткий.", "Второй документ " * 30, "Third one, in English, of medium length here."]


def test_ppl_batched_matches_lcd_loop(tiny, tok):
    m = tiny()
    # LCD evaluate.py `_perplexity`: one doc at a time, loss * (n - 1), token-weighted
    nll = tokn = 0
    with torch.no_grad():
        for d in DOCS:
            ids = tok(d, return_tensors="pt", truncation=True, max_length=40).input_ids
            nll += m(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
            tokn += ids.shape[1] - 1
    r = ppl.perplexity(m, tok, DOCS, max_len=40, batch_size=3)
    assert r["tokens"] == tokn
    assert abs(r["ppl"] - math.exp(nll / tokn)) / r["ppl"] < 1e-4


def _items():
    base = [("Сколько будет 2+2?", ["3", "4", "5", "22"], 1), ("Столица Франции?", ["Париж", "Рим", "Берлин", "Мадрид"], 0),
            ("Long question " * 20 + "?", ["a", "b", "c", "d"], 3)]  # fmt: skip
    items = []
    for k, (q, ch, a) in enumerate(base):
        p = f"Question: {q}\n" + "".join(f"{mmlu.LETTERS[i]}. {c}\n" for i, c in enumerate(ch)) + "Answer:"
        items.append({"sample_id": f"s{k}", "subject": "x", "answer": a, "prompt": p})
    return items


def test_mmlu_batched_matches_lcd_single_item_scoring(tiny, tok):
    m = tiny()
    items = _items()
    recs = mmlu.score(m, tok, items, batch_size=3)
    opts = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in "ABCD"]  # LCD evaluate.py
    with torch.no_grad():
        for it, r in zip(items, recs):
            ids = tok(it["prompt"], return_tensors="pt", truncation=True, max_length=2048).input_ids
            lp = torch.log_softmax(m(ids).logits[0, -1, :], dim=-1)
            ref = [lp[t].item() for t in opts]
            assert np.allclose(r["lp"], ref, atol=1e-4)
            assert r["pred"] == ref.index(max(ref))
    s = mmlu.summarize(recs)
    assert s["n"] == 3 and 0 <= s["acc"] <= 1


def test_rublimp_batched_matches_single(tiny, tok):
    m = tiny()
    sents = ["Для человека не такие дома строят.", "Для человеком не такие дома строят.", "Кот."]
    batched = rublimp.sentence_logprobs(m, tok, sents, batch_size=3)
    single = [rublimp.sentence_logprobs(m, tok, [s], batch_size=1)[0] for s in sents]
    assert np.allclose(batched, single, atol=1e-3)
    assert rublimp.prefix_id(tok) == tok.eos_token_id  # Qwen has no BOS


def test_stats():
    assert stats.recovery(50, 30, 35) == 0.25
    assert abs(stats.recovery_ce(5.90, 12.51, 8.74) - 0.477) < 1e-3  # LCD Qwen ru
    t = [{"sample_id": i, "subject": "s", "answer": 0, "lp": [-0.1, -3, -3, -3], "pred": 0} for i in range(4)]
    s = [{"sample_id": i, "subject": "s", "answer": 0, "lp": [-3, -0.1, -3, -3], "pred": 1} for i in range(4)]
    r = stats.mmlu_vs_teacher(t, t)
    assert abs(r["kl_abcd"]) < 1e-9 and r["agree"] == 1
    r = stats.mmlu_vs_teacher(t, s)
    assert r["kl_abcd"] > 1 and r["agree"] == 0
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 500)
    b = stats.paired_bootstrap(x, np.zeros(500), n_boot=2000)
    assert b["ci95"][0] > 0 and b["p"] < 0.01
