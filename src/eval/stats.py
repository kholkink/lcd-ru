"""Recovery ratios, MMLU teacher metrics, paired bootstrap, per-subject paired t-test (LCD's test)."""

from __future__ import annotations

import math

import numpy as np


def recovery(fp: float, int3: float, method: float) -> float:
    """(method - INT3) / (FP - INT3); works for accuracy (higher=better) and CE/ppl (lower=better) alike."""
    return (method - int3) / (fp - int3)


def recovery_ce(ppl_fp: float, ppl_int3: float, ppl_method: float) -> float:
    return recovery(math.log(ppl_fp), math.log(ppl_int3), math.log(ppl_method))


def _norm(lp: np.ndarray) -> np.ndarray:
    """Log-probs over the four letters, renormalized to a distribution over A-D."""
    lp = lp - lp.max(axis=-1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))


def align(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    ib = {r["sample_id"]: r for r in b}
    common = [r for r in a if r["sample_id"] in ib]
    return common, [ib[r["sample_id"]] for r in common]


def mmlu_vs_teacher(teacher: list[dict], student: list[dict]) -> dict:
    """Per-item KL(p^T_ABCD || p^S_ABCD), answer agreement with the teacher, log p(correct) of the student."""
    t, s = align(teacher, student)
    lt = _norm(np.array([r["lp"] for r in t]))
    ls = _norm(np.array([r["lp"] for r in s]))
    kl = (np.exp(lt) * (lt - ls)).sum(-1)
    agree = np.array([x["pred"] == y["pred"] for x, y in zip(t, s)], dtype=float)
    ans = np.array([r["answer"] for r in s])
    lpc = ls[np.arange(len(s)), ans]
    return {"kl_abcd": float(kl.mean()), "agree": float(agree.mean()), "logp_correct": float(lpc.mean()),
            "per_item": {"kl": kl, "agree": agree, "logp_correct": lpc}}  # fmt: skip


def paired_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, seed: int = 0, stat=np.mean) -> dict:
    """Difference stat(x) - stat(y) over paired items with a percentile CI and a two-sided bootstrap p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    d = np.array([stat(x[i]) - stat(y[i]) for i in idx])
    obs = stat(x) - stat(y)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(p, 1.0)), "n": len(x)}  # fmt: skip


def subject_ttest(a: list[dict], b: list[dict]) -> dict:
    """LCD's significance test: paired t-test over per-subject accuracy (scipy.stats.ttest_rel)."""
    from scipy.stats import ttest_rel

    a, b = align(a, b)
    subj: dict[str, list[tuple[bool, bool]]] = {}
    for x, y in zip(a, b):
        subj.setdefault(x["subject"], []).append((x["pred"] == x["answer"], y["pred"] == y["answer"]))
    acc_a = [np.mean([u for u, _ in v]) for v in subj.values()]
    acc_b = [np.mean([w for _, w in v]) for v in subj.values()]
    r = ttest_rel(acc_a, acc_b)
    return {"t": float(r.statistic), "p": float(r.pvalue), "df": len(acc_a) - 1}
