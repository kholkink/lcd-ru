#!/usr/bin/env python
"""C3: is the Global-MMLU damage of INT3 (and the repair by a Russian correction) shared across ru / en?

Works on the per-item A-D log-probs of phase 1 (results/items/, parallel items share sample_id). Per item:
  lp_c = log p(correct letter) renormalized over A-D;  damage d = lp_c(INT3) - lp_c(FP16);  repair r = lp_c(O1) - lp_c(INT3)
(O1 = mean over the 3 seeds; on en the Russian slot is active = control (b)).

  1. corr(d_ru, d_en) over items, against a permutation null        -> is the damage item-specific in both languages?
  2. corr(r_ru, r_en)                                                -> does the ru correction fix the same items in en?
  3. discrete: items lost in ru (FP16 right, INT3 wrong); share also lost in en vs the share expected if losses were
     independent across languages; O1 recovery (majority of 3 seeds) in the shared / ru-only groups.
Writes reports/c3_tables.md and results/c3_summary.json (derived).
"""

import json

import _common  # noqa: F401
import numpy as np

from src.utils import REPO

ITEMS = REPO / "results" / "items"
MODELS = {"qwen3b": "Qwen2.5-3B", "llama3b": "Llama-3.2-3B"}


def load(job, cond, lang):
    with open(ITEMS / job / f"mmlu_{cond}_{lang}_full.jsonl") as f:
        return {r["sample_id"]: r for r in map(json.loads, f)}


def lpc(recs, ids):
    lp = np.array([recs[i]["lp"] for i in ids])
    lp = lp - lp.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    return lp[np.arange(len(ids)), [recs[i]["answer"] for i in ids]]


def correct(recs, ids):
    return np.array([recs[i]["pred"] == recs[i]["answer"] for i in ids])


def perm_null(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])


def main():
    out, lines = {}, ["# C3 — общий и языковой урон на Global-MMLU (scripts/analyze_c3.py)\n"]
    w = lines.append
    for m, name in MODELS.items():
        F = {l: load(f"p1-{m}-fp16", "fp16", l) for l in ("ru", "en")}
        I = {l: load(f"p1-{m}-int3lcd", "int3lcd", l) for l in ("ru", "en")}
        O = [{l: load(f"p1-{m}-o1-s{s}", f"o1_s{s}", l) for l in ("ru", "en")} for s in range(3)]
        ids = sorted(set(F["ru"]) & set(F["en"]))
        d = {l: lpc(I[l], ids) - lpc(F[l], ids) for l in ("ru", "en")}
        r = {l: np.mean([lpc(o[l], ids) for o in O], 0) - lpc(I[l], ids) for l in ("ru", "en")}
        c_d, c_r = np.corrcoef(d["ru"], d["en"])[0, 1], np.corrcoef(r["ru"], r["en"])[0, 1]
        null = perm_null(d["ru"], d["en"])
        base = np.corrcoef(lpc(F["ru"], ids), lpc(F["en"], ids))[0, 1]
        # discrete
        f_ok = {l: correct(F[l], ids) for l in ("ru", "en")}
        i_ok = {l: correct(I[l], ids) for l in ("ru", "en")}
        o_ok = {l: np.mean([correct(o[l], ids) for o in O], 0) >= 2 / 3 for l in ("ru", "en")}
        lost = {l: f_ok[l] & ~i_ok[l] for l in ("ru", "en")}
        shared = lost["ru"] & lost["en"]
        ru_only = lost["ru"] & ~lost["en"]
        p_en_lost_given_fok = lost["en"].sum() / f_ok["en"].sum()
        exp_shared = (lost["ru"] & f_ok["en"]).sum() * p_en_lost_given_fok
        gained = ~f_ok["ru"] & i_ok["ru"]
        res = {
            "n": len(ids), "corr_damage": c_d, "corr_damage_null_p99": float(np.percentile(np.abs(null), 99)),
            "corr_fp16_difficulty": base, "corr_repair": c_r,
            "mean_damage_ru": float(d["ru"].mean()), "mean_damage_en": float(d["en"].mean()),
            "mean_repair_ru": float(r["ru"].mean()), "mean_repair_en": float(r["en"].mean()),
            "lost_ru": int(lost["ru"].sum()), "gained_ru_by_chance": int(gained.sum()),
            "shared": int(shared.sum()), "shared_expected_if_independent": float(exp_shared),
            "ru_only": int(ru_only.sum()),
            "o1_recovers_shared": float(o_ok["ru"][shared].mean()), "o1_recovers_ru_only": float(o_ok["ru"][ru_only].mean()),
            "o1_en_recovers_lost_en": float(o_ok["en"][lost["en"]].mean()),
            "o1_breaks_ru_ok": float((~o_ok["ru"][f_ok["ru"] & i_ok["ru"]]).mean()),
        }  # fmt: skip
        out[m] = res
        w(f"## {name} (n = {len(ids)} parallel items)\n")
        w("| | ru | en |")
        w("|---|---|---|")
        w(f"| mean damage Δ log p(correct), INT3 − FP16 | {res['mean_damage_ru']:.3f} | {res['mean_damage_en']:.3f} |")
        w(f"| mean repair Δ log p(correct), O1(ru slot) − INT3 | {res['mean_repair_ru']:.3f} | {res['mean_repair_en']:.3f} |")
        w("")
        w(f"- corr(damage ru, damage en) = **{c_d:.3f}** (permutation null |r|₉₉ = {res['corr_damage_null_p99']:.3f}; "
          f"item difficulty FP16 ru/en: {base:.3f})")
        w(f"- corr(repair ru, repair en) = **{c_r:.3f}** — the Russian correction improves the same items in both languages")
        w(f"- items lost in ru: {res['lost_ru']} (and {res['gained_ru_by_chance']} gained by chance); also lost in en: "
          f"**{res['shared']}** vs {exp_shared:.0f} expected if independent (×{res['shared'] / exp_shared:.2f}); ru-only: {res['ru_only']}")
        w(f"- O1 (≥2 of 3 seeds) recovers {res['o1_recovers_shared'] * 100:.1f}% of shared, {res['o1_recovers_ru_only'] * 100:.1f}% of ru-only;"
          f" on en it recovers {res['o1_en_recovers_lost_en'] * 100:.1f}% of the items INT3 lost in en; it breaks "
          f"{res['o1_breaks_ru_ok'] * 100:.1f}% of ru items INT3 had right\n")
    (REPO / "reports" / "c3_tables.md").write_text("\n".join(lines))
    (REPO / "results" / "c3_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
