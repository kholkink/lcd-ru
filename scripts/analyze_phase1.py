#!/usr/bin/env python
"""Phase 1 (LCD replication): collect the Colab results and compare with the paper (PLAN.md appendix A).

    python scripts/analyze_phase1.py            # sync results from Kaggle, then analyse
    python scripts/analyze_phase1.py --no-sync  # analyse what is already in results/

Writes reports/phase1_tables.md and results/phase1_summary.json (overwritten: derived, not raw results).
"""

import argparse
import gzip
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict

import _common  # noqa: F401
import numpy as np

from src.eval import stats
from src.utils import REPO, append_jsonl

MODELS = {"qwen3b": "Qwen2.5-3B", "llama3b": "Llama-3.2-3B"}
PAPER = {  # appendix A (raw counts where available)
    "qwen3b": {"ppl_ru": (5.90, 12.51, 8.74), "ppl_en": (13.71, 18.56), "mmlu_ru": (53.88, 29.64, 35.77),
               "mmlu_en": (64.68, 41.77)},
    "llama3b": {"ppl_ru": (7.74, 18.46, 10.77), "ppl_en": (12.33, 17.15), "mmlu_ru": (44.37, 31.24, 34.67),
                "mmlu_en": (55.47, 39.92)},
}  # fmt: skip
STATUS = REPO / "build" / "jobs_status" / "a" / "results"
ITEMS = REPO / "results" / "items"
OUT = REPO / "results" / "phase1.jsonl"


def sync() -> None:
    subprocess.run([sys.executable, "colab/jobs.py", "results", "--id", "p1-qwen3b-fp16"], cwd=REPO, check=False,
                   capture_output=True)  # fmt: skip  (downloads the whole status dataset)
    seen = set()
    if OUT.exists():
        seen = {(r["run"], r["kind"], r.get("cond"), r["time_utc"]) for r in map(json.loads, OUT.read_text().splitlines())}
    n = 0
    for jd in sorted(STATUS.glob("p1-*")):
        (ITEMS / jd.name).mkdir(parents=True, exist_ok=True)
        for p in jd.iterdir():
            if "__mmlu_" in p.name:  # per-item MMLU: .jsonl.gz, or a dir holding the jsonl (kaggle --unzip gunzips)
                src = p if p.is_file() else next(p.glob("*.jsonl"), None)
                if src is None:
                    continue
                dst = ITEMS / jd.name / (p.name.split("__", 1)[1].removesuffix(".gz"))
                if src.name.endswith(".gz"):
                    with gzip.open(src, "rt") as fi, open(dst, "w") as fo:
                        shutil.copyfileobj(fi, fo)
                else:
                    shutil.copy2(src, dst)
            elif p.name.endswith("results.jsonl"):
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    k = (r["run"], r["kind"], r.get("cond"), r["time_utc"])
                    if k not in seen:
                        append_jsonl(OUT, {**r, "job": jd.name})
                        seen.add(k)
                        n += 1
    print(f"sync: {n} new records -> {OUT}")


def load() -> tuple[dict, dict]:
    ev, other = {}, defaultdict(list)
    for r in map(json.loads, OUT.read_text().splitlines()):
        if r["kind"] == "eval":
            ev[r["run"]] = r
        else:
            other[r["run"]].append(r)
    return ev, other


def items(job: str, cond: str, lang: str, split: str = "full") -> list[dict] | None:
    p = ITEMS / job / f"mmlu_{cond}_{lang}_{split}.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        return [json.loads(line) for line in f]


def ms(xs: list[float], nd: int = 2) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{np.mean(xs):.{nd}f}" + (f" ± {np.std(xs, ddof=1):.{nd}f}" if len(xs) > 1 else "")


def dev_only(recs: list[dict], dev: set[str]) -> list[dict]:
    return [r for r in recs if r["sample_id"] in dev]


def analyse() -> None:
    ev, other = load()
    dev = set(json.loads((REPO / "data/splits/global_mmlu_dev.json").read_text())["sample_ids"])
    lines, summary = [], {}
    w = lines.append
    for m, mname in MODELS.items():
        fp, i3 = ev.get(f"p1_{m}_fp16"), ev.get(f"p1_{m}_int3lcd")
        o1 = [ev[k] for k in (f"p1_{m}_o1_s{s}" for s in range(3)) if k in ev]
        io = ev.get(f"p1_{m}_int3ours")
        if not (fp and i3):
            w(f"## {mname}: FP16 / INT3 results missing\n")
            continue
        P = PAPER[m]
        g = lambda r, k, f="ppl": r["metrics"].get(k, {}).get(f) if r else None  # noqa: E731
        S = summary.setdefault(m, {})
        w(f"## {mname}\n")
        # --- LCD protocol vs paper
        w("### LCD-протокол против статьи\n")
        w("| Метрика | FP16 (наш / статья) | INT3-LCD (наш / статья) | INT3/FP16 (наш / статья, откл.) | +O1, 3 сида (наш / статья) |")
        w("|---|---|---|---|---|")
        for lang in ("ru", "en"):
            k = f"ppl/ppl_lcd_{lang}_256"
            f_, i_ = g(fp, k), g(i3, k)
            pf, pi = P[f"ppl_{lang}"][:2]
            ratio, pratio = i_ / f_, pi / pf
            o = [g(r, k) for r in o1]
            pl = P[f"ppl_{lang}"][2] if len(P[f"ppl_{lang}"]) > 2 else None
            w(f"| ppl {lang} | {f_:.2f} / {pf:.2f} | {i_:.2f} / {pi:.2f} | {ratio:.3f} / {pratio:.3f} ({(ratio / pratio - 1) * 100:+.1f}%) "
              f"| {ms(o)} / {pl if pl else '—'} |")
            S[f"ppl_{lang}_ratio_dev_pct"] = (ratio / pratio - 1) * 100
        for lang in ("ru", "en"):
            k = f"mmlu/{lang}/full"
            f_, i_ = g(fp, k, "acc") * 100, g(i3, k, "acc") * 100
            pf, pi = P[f"mmlu_{lang}"][:2]
            o = [g(r, k, "acc") * 100 for r in o1]
            pl = P[f"mmlu_{lang}"][2] if len(P[f"mmlu_{lang}"]) > 2 else None
            w(f"| Global-MMLU {lang} | {f_:.2f} / {pf:.2f} | {i_:.2f} / {pi:.2f} | — | {ms(o)} / {pl if pl else '—'} |")
            S[f"mmlu_{lang}_fp16_diff"], S[f"mmlu_{lang}_int3_diff"] = f_ - pf, i_ - pi
        w("")
        # --- recovery
        k = "ppl/ppl_lcd_ru_256"
        f_, i_ = g(fp, k), g(i3, k)
        rp = [stats.recovery(f_, i_, g(r, k)) * 100 for r in o1]
        rc = [stats.recovery_ce(f_, i_, g(r, k)) * 100 for r in o1]
        pf, pi, pl = P["ppl_ru"]
        ra = [stats.recovery(g(fp, "mmlu/ru/full", "acc"), g(i3, "mmlu/ru/full", "acc"), g(r, "mmlu/ru/full", "acc")) * 100
              for r in o1]  # fmt: skip
        pa = P["mmlu_ru"]
        w("### Восстановление (O1, ru)\n")
        w("| | наш (3 сида) | статья |")
        w("|---|---|---|")
        w(f"| R_ppl, LCD-протокол | {ms(rp, 1)} % | {(pi - pl) / (pi - pf) * 100:.1f} % |")
        w(f"| R_CE, LCD-протокол | {ms(rc, 1)} % | {stats.recovery_ce(pf, pi, pl) * 100:.1f} % |")
        w(f"| R_acc, Global-MMLU ru (полный) | {ms(ra, 1)} % | {(pa[2] - pa[1]) / (pa[0] - pa[1]) * 100:.1f} % |")
        S.update(R_ppl=rp, R_CE=rc, R_acc=ra, R_ppl_paper=(pi - pl) / (pi - pf) * 100)
        # clean protocol + OOD (H2 first look)
        w("\n### Чистый протокол и OOD (R_CE, O1, ru-слот)\n")
        w("| Набор | FP16 | INT3-LCD | O1 (3 сида) | R_CE O1 |")
        w("|---|---|---|---|---|")
        for split in ("ppl_ru_256", "ppl_wiki_ru_256", "ppl_gazeta_256", "ppl_en_256", "ppl_lcd_en_256"):
            k = f"ppl/{split}"
            f_, i_ = g(fp, k), g(i3, k)
            o = [g(r, k) for r in o1]
            w(f"| {split} | {f_:.2f} | {i_:.2f} | {ms(o)} | {ms([stats.recovery_ce(f_, i_, x) * 100 for x in o], 1)} % |")
        # KL
        w("\n### Согласие с учителем\n")
        w("| | INT3-LCD | O1 (3 сида) | R_KL O1 |")
        w("|---|---|---|---|")
        for split in ("val_ru_64", "ppl_ru_256"):
            k = f"kl/{split}"
            ki = g(i3, k, "kl")
            ko = [g(r, k, "kl") for r in o1]
            w(f"| KL {split} | {ki:.4f} | {ms(ko, 4)} | {ms([(ki - x) / ki * 100 for x in ko], 1)} % |")
            w(f"| top-1 {split} | {g(i3, k, 'top1'):.4f} | {ms([g(r, k, 'top1') for r in o1], 4)} | |")
            w(f"| e_mean {split} | {g(i3, k, 'e_mean'):.4f} | {ms([g(r, k, 'e_mean') for r in o1], 4)} | |")
        rb = [g(r, "rublimp", "acc") for r in o1]
        w(f"\nRuBLiMP (полный): FP16 {g(fp, 'rublimp', 'acc'):.4f}, INT3-LCD {g(i3, 'rublimp', 'acc'):.4f}, O1 {ms(rb, 4)}\n")
        # significance on the full set (LCD's test + paired bootstrap), per seed
        t_fp = items(f"p1-{m}-fp16", "fp16", "ru")
        t_i3 = items(f"p1-{m}-int3lcd", "int3lcd", "ru")
        w("### Значимость O1 против INT3 (Global-MMLU ru, полный)\n")
        w("| Сид | Δacc, п. | 95% CI (bootstrap) | t-test по 57 предметам |")
        w("|---|---|---|---|")
        seeds_items = []
        for s in range(3):
            t_o = items(f"p1-{m}-o1-s{s}", f"o1_s{s}", "ru")
            if not (t_o and t_i3):
                continue
            seeds_items.append(t_o)
            a, b = stats.align(t_o, t_i3)
            x = np.array([r["pred"] == r["answer"] for r in a], float)
            y = np.array([r["pred"] == r["answer"] for r in b], float)
            bt = stats.paired_bootstrap(x, y, n_boot=5000)
            tt = stats.subject_ttest(t_o, t_i3)
            w(f"| {s} | {bt['diff'] * 100:+.2f} | [{bt['ci95'][0] * 100:+.2f}, {bt['ci95'][1] * 100:+.2f}] | t={tt['t']:.2f}, p={tt['p']:.2g} |")
        # --- power on dev (for the phase 3 rule)
        if t_fp and t_i3 and seeds_items:
            d_fp, d_i3 = dev_only(t_fp, dev), dev_only(t_i3, dev)
            q_i3o1 = np.mean([np.mean([u["pred"] != v["pred"] for u, v in zip(*stats.align(dev_only(t, dev), d_i3))])
                              for t in seeds_items])  # fmt: skip
            q_seeds = [np.mean([u["pred"] != v["pred"] for u, v in zip(*stats.align(dev_only(a, dev), dev_only(b, dev)))])
                       for i, a in enumerate(seeds_items) for b in seeds_items[i + 1 :]]  # fmt: skip
            gap = (g(fp, "mmlu/ru/full", "acc") - g(i3, "mmlu/ru/full", "acc")) * 100
            n = len(d_i3)
            se = 100 * math.sqrt(q_i3o1 / n)
            rkl = []
            for t in seeds_items:
                a_i3 = stats.mmlu_vs_teacher(d_fp, d_i3)
                a_o = stats.mmlu_vs_teacher(d_fp, dev_only(t, dev))
                rkl.append((a_i3["kl_abcd"] - a_o["kl_abcd"]) / a_i3["kl_abcd"] * 100)
            # per-item bootstrap SD of R_KL for one seed
            a_i3 = stats.mmlu_vs_teacher(d_fp, d_i3)["per_item"]["kl"]
            a_o = stats.mmlu_vs_teacher(d_fp, dev_only(seeds_items[0], dev))["per_item"]["kl"]
            rng = np.random.default_rng(0)
            boot = [(a_i3[i].mean() - a_o[i].mean()) / a_i3[i].mean() * 100
                    for i in (rng.integers(0, len(a_i3), len(a_i3)) for _ in range(2000))]  # fmt: skip
            w("\n### Мощность на dev (2000) — для порога фазы 3\n")
            w(f"- Разрыв FP16−INT3 (ru, полный): {gap:.2f} п.; 5 п.п. R_acc = {0.05 * gap:.2f} п.")
            w(f"- Доля расхождений ответов INT3 vs O1 на dev: {q_i3o1:.3f}; между сидами O1: {ms(q_seeds, 3)}")
            w(f"- SE парной разницы точности на dev: {se:.2f} п.; MDE (80%): {2.8 * se:.2f} п. = {2.8 * se / gap * 100:.1f} п.п. R_acc")
            w(f"- R_KL на dev (по буквам A–D), O1: {ms(rkl, 1)} % (разброс сидов); bootstrap-SD по вопросам (сид 0): {np.std(boot):.1f} п.п.")
            S.update(dev_q_int3_o1=q_i3o1, dev_q_seeds=q_seeds, dev_se_acc=se, gap=gap, R_KL_dev=rkl,
                     R_KL_dev_boot_sd=float(np.std(boot)))  # fmt: skip
        # --- checks and INT3-ours
        chk = [r["metrics"].get("off_equals_base") for k in (f"p1_{m}_o1_s{s}" for s in range(3)) for r in other.get(k, [])
               if r["kind"] == "check"]  # fmt: skip
        w(f"\nКонтроль (а) (поправка выключена = INT3 побитово): {chk}\n")
        cmp_ = [r for r in other.get(f"p1_{m}_int3ours", []) if r["kind"] == "compare_int3"]
        if io:
            w("### INT3-ours против INT3-LCD\n")
            w("| | INT3-LCD | INT3-ours | откл. |")
            w("|---|---|---|---|")
            for split in ("ppl_lcd_ru_256", "ppl_lcd_en_256", "ppl_ru_256", "ppl_en_256"):
                k = f"ppl/{split}"
                r_l, r_o = g(i3, k) / g(fp, k), g(io, k) / g(fp, k)
                w(f"| INT3/FP16 {split} | {r_l:.3f} | {r_o:.3f} | {(r_o / r_l - 1) * 100:+.1f}% |")
            for lang in ("ru", "en"):
                k = f"mmlu/{lang}/dev"
                acc_l = np.mean([r["pred"] == r["answer"] for r in dev_only(items(f"p1-{m}-int3lcd", "int3lcd", lang) or [], dev)])
                w(f"| Global-MMLU {lang} dev | {acc_l * 100:.2f} | {g(io, k, 'acc') * 100:.2f} | |")
            if cmp_:
                c = cmp_[-1]["metrics"]
                w(f"\nСовпадение кодов: {c['all']['codes_equal'] * 100:.1f}%; скейлов: {c['all']['scales_equal'] * 100:.2f}%. По типам: "
                  + ", ".join(f"{k} {v['codes_equal'] * 100:.1f}%" for k, v in c.items() if k != "all") + "\n")
        w("")
    (REPO / "reports" / "phase1_tables.md").write_text("# Фаза 1 — таблицы (генерируются scripts/analyze_phase1.py)\n\n" + "\n".join(lines))
    (REPO / "results" / "phase1_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    analyse()
