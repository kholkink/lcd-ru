#!/usr/bin/env python
"""Phase 2: sync the Colab results (diagnostics + llm-compressor check), tables and figures.

    python scripts/analyze_phase2.py [--no-sync]

Writes results/phase2.jsonl (append-only raw records), reports/phase2_tables.md and reports/figs/p2_*.png (derived).
Chart colors: the dataviz reference palette (categorical slots 1-4 validated; sequential blue 100-700 for heatmaps).
"""

import argparse
import json
import subprocess
import sys

import _common  # noqa: F401
import numpy as np

from src.utils import REPO, append_jsonl

STATUS = REPO / "build" / "jobs_status" / "a" / "results"
OUT = REPO / "results" / "phase2.jsonl"
FIGS = REPO / "reports" / "figs"
MODELS = {"qwen3b": "Qwen2.5-3B", "llama3b": "Llama-3.2-3B"}
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]  # fmt: skip


def sync() -> None:
    subprocess.run([sys.executable, "colab/jobs.py", "results", "--id", "p2-qwen3b-diag"], cwd=REPO, capture_output=True)
    seen = set()
    if OUT.exists():
        seen = {(r["run"], r["kind"], r.get("cond"), r.get("split"), r["time_utc"]) for r in map(json.loads, OUT.read_text().splitlines())}
    n = 0
    for jd in sorted(STATUS.glob("p2-*")):
        for p in jd.glob("*results.jsonl"):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                k = (r["run"], r["kind"], r.get("cond"), r.get("split"), r["time_utc"])
                if k not in seen:
                    append_jsonl(OUT, {**r, "job": jd.name})
                    seen.add(k)
                    n += 1
    print(f"sync: {n} new records -> {OUT}")


def load() -> list[dict]:
    return [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []


def diag(recs, m):
    """{(cond, split): metrics} from the latest record of each."""
    out = {}
    for r in recs:
        if r["kind"] == "diag" and r["run"] == f"p2_{m}_diag":
            out[(r["cond"], r["split"])] = r["metrics"]
    return out


def mean_o1(D, split, key):
    vals = [np.array(D[(f"o1_s{s}", split)][key]) for s in range(3) if (f"o1_s{s}", split) in D]
    return np.mean(vals, 0) if vals else None


def thirds(x):
    k = len(x) // 3
    return [float(np.mean(x[:k])), float(np.mean(x[k : 2 * k])), float(np.mean(x[2 * k :]))]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_el(D, m, name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor=SURF)
    for ax, split, title in zip(axes, ("val_ru_64", "ppl_en_256"), ("русский текст", "английский текст")):
        style(ax)
        series = [("INT3: накопленная", D[("int3lcd", split)]["e_acc"]), ("INT3: локальная", D[("int3lcd", split)]["e_locT"]),
                  ("+O1: накопленная", mean_o1(D, split, "e_acc")), ("+O1: локальная", mean_o1(D, split, "e_locT"))]  # fmt: skip
        ends = []
        for (lab, y), c in zip(series, SERIES):
            if y is None:
                continue
            x = np.arange(len(y))
            ax.plot(x, y, color=c, linewidth=2, label=lab)
            ends.append([float(y[-1]), lab, len(y) - 1])
        # end labels, nudged apart so near-identical lines do not print on top of each other
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for yl, lab, xl in ends:
            ax.annotate(lab, (xl, yl), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("блок")
    axes[0].set_ylabel("нормированная ошибка e_l")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, labelcolor=INK2, loc="lower center", ncol=4)
    fig.suptitle(f"{name}: ошибка residual stream по глубине (среднее по 64 текстам; O1 — среднее 3 сидов)", fontsize=9,
                 color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = FIGS / f"p2_{m}_el.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p


def lin_matrix(lin: dict, L: int) -> np.ndarray:
    M = np.full((L, len(KINDS)), np.nan)
    for name, v in lin.items():
        blk = int(name.split(".layers.")[1].split(".")[0])
        kind = name.rsplit(".", 1)[1]
        if kind in KINDS:
            M[blk, KINDS.index(kind)] = v
    return M


def fig_heat(D, m, name, L):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    panels = [(("int3lcd", "val_ru_64"), "lin_local", "локальная, ru"), (("int3lcd", "val_ru_64"), "lin_e2e", "сквозная, ru"),
              (("int3lcd", "ppl_en_256"), "lin_local", "локальная, en"), (("int3lcd", "ppl_en_256"), "lin_e2e", "сквозная, en")]  # fmt: skip
    panels = [panels[0], panels[2], panels[1], panels[3]]  # local ru, local en | e2e ru, e2e en
    mats = [lin_matrix(D[k][key], L) for k, key, _ in panels]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 0.18 * L + 1.6), sharey=True, facecolor=SURF)
    ims = []
    for i, (ax, M, (_, _, title)) in enumerate(zip(axes, mats, panels)):
        pair = mats[:2] if i < 2 else mats[2:]  # each measure on its own scale
        vmax = np.nanpercentile(np.concatenate([x.ravel() for x in pair]), 99)
        ims.append(ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"))
        ax.set_xticks(range(len(KINDS)), [k.replace("_proj", "") for k in KINDS], rotation=45, fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("блок", color=INK2)
    fig.subplots_adjust(wspace=0.12, right=0.93)
    pos = [ax.get_position() for ax in axes]
    shift = 0.07  # room for the first colorbar between the two groups
    for ax, p_ in zip(axes[2:], pos[2:]):
        ax.set_position([p_.x0 + shift, p_.y0, p_.width, p_.height])
    for im, ax_right, lab in ((ims[1], axes[1], "локальная"), (ims[3], axes[3], "сквозная")):
        b = ax_right.get_position()
        cax = fig.add_axes([b.x1 + 0.008, b.y0 + b.height * 0.25, 0.01, b.height * 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{lab}, отн. ошибка", color=INK2, fontsize=8)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
    fig.suptitle(f"{name}, INT3-LCD: относительная ошибка выхода Linear (локальная = на входе учителя)", fontsize=9,
                 color=INK, x=0.01, ha="left")  # fmt: skip
    p = FIGS / f"p2_{m}_heatmap.png"
    fig.savefig(p, dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return p


def analyse() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIGS.mkdir(parents=True, exist_ok=True)
    recs = load()
    lines = ["# Фаза 2 — таблицы (scripts/analyze_phase2.py)\n"]
    w = lines.append
    for m, name in MODELS.items():
        D = diag(recs, m)
        if ("int3lcd", "val_ru_64") not in D:
            w(f"## {name}: диагностики ещё нет\n")
            continue
        L = len(D[("int3lcd", "val_ru_64")]["e_acc"])
        w(f"## {name} ({L} блоков)\n")
        w("Средние по третям глубины (ранние / средние / поздние блоки). Доля распространённой = 1 − локальная/накопленная.\n")
        w("| Текст | Условие | e_acc | e_locT (вход учителя) | e_locS (вход студента) | доля распространённой |")
        w("|---|---|---|---|---|---|")
        for split in ("val_ru_64", "ppl_en_256"):
            for cond in ("int3lcd", "o1"):
                if cond == "o1":
                    acc, lt, ls = (mean_o1(D, split, k) for k in ("e_acc", "e_locT", "e_locS"))
                    if acc is None:
                        continue
                else:
                    acc, lt, ls = (np.array(D[(cond, split)][k]) for k in ("e_acc", "e_locT", "e_locS"))
                prop = 1 - lt / np.maximum(acc, 1e-12)
                f = lambda x: " / ".join(f"{v:.3f}" for v in thirds(x))  # noqa: E731
                w(f"| {split} | {'INT3' if cond == 'int3lcd' else '+O1 (3 сида)'} | {f(acc)} | {f(lt)} | {f(ls)} | "
                  f"{' / '.join(f'{v:.2f}' for v in thirds(prop))} |")  # fmt: skip
        o1_ru = mean_o1(D, "val_ru_64", "e_acc")
        if o1_ru is not None:
            i3 = np.array(D[("int3lcd", "val_ru_64")]["e_acc"])
            red = 1 - o1_ru / i3
            w(f"\nСнижение e_acc поправкой O1 (ru), по третям: {' / '.join(f'{v * 100:.0f}%' for v in thirds(red))}; "
              f"в последнем блоке {red[-1] * 100:.0f}%.")
        for split in ("val_ru_64", "ppl_en_256"):
            lin = D[("int3lcd", split)].get("lin_local", {})
            top = sorted(lin.items(), key=lambda kv: -kv[1])[:6]
            w(f"\nХудшие Linear по локальной ошибке ({split}): " + ", ".join(
                f"{n.split('.layers.')[1].replace('.self_attn', '').replace('.mlp', '')} {v:.2f}" for n, v in top))  # fmt: skip
            lin = D[("int3lcd", split)].get("lin_e2e", {})
            top = sorted(lin.items(), key=lambda kv: -kv[1])[:6]
            w(f"Худшие по сквозной ({split}): " + ", ".join(
                f"{n.split('.layers.')[1].replace('.self_attn', '').replace('.mlp', '')} {v:.2f}" for n, v in top))  # fmt: skip
        w(f"\n![e_l]({fig_el(D, m, name).relative_to(REPO / 'reports')})")
        w(f"![heatmap]({fig_heat(D, m, name, L).relative_to(REPO / 'reports')})\n")
    # llm-compressor check
    w("## llm-compressor против чекпоинта автора (Qwen2.5-3B)\n")
    ev = {r["run"]: r for r in recs if r["kind"] == "eval"}
    cmp_ = {r["run"]: r["metrics"] for r in recs if r["kind"] == "compare_int3"}
    ph1 = {r["run"]: r for r in map(json.loads, (REPO / "results/phase1.jsonl").read_text().splitlines()) if r["kind"] == "eval"}
    fp, i3 = ph1.get("p1_qwen3b_fp16"), ph1.get("p1_qwen3b_int3lcd")
    w("| Вариант | коды = автор | скейлы = автор | INT3/FP16 ppl ru (LCD) | INT3/FP16 ppl en (LCD) | MMLU ru dev | MMLU en dev |")
    w("|---|---|---|---|---|---|---|")
    g = lambda r, k, f="ppl": r["metrics"][k][f]  # noqa: E731
    for run, lab in (("p2_qwen3b_llmc_qac", "llm-compressor 0.14, qac=True"), ("p2_qwen3b_llmc_noqac", "llm-compressor 0.14, qac=False")):
        if run not in ev:
            w(f"| {lab} | — | — | — | — | — | — |")
            continue
        c = cmp_.get(run, {}).get("all", {})
        e = ev[run]
        w(f"| {lab} | {c.get('codes_equal', float('nan')) * 100:.1f}% | {c.get('scales_equal', float('nan')) * 100:.1f}% | "
          f"{g(e, 'ppl/ppl_lcd_ru_256') / g(fp, 'ppl/ppl_lcd_ru_256'):.3f} | {g(e, 'ppl/ppl_lcd_en_256') / g(fp, 'ppl/ppl_lcd_en_256'):.3f} | "
          f"{g(e, 'mmlu/ru/dev', 'acc') * 100:.2f} | {g(e, 'mmlu/en/dev', 'acc') * 100:.2f} |")  # fmt: skip
    if fp and i3:
        w(f"| INT3-LCD (автор) | 100% | 100% | {g(i3, 'ppl/ppl_lcd_ru_256') / g(fp, 'ppl/ppl_lcd_ru_256'):.3f} | "
          f"{g(i3, 'ppl/ppl_lcd_en_256') / g(fp, 'ppl/ppl_lcd_en_256'):.3f} | 30.20 | 41.30 |")  # fmt: skip
    (REPO / "reports" / "phase2_tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    analyse()
