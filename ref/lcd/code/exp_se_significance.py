"""
exp_se_significance.py
======================
Computes binomial standard errors and paired t-tests (INT3 vs LCD) for
GlobalMMLU Table 3.  Results are written to
  lcd_results/se_significance_results.json

Usage:
    python exp_se_significance.py
"""

import json
import math
import os
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "lcd_results")

FILES = {
    "qwen2.5-3b": {
        "fp16": os.path.join(RESULTS_DIR, "qwen2.5-3b", "eval_fp16.json"),
        "int3": os.path.join(RESULTS_DIR, "qwen2.5-3b", "eval_int3.json"),
        "lcd":  os.path.join(RESULTS_DIR, "qwen2.5-3b", "eval_lcd.json"),
    },
    "llama3.2-3b": {
        "fp16": os.path.join(RESULTS_DIR, "llama3.2-3b", "eval_fp16.json"),
        "int3": os.path.join(RESULTS_DIR, "llama3.2-3b", "eval_int3.json"),
        "lcd":  os.path.join(RESULTS_DIR, "llama3.2-3b", "eval_lcd.json"),
    },
}

LANGUAGES = ["english", "arabic", "japanese", "chinese", "hindi",
             "russian", "french", "spanish", "korean"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def binomial_se(p, n):
    """SE = sqrt(p*(1-p)/n)"""
    if n == 0:
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_per_subject_vector(data, lang):
    """Return list of per-subject accuracies for a language (57 values)."""
    ps = data["global_mmlu"][lang]["per_subject"]
    return [v["accuracy"] for v in ps.values()]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
results = {}

for model, paths in FILES.items():
    fp16_data = load_json(paths["fp16"])
    int3_data = load_json(paths["int3"])
    lcd_data  = load_json(paths["lcd"])

    results[model] = {}

    for lang in LANGUAGES:
        # ---- overall accuracy and total n ----
        fp16_entry = fp16_data["global_mmlu"][lang]
        int3_entry = int3_data["global_mmlu"][lang]
        lcd_entry  = lcd_data["global_mmlu"][lang]

        fp16_acc = fp16_entry["accuracy"]
        fp16_n   = fp16_entry["total"]
        int3_acc = int3_entry["accuracy"]
        int3_n   = int3_entry["total"]
        lcd_acc  = lcd_entry["accuracy"]
        lcd_n    = lcd_entry["total"]

        fp16_se = binomial_se(fp16_acc, fp16_n)
        int3_se = binomial_se(int3_acc, int3_n)
        lcd_se  = binomial_se(lcd_acc,  lcd_n)

        # ---- paired t-test across 57 subjects (INT3 vs LCD) ----
        # English: LCD == INT3 by design, so t/p = null
        if lang == "english":
            t_stat = None
            p_val  = None
        else:
            int3_vec = get_per_subject_vector(int3_data, lang)
            lcd_vec  = get_per_subject_vector(lcd_data,  lang)
            t_stat, p_val = stats.ttest_rel(int3_vec, lcd_vec)
            t_stat = float(t_stat)
            p_val  = float(p_val)

        results[model][lang] = {
            "fp16_acc": round(fp16_acc, 6),
            "fp16_se":  round(fp16_se,  6),
            "int3_acc": round(int3_acc, 6),
            "int3_se":  round(int3_se,  6),
            "lcd_acc":  round(lcd_acc,  6),
            "lcd_se":   round(lcd_se,   6),
            "int3_vs_lcd_t": round(t_stat, 4) if t_stat is not None else None,
            "int3_vs_lcd_p": round(p_val,  6) if p_val  is not None else None,
        }

        print(f"[{model}] {lang:10s}  "
              f"FP16={fp16_acc:.4f}±{fp16_se:.4f}  "
              f"INT3={int3_acc:.4f}±{int3_se:.4f}  "
              f"LCD={lcd_acc:.4f}±{lcd_se:.4f}  "
              f"t={t_stat}  p={p_val}")

# ---------------------------------------------------------------------------
# Write results JSON
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "se_significance_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to: {out_path}")
