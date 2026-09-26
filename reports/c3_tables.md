# C3 — общий и языковой урон на Global-MMLU (scripts/analyze_c3.py)

## Qwen2.5-3B (n = 14041 parallel items)

| | ru | en |
|---|---|---|
| mean damage Δ log p(correct), INT3 − FP16 | -0.372 | -0.423 |
| mean repair Δ log p(correct), O1(ru slot) − INT3 | 0.073 | 0.080 |

- corr(damage ru, damage en) = **0.442** (permutation null |r|₉₉ = 0.020; item difficulty FP16 ru/en: 0.671)
- corr(repair ru, repair en) = **0.214** — the Russian correction improves the same items in both languages
- items lost in ru: 4892 (and 1496 gained by chance); also lost in en: **2289** vs 2119 expected if independent (×1.08); ru-only: 2603
- O1 (≥2 of 3 seeds) recovers 24.6% of shared, 38.9% of ru-only; on en it recovers 25.5% of the items INT3 lost in en; it breaks 25.0% of ru items INT3 had right

## Llama-3.2-3B (n = 14041 parallel items)

| | ru | en |
|---|---|---|
| mean damage Δ log p(correct), INT3 − FP16 | -0.226 | -0.350 |
| mean repair Δ log p(correct), O1(ru slot) − INT3 | 0.077 | 0.091 |

- corr(damage ru, damage en) = **0.254** (permutation null |r|₉₉ = 0.022; item difficulty FP16 ru/en: 0.573)
- corr(repair ru, repair en) = **0.113** — the Russian correction improves the same items in both languages
- items lost in ru: 3733 (and 1867 gained by chance); also lost in en: **1343** vs 1230 expected if independent (×1.09); ru-only: 2390
- O1 (≥2 of 3 seeds) recovers 29.0% of shared, 35.5% of ru-only; on en it recovers 27.4% of the items INT3 lost in en; it breaks 24.3% of ru items INT3 had right
