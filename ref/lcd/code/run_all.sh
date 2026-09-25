#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# LCD Experiment Suite — Run All Experiments
# ═══════════════════════════════════════════════════════════════════════════════
#
# Designed for a fresh VM: downloads models from HF Hub first, then runs
# everything using local paths.
#
# Usage:
#   chmod +x run_all.sh
#   HF_USER=your-username ./run_all.sh
#
#   # Override defaults:
#   HF_USER=myname DEVICE=cuda:1 ./run_all.sh
#
#   # Skip specific experiments:
#   HF_USER=myname SKIP_EVALS=1 ./run_all.sh

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these or override via environment variables
# ─────────────────────────────────────────────────────────────────────────────
HF_USER="${HF_USER:?Set HF_USER to your HuggingFace username}"
DEVICE="${DEVICE:-cuda}"
MODELS_DIR="${MODELS_DIR:-lcd_models}"
RESULTS_DIR="${RESULTS_DIR:-lcd_results}"
LOGS_DIR="${RESULTS_DIR}/logs"

# Toggle experiments (set to 1 to skip)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EVALS="${SKIP_EVALS:-0}"
SKIP_AGNOSTIC="${SKIP_AGNOSTIC:-0}"
SKIP_RANK_ABLATION="${SKIP_RANK_ABLATION:-0}"

# ─────────────────────────────────────────────────────────────────────────────
# Derived paths (set after download)
# ─────────────────────────────────────────────────────────────────────────────
QWEN_FP16="Qwen/Qwen2.5-3B"
QWEN_INT3="${MODELS_DIR}/qwen2.5-3b-int3-gptq"
QWEN_LCD="${MODELS_DIR}/qwen2.5-3b-int3-gptq-lcd"

LLAMA_FP16="meta-llama/Llama-3.2-3B"
LLAMA_INT3="${MODELS_DIR}/llama3.2-3b-int3-gptq"
LLAMA_LCD="${MODELS_DIR}/llama3.2-3b-int3-gptq-lcd"

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$LOGS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="${LOGS_DIR}/run_all_${TIMESTAMP}.log"

declare -A STATUS
TOTAL_T0=$(date +%s)

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

run_experiment() {
    local name="$1"
    local logfile="$2"
    shift 2
    local cmd=("$@")

    log "START  $name"
    log "  CMD: ${cmd[*]}"
    log "  LOG: $logfile"

    local t0=$(date +%s)

    if "${cmd[@]}" 2>&1 | tee "$logfile"; then
        local t1=$(date +%s)
        local dt=$(( (t1 - t0) / 60 ))
        STATUS[$name]="OK (${dt}min)"
        log "DONE   $name — ${dt} min"
        return 0
    else
        local t1=$(date +%s)
        local dt=$(( (t1 - t0) / 60 ))
        STATUS[$name]="FAILED (${dt}min)"
        log "FAILED $name — see $logfile"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  LCD Experiment Suite"
echo "  HF User    : $HF_USER"
echo "  Device     : $DEVICE"
echo "  Models dir : $MODELS_DIR/"
echo "  Results    : $RESULTS_DIR/"
echo "  Timestamp  : $TIMESTAMP"
echo "═══════════════════════════════════════════════════════════════"
echo ""
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Download models from HuggingFace Hub
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    log "Downloading models from HuggingFace Hub ..."

    download_model() {
        local repo="$1"
        local dest="$2"
        if [[ -d "$dest" ]] && [[ -f "$dest/config.json" ]]; then
            log "  CACHED  $dest"
            return 0
        fi
        log "  DOWNLOADING  $repo → $dest"
        huggingface-cli download "$repo" --local-dir "$dest"
    }

    download_model "${HF_USER}/qwen2.5-3b-int3-gptq"     "$QWEN_INT3"
    download_model "${HF_USER}/qwen2.5-3b-int3-gptq-lcd"  "$QWEN_LCD"
    download_model "${HF_USER}/llama3.2-3b-int3-gptq"     "$LLAMA_INT3"
    download_model "${HF_USER}/llama3.2-3b-int3-gptq-lcd"  "$LLAMA_LCD"

    # Verify LCD corrections exist
    for lcd_dir in "$QWEN_LCD" "$LLAMA_LCD"; do
        if [[ ! -f "${lcd_dir}/lcd_corrections.pt" ]]; then
            log "  WARNING: lcd_corrections.pt not found in ${lcd_dir}"
        fi
    done

    STATUS["download"]="OK"
    log "Download complete."
    echo ""
else
    log "SKIP   download (SKIP_DOWNLOAD=1)"
    STATUS["download"]="SKIPPED"
fi


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Full Evaluations (perplexity + GlobalMMLU, all conditions)
#    Run evals FIRST so we have FP16/INT3 baselines for ablation experiments
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_EVALS" != "1" ]]; then

    # ── Qwen2.5-3B ───────────────────────────────────────────────────────
    run_experiment \
        "evals/qwen2.5-3b" \
        "${LOGS_DIR}/evals_qwen_${TIMESTAMP}.log" \
        python run_evals.py \
            --fp16-model "$QWEN_FP16" \
            --int3-model "$QWEN_INT3" \
            --lcd-model  "$QWEN_LCD" \
            --device "$DEVICE" \
            --output-dir "$RESULTS_DIR" \
        || true

    # ── Llama-3.2-3B ─────────────────────────────────────────────────────
    run_experiment \
        "evals/llama3.2-3b" \
        "${LOGS_DIR}/evals_llama_${TIMESTAMP}.log" \
        python run_evals.py \
            --fp16-model "$LLAMA_FP16" \
            --int3-model "$LLAMA_INT3" \
            --lcd-model  "$LLAMA_LCD" \
            --device "$DEVICE" \
            --output-dir "$RESULTS_DIR" \
        || true

    # ── Synthesize lcd_results.json for ablation experiments ─────────────
    #    The agnostic/rank scripts expect {results: {lang: {fp16_ppl, ...}}}
    #    We build this from eval_fp16.json + eval_int3.json + eval_lcd.json
    log "Synthesizing lcd_results.json from eval outputs ..."

    python - "$RESULTS_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])

for model_name in ["qwen2.5-3b", "llama3.2-3b"]:
    model_dir = results_dir / model_name
    out_path  = model_dir / "lcd_results.json"

    fp16_path = model_dir / "eval_fp16.json"
    int3_path = model_dir / "eval_int3.json"
    lcd_path  = model_dir / "eval_lcd.json"

    if not fp16_path.exists() or not int3_path.exists():
        print(f"  SKIP {model_name} — missing eval_fp16.json or eval_int3.json")
        continue

    fp16_data = json.loads(fp16_path.read_text())
    int3_data = json.loads(int3_path.read_text())
    lcd_data  = json.loads(lcd_path.read_text()) if lcd_path.exists() else {}

    fp16_ppl = fp16_data.get("perplexity", {})
    int3_ppl = int3_data.get("perplexity", {})
    lcd_ppl  = lcd_data.get("perplexity", {})

    results = {}
    for lang in fp16_ppl:
        fp = fp16_ppl.get(lang, 0)
        q  = int3_ppl.get(lang, 0)
        c  = lcd_ppl.get(lang, q)
        gap = q - fp
        rec = (1 - (c - fp) / max(gap, 1e-8)) * 100 if gap > 0 else 0.0
        results[lang] = {
            "fp16_ppl": fp,
            "quant_ppl": q,
            "corrected_ppl": c,
            "recovery_pct": round(rec, 2),
        }

    out = {"model": fp16_data.get("model", model_name), "results": results}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  WROTE {out_path} ({len(results)} languages)")
PYEOF

else
    log "SKIP   evals (SKIP_EVALS=1)"
    STATUS["evals"]="SKIPPED"
fi


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Language-Agnostic Baseline (Qwen2.5-3B only)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_AGNOSTIC" != "1" ]]; then

    run_experiment \
        "agnostic-baseline" \
        "${LOGS_DIR}/agnostic_baseline_${TIMESTAMP}.log" \
        python exp_agnostic_baseline.py \
            --device "$DEVICE" \
            --quantized-dir "$QWEN_INT3" \
            --lcd-results "${RESULTS_DIR}/qwen2.5-3b/lcd_results.json" \
            --output "${RESULTS_DIR}/qwen2.5-3b/exp_agnostic_baseline.json" \
        || true

else
    log "SKIP   agnostic-baseline (SKIP_AGNOSTIC=1)"
    STATUS["agnostic-baseline"]="SKIPPED"
fi


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Rank Ablation (Qwen2.5-3B only)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_RANK_ABLATION" != "1" ]]; then

    run_experiment \
        "rank-ablation" \
        "${LOGS_DIR}/rank_ablation_${TIMESTAMP}.log" \
        python exp_rank_ablation.py \
            --device "$DEVICE" \
            --quantized-dir "$QWEN_INT3" \
            --lcd-results "${RESULTS_DIR}/qwen2.5-3b/lcd_results.json" \
            --output "${RESULTS_DIR}/qwen2.5-3b/exp_rank_ablation.json" \
        || true

else
    log "SKIP   rank-ablation (SKIP_RANK_ABLATION=1)"
    STATUS["rank-ablation"]="SKIPPED"
fi


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_T1=$(date +%s)
TOTAL_MIN=$(( (TOTAL_T1 - TOTAL_T0) / 60 ))
TOTAL_HR=$(echo "scale=1; $TOTAL_MIN / 60" | bc)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  EXPERIMENT SUITE COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
printf "  %-30s %s\n" "Experiment" "Status"
echo "  ──────────────────────────────────────────────────"
for key in "download" "evals/qwen2.5-3b" "evals/llama3.2-3b" "agnostic-baseline" "rank-ablation"; do
    if [[ -n "${STATUS[$key]+x}" ]]; then
        printf "  %-30s %s\n" "$key" "${STATUS[$key]}"
    fi
done
echo ""
echo "  Total wall time: ${TOTAL_MIN} min (${TOTAL_HR} hr)"
echo ""

# ── List all result files ─────────────────────────────────────────────────
echo "  Result files:"
echo "  ──────────────────────────────────────────────────"
find "$RESULTS_DIR" -name "*.json" -not -path "*/logs/*" 2>/dev/null | sort | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "    ${size}  ${f}"
done

echo ""
echo "  Logs      : ${LOGS_DIR}/"
echo "  Summary   : ${SUMMARY_LOG}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
