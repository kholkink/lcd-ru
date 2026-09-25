#!/usr/bin/env bash
# Package the committed code (git HEAD, no data / runs / ref PDFs) as the private Kaggle dataset "<user>/lcd-ru-code"
# and write GIT_HASH into it (every run records it). Refuses a dirty tree unless ALLOW_DIRTY=1 (then the working tree
# is packed and the hash gets a "-dirty" suffix).      Usage: bash kaggle/push_code.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=${KAGGLE:-$(command -v kaggle || echo .venv/bin/kaggle)}
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
USER_NAME=$($PY kaggle/kutil.py whoami)
STAGE=$(mktemp -d)
HEAD=$(git rev-parse HEAD 2>/dev/null || echo none)
if [ -z "$(git status --porcelain)" ] && [ "$HEAD" != none ]; then
  git archive HEAD | tar -x -C "$STAGE"
  HASH=$HEAD
elif [ "${ALLOW_DIRTY:-0}" = 1 ]; then
  rsync -a --exclude '/.venv' --exclude '/.env' --exclude '/data/texts' --exclude '/runs' --exclude '/build' \
        --exclude '/.git' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pt' --exclude '*.safetensors' \
        ./ "$STAGE/"
  HASH="$HEAD-dirty"
else
  echo "working tree is dirty (or no commit yet): commit first, or ALLOW_DIRTY=1 bash kaggle/push_code.sh"; exit 1
fi
rm -rf "$STAGE/ref/lcd/paper.pdf"
VERSION=$(date -u +%Y%m%d%H%M%S)
echo "$VERSION" > "$STAGE/CODE_VERSION"
echo "$HASH" > "$STAGE/GIT_HASH"
for must in src/__init__.py src/pipeline.py scripts/smoke.py colab/worker.py colab/session_setup.sh requirements.txt; do
  [ -f "$STAGE/$must" ] || { echo "MISSING in staged code: $must"; exit 1; }
done
grep -rIl --exclude=push_code.sh "hf_[A-Za-z0-9]\{30,\}\|KGAT_" "$STAGE" && { echo "secret found in staged code"; exit 1; } || true
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "lcd-ru-code", "id": "${USER_NAME}/lcd-ru-code", "licenses": [{"name": "other"}]}
JSON
if "$KAGGLE" datasets status "${USER_NAME}/lcd-ru-code" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "update $(date +%F_%H%M) $HASH" --dir-mode zip
else
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode zip
fi
rm -rf "$STAGE"
mkdir -p build && echo "$VERSION" > build/code_version.txt   # colab/jobs.py stamps new jobs with it
echo "dataset: https://www.kaggle.com/datasets/${USER_NAME}/lcd-ru-code (private), version $VERSION, git $HASH"
