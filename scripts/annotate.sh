#!/usr/bin/env bash
# Build the Stage-I supervision: 1-shot manifest -> primitive segments -> VLM features.
#
#   bash scripts/annotate.sh --libero-root third_party/LIBERO
#
# Steps 1 and 2 are CPU-only (step 2 needs the simulator for articulated tasks);
# step 3 needs a GPU for the frozen encoder and is skipped with --no-features.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MANIFEST="$REPO/data/manifests/libero_1shot.json"
WITH_FEATURES=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)    MANIFEST="$2"; shift 2 ;;
    --no-features) WITH_FEATURES=0; shift ;;
    *)             EXTRA+=("$1"); shift ;;
  esac
done

"$PYTHON" "$REPO/data/prepare_1shot.py"   --out "$MANIFEST" ${EXTRA[@]+"${EXTRA[@]}"}
"$PYTHON" "$REPO/data/annotate_demos.py"  --manifest "$MANIFEST"
if [[ "$WITH_FEATURES" == "1" ]]; then
  "$PYTHON" "$REPO/data/extract_features.py" --manifest "$MANIFEST"
fi
