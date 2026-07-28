#!/usr/bin/env bash
# NS-VLA evaluation entry point.
#
#   bash scripts/eval.sh --benchmark libero      --checkpoint runs/pointer/clf.pt
#   bash scripts/eval.sh --benchmark libero_plus --checkpoint runs/pointer/clf.pt
#
# --benchmark selects a config under configs/eval/; any further arguments are passed
# through to scripts/run_eval.py, whose --help lists them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
BENCHMARK="libero"
CHECKPOINT=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)  BENCHMARK="$2";  shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

CONFIG="$REPO/configs/eval/${BENCHMARK}.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "no config for benchmark '$BENCHMARK' (looked for $CONFIG)" >&2
  exit 2
fi

ARGS=(--config "$CONFIG")
[[ -n "$CHECKPOINT" ]] && ARGS+=(--checkpoint "$CHECKPOINT")

exec "$PYTHON" "$REPO/scripts/run_eval.py" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
