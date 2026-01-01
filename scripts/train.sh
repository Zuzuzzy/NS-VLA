#!/usr/bin/env bash
# NS-VLA training entry point.
#
#   bash scripts/train.sh --stage pretrain --config configs/train/pretrain.yaml
#   bash scripts/train.sh --stage rl       --config configs/train/rl_grpo.yaml
#
# Any further arguments are passed through to the underlying python entry point
# (scripts/train_supervised.py or scripts/train_rl.py); --help lists them.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
STAGE=""
CONFIG=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)  STAGE="$2";  shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$STAGE" || -z "$CONFIG" ]]; then
  echo "usage: bash scripts/train.sh --stage {pretrain|rl} --config <yaml> [options]" >&2
  exit 2
fi

case "$STAGE" in
  pretrain) ENTRY="$REPO/scripts/train_supervised.py" ;;
  rl)       ENTRY="$REPO/scripts/train_rl.py" ;;
  *) echo "unknown stage '$STAGE' (expected: pretrain, rl)" >&2; exit 2 ;;
esac

exec "$PYTHON" "$ENTRY" --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
