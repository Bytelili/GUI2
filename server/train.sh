#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv}"
CONFIG="${1:?Usage: server/train.sh configs/llamafactory/proactive_sft.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "ERROR: Environment not found at $ENV_DIR. Run bash server/setup_server.sh first." >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
cd "$ROOT_DIR"

OUTPUT_DIR="$(
python - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as file:
    print(yaml.safe_load(file)["output_dir"])
PY
)"

if [[ "$(basename "$OUTPUT_DIR")" == *clean_v2* ]] && [[ ! -f "$OUTPUT_DIR/papo_preflight.json" ]]; then
  echo "ERROR: Strict formal training cannot bypass the preflight gate." >&2
  echo "Use: bash server/train_auto_resume.sh $CONFIG" >&2
  exit 1
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
else
  llamafactory-cli train "$CONFIG"
fi
