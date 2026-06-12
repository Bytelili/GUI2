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

if [[ "$NUM_GPUS" -gt 1 ]]; then
  FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
else
  llamafactory-cli train "$CONFIG"
fi
