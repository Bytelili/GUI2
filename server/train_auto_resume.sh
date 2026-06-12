#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?Usage: bash server/train_auto_resume.sh CONFIG.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

cd "$ROOT_DIR"
python scripts/15_training_preflight.py --training-config "$CONFIG"

OUTPUT_DIR="$(
python - "$CONFIG" <<'PYTHON'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as file:
    print(yaml.safe_load(file)["output_dir"])
PYTHON
)"

LATEST_CHECKPOINT="$(
find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null |
sort -V |
tail -n 1
)"
RUNTIME_CONFIG="$(mktemp /tmp/papo_train_runtime_XXXXXX.yaml)"
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

python - "$CONFIG" "$RUNTIME_CONFIG" "$LATEST_CHECKPOINT" <<'PYTHON'
import sys
from pathlib import Path
import yaml

source = Path(sys.argv[1])
runtime = Path(sys.argv[2])
checkpoint = sys.argv[3]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
if checkpoint:
    config["resume_from_checkpoint"] = checkpoint
else:
    config.pop("resume_from_checkpoint", None)
runtime.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PYTHON

echo "===== Strict automatic resume information ====="
echo "Original config: $CONFIG"
echo "Runtime config: $RUNTIME_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Latest checkpoint: ${LATEST_CHECKPOINT:-NONE}"
echo "GPU count: $NUM_GPUS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NUM_GPUS="$NUM_GPUS" \
bash server/train.sh "$RUNTIME_CONFIG"

python scripts/16_finalize_best_checkpoint.py --training-config "$CONFIG"
echo "Strict training, resume, and best-checkpoint finalization completed."
