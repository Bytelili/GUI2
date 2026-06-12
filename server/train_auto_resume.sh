#!/usr/bin/env bash
set -uo pipefail

CONFIG="${1:?Usage: bash server/train_auto_resume.sh CONFIG.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config does not exist: $CONFIG"
else
    OUTPUT_DIR="$(
        python - "$CONFIG" <<'PYTHON'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as file:
    config = yaml.safe_load(file)

print(config["output_dir"])
PYTHON
    )"

    mkdir -p "$OUTPUT_DIR"

    LATEST_CHECKPOINT="$(
        find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null |
        sort -V |
        tail -n 1
    )"

    RUNTIME_CONFIG="$(mktemp /tmp/papo_train_runtime_XXXXXX.yaml)"

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

runtime.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PYTHON

    echo "===== Automatic resume information ====="
    echo "Original config: $CONFIG"
    echo "Runtime config: $RUNTIME_CONFIG"
    echo "Output directory: $OUTPUT_DIR"
    echo "Latest checkpoint: ${LATEST_CHECKPOINT:-NONE}"
    echo "GPU count: $NUM_GPUS"

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Training mode: resume from latest checkpoint"
    else
        echo "Training mode: start from SFT adapter"
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    NUM_GPUS="$NUM_GPUS" \
    bash server/train.sh "$RUNTIME_CONFIG"

    TRAIN_STATUS=$?
    rm -f "$RUNTIME_CONFIG"

    echo "Training exit status: $TRAIN_STATUS"
    echo "Automatic resume launcher finished; parent terminal remains open."

    if [ "$TRAIN_STATUS" -eq 0 ]; then
        true
    else
        false
    fi
fi
