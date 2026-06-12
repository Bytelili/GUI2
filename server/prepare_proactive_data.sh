#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config.yaml"
PROACTIVE_CONFIG="$ROOT_DIR/configs/llamafactory/generated/proactive_sft.yaml"
RAW_ROOT="${RAW_ROOT:-}"

if [[ -n "$RAW_ROOT" ]]; then
  export PAPO_RAW_ROOT="$RAW_ROOT"
fi

resolve_config_path() {
python - "$CONFIG_PATH" "$1" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent / "src"))
from papo.config import config_path, load_config

print(config_path(load_config(sys.argv[1]), sys.argv[2]))
PY
}

PAPO_RAW_ROOT="$(resolve_config_path paths.raw_root)"
TRAIN_DATA_DIR="$(resolve_config_path paths.llamafactory_data_dir)"

mkdir -p "$TRAIN_DATA_DIR"
ln -sfn "$PAPO_RAW_ROOT" "$TRAIN_DATA_DIR/RawDataset"

echo "===== 1. Validate configured paths ====="
python "$ROOT_DIR/scripts/12_validate_config_paths.py" \
  --config "$CONFIG_PATH" \
  --create_output_dirs

echo "===== 2. Build strict temporal data protocol ====="
python "$ROOT_DIR/scripts/14_build_data_protocol.py" --config "$CONFIG_PATH"

echo "===== 3. Build only Proactive train/eval tasks and exports ====="
python "$ROOT_DIR/scripts/09_run_config_pipeline.py" \
  --config "$CONFIG_PATH" \
  --stages proactive_tasks,proactive_export

echo "===== 4. Render training configurations ====="
python "$ROOT_DIR/scripts/10_render_training_configs.py" --config "$CONFIG_PATH"

echo "===== 5. Validate only Proactive train/eval datasets and images ====="
python "$ROOT_DIR/scripts/08_validate_llamafactory_data.py" \
  --dataset_dir "$TRAIN_DATA_DIR" \
  --datasets papo_proactive_train_sft,papo_proactive_eval_sft \
  --check_images

echo "===== 6. Run strict Proactive training preflight without creating a resume gate ====="
python "$ROOT_DIR/scripts/15_training_preflight.py" \
  --config "$CONFIG_PATH" \
  --training-config "$PROACTIVE_CONFIG" \
  --check-only

echo "Proactive data is ready at $TRAIN_DATA_DIR"
echo "No Execution data was built and no training was started."
