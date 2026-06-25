#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config.yaml"
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
PAPO_WORK_DIR="$(resolve_config_path paths.work_dir)"

mkdir -p "$TRAIN_DATA_DIR"
ln -sfn "$PAPO_RAW_ROOT" "$TRAIN_DATA_DIR/RawDataset"

if [[ -d "$ROOT_DIR/data/proactive_fixed_clean" ]]; then
  echo "[prepare] Syncing proactive_fixed_clean into LLaMA-Factory data dir..."
  mkdir -p "$TRAIN_DATA_DIR/proactive_fixed_clean"
  rsync -a "$ROOT_DIR/data/proactive_fixed_clean/" "$TRAIN_DATA_DIR/proactive_fixed_clean/"
else
  echo "[prepare] proactive_fixed_clean not found, skipping fixed proactive data sync."
fi

python "$ROOT_DIR/scripts/12_validate_config_paths.py" \
  --config "$CONFIG_PATH" \
  --create_output_dirs
python "$ROOT_DIR/scripts/13_smoke_test_papo_objective.py"
python "$ROOT_DIR/scripts/14_build_data_protocol.py" --config "$CONFIG_PATH"
python "$ROOT_DIR/scripts/09_run_config_pipeline.py" \
  --config "$CONFIG_PATH"
python "$ROOT_DIR/scripts/10_render_training_configs.py" --config "$CONFIG_PATH"
python "$ROOT_DIR/scripts/08_validate_llamafactory_data.py" \
  --dataset_dir "$TRAIN_DATA_DIR" \
  --check_images
python "$ROOT_DIR/scripts/11_validate_papo_artifacts.py" \
  --work_dir "$PAPO_WORK_DIR"

echo "Training data ready at $TRAIN_DATA_DIR"
