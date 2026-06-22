#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ROOT_CONFIG="${1:-../config.yaml}"

python -m tn_dpo_gui.scripts.preprocess_data --root-config "$ROOT_CONFIG"
python -m tn_dpo_gui.scripts.build_user_index --root-config "$ROOT_CONFIG"
python -m tn_dpo_gui.scripts.build_pairs --config configs/build_pairs.yaml
python -m tn_dpo_gui.scripts.train_ranker --config configs/train_ranker.yaml
python -m tn_dpo_gui.scripts.train_gate --config configs/train_gate.yaml
python -m tn_dpo_gui.scripts.eval --config configs/eval.yaml
