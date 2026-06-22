#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m tn_dpo_gui.scripts.preprocess_data --demo --output-dir data/demo
python -m tn_dpo_gui.scripts.build_user_index --trajectories data/demo/trajectories.jsonl --output data/demo/user_index.json
python -m tn_dpo_gui.scripts.build_pairs --config configs/build_pairs.yaml
python -m tn_dpo_gui.scripts.train_ranker --config configs/train_ranker.yaml
python -m tn_dpo_gui.scripts.train_gate --config configs/train_gate.yaml
python -m tn_dpo_gui.scripts.eval --config configs/eval.yaml
