#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m tn_dpo_gui.scripts.preprocess_data --demo --output-dir data/demo
python -m tn_dpo_gui.scripts.build_user_index --trajectories data/demo/trajectories.jsonl --output data/demo/user_index.json
python -m tn_dpo_gui.scripts.build_pairs --config configs/build_pairs.demo.yaml
python -m tn_dpo_gui.scripts.train_ranker --config configs/train_ranker.demo.yaml
python -m tn_dpo_gui.scripts.train_gate --config configs/train_gate.demo.yaml
python -m tn_dpo_gui.scripts.eval --config configs/eval.demo.yaml
