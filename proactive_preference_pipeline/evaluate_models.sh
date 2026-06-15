#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-strict_holdout}"
MODELS="${MODELS:-listwise,dpo}"
LEVELS="${LEVELS:-0,1,2,3}"

cd "$ROOT_DIR"
IFS=',' read -r -a model_names <<< "$MODELS"
IFS=',' read -r -a levels <<< "$LEVELS"

for model_name in "${model_names[@]}"; do
  case "$model_name" in
    listwise)
      adapter="$ROOT_DIR/LLaMA-Factory/saves/papo/proactive_preference_listwise_clean_v2_best"
      ;;
    dpo)
      adapter="$ROOT_DIR/LLaMA-Factory/saves/papo/proactive_preference_dpo_clean_v2_best"
      ;;
    *)
      echo "ERROR: Unknown model '$model_name'. Expected listwise or dpo." >&2
      exit 2
      ;;
  esac
  test -f "$adapter/adapter_model.safetensors"
  test -f "$adapter/papo_training_provenance.json"
  for level in "${levels[@]}"; do
    ADAPTER="$adapter" \
    RUN_ROOT="$ROOT_DIR/reports/proactive_preference/${model_name}/${MODE}" \
    RESULT_NAME="proactive_preference_${model_name}" \
    bash server/evaluate_proactive_best.sh "$MODE" "$level"
  done
done

python proactive_preference_pipeline/summarize_results.py \
  --reports-root reports/proactive_preference \
  --mode "$MODE" \
  --models "${model_names[@]}"
