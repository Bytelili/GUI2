#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/proactive_preference_pipeline"
MODE="${1:-audit}"
NUM_GPUS="${NUM_GPUS:-4}"
SFT_ADAPTER="${SFT_ADAPTER:-$ROOT_DIR/LLaMA-Factory/saves/papo/proactive_sft_clean_v2_best}"
CANDIDATE_DIR="$ROOT_DIR/data/proactive_preference/model_candidates"
LOG_DIR="$ROOT_DIR/runs/proactive_preference"
TRAIN_TASKS="$ROOT_DIR/data/papo_tasks/proactive_train_config.jsonl"
EVAL_TASKS="$ROOT_DIR/data/papo_tasks/proactive_eval_config.jsonl"
TRAIN_CANDIDATES="$CANDIDATE_DIR/train_candidates.jsonl"
EVAL_CANDIDATES="$CANDIDATE_DIR/eval_candidates.jsonl"

cd "$ROOT_DIR"
mkdir -p "$CANDIDATE_DIR" "$LOG_DIR"

generate_split() {
  local split="$1"
  local tasks="$2"
  local output="$3"
  local pids=()
  local logs=()
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    log="$LOG_DIR/generate_${split}_shard_${gpu}.log"
    logs+=("$log")
    CUDA_VISIBLE_DEVICES="$gpu" python "$PIPELINE_DIR/generate_model_candidates.py" \
      --tasks "$tasks" \
      --adapter "$SFT_ADAPTER" \
      --output "$CANDIDATE_DIR/${split}_shard_${gpu}.jsonl" \
      --shard-index "$gpu" \
      --num-shards "$NUM_GPUS" \
      --num-candidates 4 \
      > "$log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: At least one $split candidate shard failed." >&2
    for log in "${logs[@]}"; do
      echo "===== $log =====" >&2
      tail -n 100 "$log" >&2 || true
    done
    return "$status"
  fi
  python "$PIPELINE_DIR/merge_candidate_shards.py" \
    --tasks "$tasks" \
    --shards "$CANDIDATE_DIR/${split}_shard_"*.jsonl \
    --output "$output"
}

prepare_preferences() {
  local candidate_args=()
  if [[ -s "$TRAIN_CANDIDATES" && -s "$EVAL_CANDIDATES" ]]; then
    candidate_args=(
      --train-model-candidates "$TRAIN_CANDIDATES"
      --eval-model-candidates "$EVAL_CANDIDATES"
    )
  fi
  python "$PIPELINE_DIR/build_preferences.py" "${candidate_args[@]}"
  python "$PIPELINE_DIR/render_training_configs.py"
  python scripts/08_validate_llamafactory_data.py \
    --dataset_dir LLaMA-Factory/data/papo \
    --datasets \
    papo_proactive_train_listwise,papo_proactive_eval_listwise,papo_proactive_train_dpo,papo_proactive_eval_dpo \
    --check_images
  python scripts/15_training_preflight.py \
    --training-config configs/llamafactory/generated/proactive_preference_listwise.yaml \
    --check-only
  python "$PIPELINE_DIR/preflight.py" \
    --training-config configs/llamafactory/generated/proactive_preference_listwise.yaml
}

case "$MODE" in
  generate)
    generate_split train "$TRAIN_TASKS" "$TRAIN_CANDIDATES"
    generate_split eval "$EVAL_TASKS" "$EVAL_CANDIDATES"
    ;;
  prepare)
    prepare_preferences
    ;;
  listwise)
    python "$PIPELINE_DIR/preflight.py" \
      --training-config configs/llamafactory/generated/proactive_preference_listwise.yaml
    bash server/train_auto_resume.sh \
      configs/llamafactory/generated/proactive_preference_listwise.yaml
    ;;
  dpo)
    python "$PIPELINE_DIR/preflight.py" \
      --training-config configs/llamafactory/generated/proactive_preference_dpo.yaml
    python scripts/15_training_preflight.py \
      --training-config configs/llamafactory/generated/proactive_preference_dpo.yaml \
      --check-only
    bash server/train_auto_resume.sh \
      configs/llamafactory/generated/proactive_preference_dpo.yaml
    ;;
  audit)
    echo "===== Preference manifest ====="
    cat data/proactive_preference/preference_manifest.json 2>/dev/null || true
    echo "===== Generated configs ====="
    for config in \
      configs/llamafactory/generated/proactive_preference_listwise.yaml \
      configs/llamafactory/generated/proactive_preference_dpo.yaml
    do
      echo "--- $config"
      cat "$config" 2>/dev/null || true
    done
    echo "===== Checkpoints ====="
    find LLaMA-Factory/saves/papo -maxdepth 2 \
      -path '*proactive_preference*' -printf '%p\n' 2>/dev/null | sort -V || true
    echo "===== GPUs ====="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
    ;;
  *)
    echo "Usage: bash proactive_preference_pipeline/run_pipeline.sh {generate|prepare|listwise|dpo|audit}" >&2
    exit 2
    ;;
esac
