#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-strict_holdout}"
LEVELS="${LEVELS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
LIMIT="${LIMIT:-0}"
UI_TARS_MODEL="${UI_TARS_MODEL:-/home/dumike/zyy/GUI/backbone/UI-TARS-7B}"
UI_TARS_TEMPLATE="${UI_TARS_TEMPLATE:-qwen2_vl}"
SFT_CONFIG="${SFT_CONFIG:-$ROOT_DIR/configs/llamafactory/generated/ui_tars_7b_proactive_sft.yaml}"
SFT_ADAPTER="${SFT_ADAPTER:-$ROOT_DIR/LLaMA-Factory/saves/papo/ui_tars_7b_proactive_sft_clean_v2_best}"
EVAL_MODEL_LABEL="${EVAL_MODEL_LABEL:-}"
EVAL_ADAPTER="${EVAL_ADAPTER:-}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT_DIR/reports/ui_tars_proactive}"
EMBEDDING_MODEL="$ROOT_DIR/models/paraphrase-multilingual-MiniLM-L12-v2"
ACTION="${1:-audit}"

cd "$ROOT_DIR"
source server_env.sh
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a levels <<< "$LEVELS"

validate_common() {
  echo "===== Validate UI-TARS experiment environment ====="
  test -d "$UI_TARS_MODEL"
  test -d "$EMBEDDING_MODEL"
  test -f "$ROOT_DIR/evaluation/fingertip/evaluate_reports.py"
  python - "$UI_TARS_MODEL" <<'PY'
from pathlib import Path
import json
import sys

model = Path(sys.argv[1])
print("Model:", model)
print("Exists:", model.is_dir())
for name in ["config.json", "tokenizer_config.json", "generation_config.json", "README.md"]:
    path = model / name
    print(f"\n{name}: exists={path.exists()}")
    if path.suffix == ".json" and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ["_name_or_path", "model_type", "architectures", "eos_token_id", "pad_token_id"]:
            print(f"  {key}: {data.get(key)}")
        if "chat_template" in data:
            print("  has_chat_template: True")
PY
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
}

check_idle_gpus() {
  mapfile -t ACTIVE_GPU_PROCESSES < <(
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null |
    sed '/^[[:space:]]*$/d'
  )
  if [[ "${#ACTIVE_GPU_PROCESSES[@]}" -gt 0 && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    printf 'ERROR: Active GPU compute processes detected: %s\n' "${ACTIVE_GPU_PROCESSES[@]}" >&2
    echo "Set ALLOW_BUSY_GPUS=1 only after confirming resource sharing is intentional." >&2
    exit 1
  fi
}

render_sft_config() {
  echo "===== Render UI-TARS SFT config ====="
  python ui_tars_proactive/render_sft_config.py \
    --model "$UI_TARS_MODEL" \
    --template "$UI_TARS_TEMPLATE" \
    --output "$SFT_CONFIG"
  python scripts/15_training_preflight.py \
    --training-config "$SFT_CONFIG" \
    --check-only
}

train_sft() {
  validate_common
  check_idle_gpus
  render_sft_config
  echo "===== Train or resume UI-TARS Proactive SFT ====="
  NUM_GPUS="$NUM_SHARDS" bash server/train_auto_resume.sh "$SFT_CONFIG"
}

evaluate_one() {
  local model_label="$1"
  local adapter="$2"
  local level="$3"
  local effective_report_root="$REPORT_ROOT"
  if [[ "$LIMIT" -gt 0 ]]; then
    effective_report_root="$REPORT_ROOT/smoke_limit_${LIMIT}"
  fi
  local run_root="$effective_report_root/$model_label/$MODE"
  local run_dir="$run_root/level_$level"
  local tasks="$ROOT_DIR/data/papo_tasks/proactive_test_${MODE}_level_${level}.jsonl"
  local result_csv="$run_dir/${model_label}_${MODE}_level_${level}.csv"
  local pids=()
  local shards=()
  local logs=()
  local adapter_args=()
  local provenance_args=()
  local limit_args=()

  mkdir -p "$run_dir/shards"
  python scripts/17_prepare_proactive_evaluation.py \
    --config config.yaml \
    --screenshot-level "$level"
  test -s "$tasks"
  if [[ -n "$adapter" ]]; then
    test -f "$adapter/adapter_model.safetensors"
    test -f "$adapter/papo_training_provenance.json"
    adapter_args=(--adapter "$adapter")
    provenance_args=(--require-adapter-provenance)
  fi
  if [[ "$LIMIT" -gt 0 ]]; then
    limit_args=(--limit "$LIMIT")
  fi

  echo "===== Predict: model=$model_label mode=$MODE level=$level ====="
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="$run_dir/shards/shard_${shard}.jsonl"
    log_path="$run_dir/shards/shard_${shard}.log"
    shards+=("$shard_path")
    logs+=("$log_path")
    CUDA_VISIBLE_DEVICES="$shard" \
    python ui_tars_proactive/run_predictions.py \
      --config config.yaml \
      --tasks "$tasks" \
      --model "$UI_TARS_MODEL" \
      "${adapter_args[@]}" \
      --output "$shard_path" \
      --template "$UI_TARS_TEMPLATE" \
      --model-label "$model_label" \
      --shard-index "$shard" \
      --num-shards "$NUM_SHARDS" \
      "${limit_args[@]}" \
      "${provenance_args[@]}" \
      > "$log_path" 2>&1 &
    pids+=("$!")
  done

  while true; do
    active=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        active=$((active + 1))
      fi
    done
    if [[ "$active" -eq 0 ]]; then
      break
    fi
    echo "----- Active UI-TARS prediction shards: $active / $NUM_SHARDS -----"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    for log_path in "${logs[@]}"; do
      echo "----- $log_path -----"
      tail -n 2 "$log_path" 2>/dev/null || true
    done
    sleep "$MONITOR_INTERVAL"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: At least one UI-TARS prediction shard failed." >&2
    for log_path in "${logs[@]}"; do
      echo "===== Failure log: $log_path =====" >&2
      tail -n 120 "$log_path" >&2 || true
    done
    exit "$status"
  fi
  if [[ "$LIMIT" -gt 0 ]]; then
    echo "Smoke prediction completed for $model_label level $level; merge/eval skipped because LIMIT=$LIMIT."
    return
  fi

  python ui_tars_proactive/merge_predictions.py \
    --tasks "$tasks" \
    --output "$result_csv" \
    --model-label "$model_label" \
    --shards "${shards[@]}"

  python evaluation/fingertip/evaluate_reports.py \
    --proactive "$result_csv" \
    --embedding-model "$EMBEDDING_MODEL" \
    --output-dir "$run_dir/metrics"

  echo "===== Completed UI-TARS evaluation artifact list ====="
  find "$run_dir" -maxdepth 3 -type f -printf '%p  %k KB\n' | sort
}

evaluate_model() {
  local model_label="$1"
  local adapter="${2:-}"
  validate_common
  check_idle_gpus
  for level in "${levels[@]}"; do
    evaluate_one "$model_label" "$adapter" "$level"
  done
}

summarize() {
  if [[ "$LIMIT" -gt 0 ]]; then
    echo "LIMIT=$LIMIT, summary skipped because smoke runs do not create official metrics."
    return
  fi
  python ui_tars_proactive/summarize_results.py \
    --reports-root "$REPORT_ROOT" \
    --mode "$MODE" \
    --models ui_tars_7b_base ui_tars_7b_sft
}

summarize_adapter() {
  if [[ "$LIMIT" -gt 0 ]]; then
    echo "LIMIT=$LIMIT, summary skipped because smoke runs do not create official metrics."
    return
  fi
  python ui_tars_proactive/summarize_results.py \
    --reports-root "$REPORT_ROOT" \
    --mode "$MODE" \
    --models "$EVAL_MODEL_LABEL" \
    --output-dir "$REPORT_ROOT/summary/$EVAL_MODEL_LABEL"
}

case "$ACTION" in
  audit)
    validate_common
    echo "SFT config: $SFT_CONFIG"
    echo "SFT adapter: $SFT_ADAPTER"
    test -f "$SFT_ADAPTER/adapter_model.safetensors" && echo "SFT adapter exists: YES" || echo "SFT adapter exists: NO"
    ;;
  render_sft_config)
    render_sft_config
    ;;
  train_sft)
    train_sft
    ;;
  eval_base)
    evaluate_model ui_tars_7b_base ""
    summarize
    ;;
  eval_sft)
    evaluate_model ui_tars_7b_sft "$SFT_ADAPTER"
    summarize
    ;;
  eval_adapter)
    if [[ -z "$EVAL_MODEL_LABEL" || -z "$EVAL_ADAPTER" ]]; then
      echo "EVAL_MODEL_LABEL and EVAL_ADAPTER are required for eval_adapter." >&2
      exit 2
    fi
    evaluate_model "$EVAL_MODEL_LABEL" "$EVAL_ADAPTER"
    summarize_adapter
    ;;
  eval_all)
    evaluate_model ui_tars_7b_base ""
    evaluate_model ui_tars_7b_sft "$SFT_ADAPTER"
    summarize
    ;;
  full)
    train_sft
    evaluate_model ui_tars_7b_base ""
    evaluate_model ui_tars_7b_sft "$SFT_ADAPTER"
    summarize
    ;;
  summary)
    summarize
    ;;
  *)
    echo "Usage: bash ui_tars_proactive/run_ui_tars_7b.sh {audit|render_sft_config|train_sft|eval_base|eval_sft|eval_adapter|eval_all|full|summary}" >&2
    exit 2
    ;;
esac
