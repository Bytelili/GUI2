#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-strict_holdout}"
LEVEL="${2:-3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
ADAPTER="$ROOT_DIR/LLaMA-Factory/saves/papo/proactive_sft_clean_v2_best"
TASKS="$ROOT_DIR/data/papo_tasks/proactive_test_${MODE}_level_${LEVEL}.jsonl"
RUN_DIR="$ROOT_DIR/reports/proactive/${MODE}/level_${LEVEL}"
RESULT_CSV="$RUN_DIR/proactive_best_${MODE}_level_${LEVEL}.csv"
EMBEDDING_MODEL="$ROOT_DIR/models/paraphrase-multilingual-MiniLM-L12-v2"

if [[ "$MODE" != "strict_holdout" && "$MODE" != "official_online" ]]; then
  echo "ERROR: MODE must be strict_holdout or official_online." >&2
  exit 1
fi
if [[ "$LEVEL" != "0" && "$LEVEL" != "1" && "$LEVEL" != "2" && "$LEVEL" != "3" ]]; then
  echo "ERROR: LEVEL must be 0, 1, 2, or 3." >&2
  exit 1
fi
if [[ "$NUM_SHARDS" -lt 1 || "$NUM_SHARDS" -gt 4 ]]; then
  echo "ERROR: NUM_SHARDS must be between 1 and 4." >&2
  exit 1
fi

cd "$ROOT_DIR"
source server_env.sh
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RUN_DIR/shards"

echo "===== 1. Validate environment and idle GPUs ====="
test -d "$EMBEDDING_MODEL"
test -f "$ROOT_DIR/evaluation/fingertip/evaluate_reports.py"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$NUM_SHARDS" -gt "$GPU_COUNT" ]]; then
  echo "ERROR: NUM_SHARDS=$NUM_SHARDS exceeds detected GPU count $GPU_COUNT." >&2
  exit 1
fi
mapfile -t ACTIVE_GPU_PROCESSES < <(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null |
  sed '/^[[:space:]]*$/d'
)
if [[ "${#ACTIVE_GPU_PROCESSES[@]}" -gt 0 && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  printf 'ERROR: Active GPU compute processes detected: %s\n' "${ACTIVE_GPU_PROCESSES[@]}" >&2
  echo "Set ALLOW_BUSY_GPUS=1 only after confirming resource sharing is intentional." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "===== 2. Prepare audited Proactive official-test tasks ====="
python scripts/17_prepare_proactive_evaluation.py \
  --config config.yaml \
  --screenshot-level "$LEVEL"

echo "===== 3. Validate finalized best adapter and task file ====="
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$ADAPTER/papo_training_provenance.json"
test -s "$TASKS"
sha256sum "$TASKS" "$ADAPTER/adapter_model.safetensors" "$ADAPTER/papo_training_provenance.json"

echo "===== 4. Run resumable prediction shards ====="
pids=()
shards=()
logs=()
stop_children() {
  echo "Interrupted; stopping prediction shards..." >&2
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
  exit 130
}
trap stop_children INT TERM

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_path="$RUN_DIR/shards/shard_${shard}.jsonl"
  log_path="$RUN_DIR/shards/shard_${shard}.log"
  shards+=("$shard_path")
  logs+=("$log_path")
  CUDA_VISIBLE_DEVICES="$shard" \
  python scripts/18_run_proactive_predictions.py \
    --config config.yaml \
    --tasks "$TASKS" \
    --adapter "$ADAPTER" \
    --output "$shard_path" \
    --shard-index "$shard" \
    --num-shards "$NUM_SHARDS" \
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
  echo "----- Active prediction shards: $active / $NUM_SHARDS -----"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  for log_path in "${logs[@]}"; do
    echo "----- $log_path -----"
    tail -n 2 "$log_path" 2>/dev/null || true
  done
  sleep "$MONITOR_INTERVAL"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: At least one prediction shard failed." >&2
  for log_path in "${logs[@]}"; do
    echo "===== Failure log: $log_path =====" >&2
    tail -n 100 "$log_path" >&2 || true
  done
  exit "$status"
fi
trap - INT TERM

echo "===== 5. Merge and validate complete predictions ====="
python scripts/19_merge_proactive_predictions.py \
  --tasks "$TASKS" \
  --adapter "$ADAPTER" \
  --output "$RESULT_CSV" \
  --shards "${shards[@]}"

echo "===== 6. Run official FingerTip Proactive similarity evaluation ====="
python evaluation/fingertip/evaluate_reports.py \
  --proactive "$RESULT_CSV" \
  --embedding-model "$EMBEDDING_MODEL" \
  --output-dir "$RUN_DIR/metrics"

echo "===== 7. Show result artifacts ====="
find "$RUN_DIR" -maxdepth 3 -type f -printf '%p  %k KB\n' | sort
echo "Proactive evaluation completed: mode=$MODE, level=$LEVEL"
