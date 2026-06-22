#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dumike/zyy/GUI2}"
RELEASE_ID="20260622T042745Z"
SOURCE_RELEASE="${SOURCE_RELEASE:-$PROJECT_ROOT/data/releases/papo_group_listwise_v4_smoke/$RELEASE_ID}"
GROUP_DATASET_DIR="$PROJECT_ROOT/LLaMA-Factory/data/papo_group_listwise_v4_smoke_$RELEASE_ID"
CONTROL_DATASET_DIR="$PROJECT_ROOT/LLaMA-Factory/data/papo_v4_oracle_control_smoke_$RELEASE_ID"
GROUP_CONFIG="$PROJECT_ROOT/configs/llamafactory/ui_tars_7b_papo_group_listwise_v4_retrieval_smoke.yaml"
CONTROL_CONFIG="$PROJECT_ROOT/configs/llamafactory/ui_tars_7b_papo_v4_oracle_control_smoke.yaml"
GROUP_OUTPUT="$PROJECT_ROOT/LLaMA-Factory/saves/papo/ui_tars_7b_papo_group_listwise_v4_retrieval_smoke_$RELEASE_ID"
CONTROL_OUTPUT="$PROJECT_ROOT/LLaMA-Factory/saves/papo/ui_tars_7b_papo_v4_oracle_control_smoke_$RELEASE_ID"
RUN_DIR="$PROJECT_ROOT/runs/papo/group_listwise_v4_retrieval_smoke_$RELEASE_ID"
REPORT_DIR="$PROJECT_ROOT/reports/proactive/group_listwise_v4_retrieval_smoke_$RELEASE_ID"
ACTION="${1:-status}"

cd "$PROJECT_ROOT"
source server_env.sh
mkdir -p "$RUN_DIR" "$REPORT_DIR"

latest_log() {
  find "$RUN_DIR" -maxdepth 1 -type f -name 'grouped_train_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

require_idle_gpus() {
  local active
  active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$active" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    echo "ERROR: active GPU processes detected:" >&2
    echo "$active" >&2
    echo "Set ALLOW_BUSY_GPUS=1 only after confirming intentional sharing." >&2
    exit 1
  fi
}

prepare() {
  echo "===== Verify and register unchanged retrieval-only smoke release ====="
  test -f "$SOURCE_RELEASE/listwise_v4_manifest.json"
  python scripts/28_register_proactive_listwise_v4.py \
    --release-dir "$SOURCE_RELEASE" \
    --dataset-dir "$GROUP_DATASET_DIR" \
    --allow-synthetic-smoke

  echo "===== Build same-task oracle-only control without changing source release ====="
  if [[ ! -f "$CONTROL_DATASET_DIR/oracle_control_manifest.json" ]]; then
    python scripts/31_prepare_papo_group_listwise_v4_smoke_control.py \
      --release-dir "$SOURCE_RELEASE" \
      --output-dir "$CONTROL_DATASET_DIR"
  else
    python scripts/31_prepare_papo_group_listwise_v4_smoke_control.py \
      --release-dir "$SOURCE_RELEASE" \
      --output-dir "$CONTROL_DATASET_DIR" \
      --verify-only
  fi

  echo "===== Run data, environment, image and configuration preflight ====="
  python scripts/32_preflight_papo_group_listwise_v4_smoke.py \
    --training-config "$GROUP_CONFIG" \
    --release-dir "$GROUP_DATASET_DIR" \
    --report "$REPORT_DIR/server_preflight.json"

  echo "===== Run grouped loading/loss regression tests ====="
  python -m unittest discover -s tests -p 'test_papo_group_listwise_v4_loss.py' -v
  python -m unittest discover -s tests -p 'test_papo_group_listwise_v4_smoke_experiment.py' -v
  echo "PAPO retrieval-only grouped Listwise-v4 smoke preparation passed."
}

train_grouped() {
  prepare
  require_idle_gpus
  if pgrep -af "llamafactory.*$(basename "$GROUP_CONFIG")" >/dev/null; then
    echo "ERROR: grouped smoke training is already active." >&2
    exit 1
  fi
  local log pid
  log="$RUN_DIR/grouped_train_$(date +%Y%m%d_%H%M%S).log"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  nohup llamafactory-cli train "$GROUP_CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_DIR/grouped_train.pid"
  echo "PID: $pid"
  echo "Log: $log"
  sleep 20
  tail -n 80 "$log" || true
}

train_control() {
  prepare
  require_idle_gpus
  if pgrep -af "llamafactory.*$(basename "$CONTROL_CONFIG")" >/dev/null; then
    echo "ERROR: control smoke training is already active." >&2
    exit 1
  fi
  local log pid
  log="$RUN_DIR/control_train_$(date +%Y%m%d_%H%M%S).log"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  nohup llamafactory-cli train "$CONTROL_CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_DIR/control_train.pid"
  echo "PID: $pid"
  echo "Log: $log"
  sleep 20
  tail -n 80 "$log" || true
}

status() {
  echo "===== Processes ====="
  pgrep -af 'ui_tars_7b_papo_(group_listwise_v4_retrieval|v4_oracle_control)_smoke' || echo "No active smoke process"
  echo "===== GPUs ====="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo "===== Latest grouped log ====="
  local log
  log="$(latest_log)"
  if [[ -n "$log" ]]; then
    echo "$log"
    tail -n 100 "$log"
  else
    echo "No grouped training log yet"
  fi
}

report() {
  local log
  log="$(latest_log)"
  if [[ -z "$log" ]]; then
    echo "ERROR: no grouped training log found in $RUN_DIR" >&2
    exit 1
  fi
  python scripts/33_report_papo_group_listwise_v4_smoke.py \
    --training-config "$GROUP_CONFIG" \
    --output-dir "$GROUP_OUTPUT" \
    --log "$log" \
    --report-dir "$REPORT_DIR"
  cat "$REPORT_DIR/group_listwise_v4_smoke_report.md"
}

case "$ACTION" in
  prepare) prepare ;;
  train-grouped) train_grouped ;;
  train-control) train_control ;;
  status) status ;;
  report) report ;;
  *)
    echo "Usage: bash server/run_papo_group_listwise_v4_retrieval_smoke.sh {prepare|train-grouped|train-control|status|report}" >&2
    exit 2
    ;;
esac
