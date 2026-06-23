#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dumike/zyy/GUI2}"
RELEASE_ID="20260622T065645Z"
SOURCE_RELEASE="${SOURCE_RELEASE:-$PROJECT_ROOT/data/releases/papo_group_listwise_v4_retrieval_only/$RELEASE_ID}"
DATASET_DIR="$PROJECT_ROOT/LLaMA-Factory/data/papo_group_listwise_v4_retrieval_only_$RELEASE_ID"
CONFIG="$PROJECT_ROOT/configs/llamafactory/ui_tars_7b_papo_group_listwise_v4_retrieval_only.yaml"
OUTPUT="$PROJECT_ROOT/LLaMA-Factory/saves/papo/ui_tars_7b_papo_group_listwise_v4_retrieval_only_$RELEASE_ID"
RUN_DIR="$PROJECT_ROOT/runs/papo/group_listwise_v4_retrieval_only_$RELEASE_ID"
REPORT_DIR="$PROJECT_ROOT/reports/proactive/group_listwise_v4_retrieval_only_$RELEASE_ID"
ACTION="${1:-status}"

cd "$PROJECT_ROOT"
source server_env.sh
mkdir -p "$RUN_DIR" "$REPORT_DIR"

latest_log() {
  find "$RUN_DIR" -maxdepth 1 -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

show_snapshot() {
  echo "===== Processes ====="
  pgrep -af 'ui_tars_7b_papo_group_listwise_v4_retrieval_only|llamafactory|torchrun|launcher.py' || echo "No active retrieval-only process"
  echo "===== GPUs ====="
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  local log
  log="$(latest_log)"
  if [[ -n "$log" ]]; then
    echo "===== Log ====="
    echo "$log"
    python - "$log" <<'PY'
import ast
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
progress = None
train_metrics = None
eval_metrics = None

for line in reversed(lines):
    stripped = line.strip()
    if progress is None and re.search(r"\b\d+/\d+\s+\[", stripped):
        progress = stripped
    if train_metrics is None and stripped.startswith("{'loss':"):
        try:
            train_metrics = ast.literal_eval(stripped)
        except Exception:
            pass
    if eval_metrics is None and stripped.startswith("{'eval_loss':"):
        try:
            eval_metrics = ast.literal_eval(stripped)
        except Exception:
            pass
    if progress and train_metrics and eval_metrics:
        break

if progress:
    print("===== Progress =====")
    print(progress)

if train_metrics:
    print("===== Latest train metrics =====")
    for key in [
        "loss",
        "grad_norm",
        "learning_rate",
        "papo_group_loss",
        "papo_oracle_top1_accuracy",
        "papo_oracle_margin",
        "papo_target_entropy",
        "papo_policy_entropy",
        "epoch",
    ]:
        if key in train_metrics:
            print(f"{key}: {train_metrics[key]}")

if eval_metrics:
    print("===== Latest eval metrics =====")
    for key in [
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
    ]:
        if key in eval_metrics:
            print(f"{key}: {eval_metrics[key]}")
PY
    echo "===== Log tail ====="
    tail -n 40 "$log"
  else
    echo "No training log yet"
  fi
}

prepare() {
  test -f "$SOURCE_RELEASE/listwise_v4_manifest.json"
  python scripts/27_audit_proactive_listwise_v4.py \
    --release-dir "$SOURCE_RELEASE" \
    --report-dir "$REPORT_DIR/server_reaudit" \
    --image-root /home/dumike/zyy/GUI/data/raw/fingertip20k
  python scripts/28_register_proactive_listwise_v4.py \
    --release-dir "$SOURCE_RELEASE" \
    --dataset-dir "$DATASET_DIR" \
    --allow-nonformal-retrieval
  python scripts/35_preflight_papo_group_listwise_v4_retrieval_only.py \
    --training-config "$CONFIG" \
    --release-dir "$DATASET_DIR" \
    --report "$REPORT_DIR/server_preflight.json"
  python -m unittest discover -s tests -p 'test_papo_group_listwise_v4_loss.py' -v
  echo "RETRIEVAL-ONLY PREPARATION PASSED; THIS IS NOT FULL-V4"
}

train() {
  if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    prepare
  else
    echo "SKIP_PREPARE=1 -> skipping re-audit, registration, preflight and unit tests."
  fi
  local active log pid
  active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$active" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    echo "ERROR: active GPU processes detected:" >&2
    echo "$active" >&2
    exit 1
  fi
  if pgrep -af "llamafactory.*$(basename "$CONFIG")" >/dev/null; then
    echo "ERROR: retrieval-only training is already active." >&2
    exit 1
  fi
  log="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  nohup llamafactory-cli train "$CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_DIR/train.pid"
  echo "PID: $pid"
  echo "Log: $log"
  sleep 20
  tail -n 100 "$log" || true
}

status() {
  show_snapshot
}

monitor() {
  local interval="${MONITOR_INTERVAL:-60}"
  while pgrep -af "llamafactory.*$(basename "$CONFIG")" >/dev/null; do
    date
    show_snapshot
    sleep "$interval"
  done
  date
  echo "===== Training process exited; final snapshot ====="
  show_snapshot
}

report() {
  local log
  log="$(latest_log)"
  test -n "$log"
  python scripts/33_report_papo_group_listwise_v4_smoke.py \
    --training-config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --log "$log" \
    --report-dir "$REPORT_DIR" \
    --report-prefix group_listwise_v4_retrieval_only \
    --experiment-kind "PAPO Grouped Listwise-v4 Retrieval-Only" \
    --claim-boundary "Full-scale history-retrieval-only engineering experiment; no model candidates; not full-v4."
  cat "$REPORT_DIR/group_listwise_v4_retrieval_only_report.md"
}

case "$ACTION" in
  prepare) prepare ;;
  train) train ;;
  status) status ;;
  monitor) monitor ;;
  report) report ;;
  *) echo "Usage: $0 {prepare|train|status|monitor|report}" >&2; exit 2 ;;
esac
