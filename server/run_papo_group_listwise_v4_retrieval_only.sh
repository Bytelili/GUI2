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
CHECKPOINT_EVAL_ROOT="$REPORT_DIR/checkpoint_eval"
ACTION="${1:-status}"
PROBE_CONFIG="$RUN_DIR/eval_probe.yaml"
PROBE_REPORT_DIR="$REPORT_DIR/eval_batch_probe"

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

probe_eval() {
  if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    prepare
  else
    echo "SKIP_PREPARE=1 -> skipping re-audit, registration, preflight and unit tests."
  fi
  local active log pid probe_output probe_logging
  active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$active" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    echo "ERROR: active GPU processes detected:" >&2
    echo "$active" >&2
    exit 1
  fi
  probe_output="$OUTPUT/__eval_probe__"
  probe_logging="$RUN_DIR/__eval_probe__"
  python - "$CONFIG" "$PROBE_CONFIG" "$probe_output" "$probe_logging" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
probe_output = sys.argv[3]
probe_logging = sys.argv[4]

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.update({
    "output_dir": probe_output,
    "logging_dir": probe_logging,
    "max_steps": 1,
    "eval_on_start": True,
    "eval_strategy": "steps",
    "eval_steps": 1,
    "save_strategy": "steps",
    "save_steps": 999999,
    "load_best_model_at_end": False,
})
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Wrote probe config: {dst}")
print(f"per_device_eval_batch_size: {cfg.get('per_device_eval_batch_size')}")
print(f"eval_on_start: {cfg.get('eval_on_start')}")
print(f"max_steps: {cfg.get('max_steps')}")
PY
  log="$RUN_DIR/eval_probe_$(date +%Y%m%d_%H%M%S).log"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  nohup llamafactory-cli train "$PROBE_CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$RUN_DIR/eval_probe.pid"
  echo "PID: $pid"
  echo "Log: $log"
  sleep 20
  tail -n 120 "$log" || true
}

probe_eval_grid() {
  if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    prepare
  else
    echo "SKIP_PREPARE=1 -> skipping re-audit, registration, preflight and unit tests."
  fi
  local active candidates batch log probe_output probe_logging probe_cfg
  active="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$active" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    echo "ERROR: active GPU processes detected:" >&2
    echo "$active" >&2
    exit 1
  fi
  candidates="${EVAL_BATCH_CANDIDATES:-1 2 4}"
  mkdir -p "$PROBE_REPORT_DIR"
  : > "$PROBE_REPORT_DIR/probe_runs.jsonl"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  for batch in $candidates; do
    echo "===== Eval batch probe: $batch ====="
    probe_cfg="$RUN_DIR/eval_probe_b${batch}.yaml"
    probe_output="$OUTPUT/__eval_probe_b${batch}__"
    probe_logging="$RUN_DIR/__eval_probe_b${batch}__"
    rm -rf "$probe_output" "$probe_logging"
    python - "$CONFIG" "$probe_cfg" "$probe_output" "$probe_logging" "$batch" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
probe_output = sys.argv[3]
probe_logging = sys.argv[4]
batch = int(sys.argv[5])

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.update({
    "output_dir": probe_output,
    "logging_dir": probe_logging,
    "per_device_eval_batch_size": batch,
    "max_steps": 1,
    "eval_on_start": True,
    "eval_strategy": "steps",
    "eval_steps": 1,
    "save_strategy": "steps",
    "save_steps": 999999,
    "load_best_model_at_end": False,
    "plot_loss": False,
})
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Wrote probe config: {dst}")
print(f"per_device_eval_batch_size: {cfg.get('per_device_eval_batch_size')}")
print(f"eval_on_start: {cfg.get('eval_on_start')}")
print(f"max_steps: {cfg.get('max_steps')}")
PY
    log="$RUN_DIR/eval_probe_b${batch}_$(date +%Y%m%d_%H%M%S).log"
    set +e
    llamafactory-cli train "$probe_cfg" >"$log" 2>&1
    status=$?
    set -e
    python - "$batch" "$status" "$log" "$probe_output" >> "$PROBE_REPORT_DIR/probe_runs.jsonl" <<'PY'
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

batch = int(sys.argv[1])
exit_code = int(sys.argv[2])
log_path = Path(sys.argv[3])
output_dir = Path(sys.argv[4])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

result = {
    "batch": batch,
    "exit_code": exit_code,
    "log": str(log_path),
    "output_dir": str(output_dir),
    "status": "failed" if exit_code != 0 else "ok",
    "oom": bool(re.search(r"out of memory|CUDA OOM", text, flags=re.IGNORECASE)),
    "traceback": "Traceback (most recent call last)" in text,
}

trainer_states = sorted(output_dir.glob("**/trainer_state.json"))
state = None
if trainer_states:
    try:
        state = json.loads(trainer_states[-1].read_text(encoding="utf-8"))
    except Exception:
        state = None

metrics = None
if state:
    for item in reversed(list(state.get("log_history") or [])):
        if "eval_loss" in item:
            metrics = item
            break

if metrics is None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{'eval_loss':"):
            try:
                metrics = ast.literal_eval(line)
                break
            except Exception:
                pass

if metrics:
    result["eval_loss"] = metrics.get("eval_loss")
    result["eval_runtime"] = metrics.get("eval_runtime")
    result["eval_samples_per_second"] = metrics.get("eval_samples_per_second")
    result["eval_steps_per_second"] = metrics.get("eval_steps_per_second")

if result["oom"] or result["traceback"] or exit_code != 0:
    result["status"] = "failed"
elif metrics:
    result["status"] = "passed"
else:
    result["status"] = "incomplete"

print(json.dumps(result, ensure_ascii=False))
PY
    tail -n 60 "$log" || true
  done
  python - "$PROBE_REPORT_DIR/probe_runs.jsonl" "$PROBE_REPORT_DIR" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
runs = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
passed = [run for run in runs if run.get("status") == "passed" and run.get("eval_samples_per_second") is not None]
best = None
if passed:
    best = max(
        passed,
        key=lambda run: (
            float(run.get("eval_samples_per_second") or 0.0),
            int(run.get("batch") or 0),
        ),
    )
report = {
    "status": "passed" if best else "failed",
    "candidates": runs,
    "recommended_batch": best.get("batch") if best else None,
    "recommended_by": "highest eval_samples_per_second among successful runs; ties broken by larger batch",
    "recommended_metrics": best,
}
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "probe_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# Eval Batch Probe Summary",
    "",
    f"- Recommended batch: `{report['recommended_batch']}`",
    f"- Rule: {report['recommended_by']}",
    "",
    "| batch | status | exit_code | oom | eval_runtime | eval_samples_per_second | eval_steps_per_second | eval_loss |",
    "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
]
for run in runs:
    lines.append(
        f"| {run.get('batch')} | {run.get('status')} | {run.get('exit_code')} | {run.get('oom')} | "
        f"{run.get('eval_runtime')} | {run.get('eval_samples_per_second')} | "
        f"{run.get('eval_steps_per_second')} | {run.get('eval_loss')} |"
    )
(report_dir / "probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Written: {report_dir / 'probe_summary.json'}")
print(f"Written: {report_dir / 'probe_summary.md'}")
PY
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

eval_checkpoints() {
  local checkpoints=()
  local labels=()
  local checkpoint label step summary_dir

  mapfile -t checkpoints < <(
    {
      find "$OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
      if [[ -f "$OUTPUT/adapter_model.safetensors" ]]; then
        printf '%s\n' "$OUTPUT"
      fi
    } | awk '!seen[$0]++'
  )

  if [[ -n "${CHECKPOINT_STEPS:-}" ]]; then
    local filtered=()
    for checkpoint in "${checkpoints[@]}"; do
      step="$(basename "$checkpoint")"
      step="${step#checkpoint-}"
      if [[ "$checkpoint" == "$OUTPUT" ]]; then
        step="final"
      fi
      for wanted in ${CHECKPOINT_STEPS}; do
        if [[ "$step" == "$wanted" ]]; then
          filtered+=("$checkpoint")
          break
        fi
      done
    done
    checkpoints=("${filtered[@]}")
  fi

  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No preserved checkpoints were found under $OUTPUT" >&2
    exit 1
  fi

  mkdir -p "$CHECKPOINT_EVAL_ROOT"
  for checkpoint in "${checkpoints[@]}"; do
    if [[ "$checkpoint" == "$OUTPUT" ]]; then
      step="final"
    else
      step="$(basename "$checkpoint")"
      step="${step#checkpoint-}"
    fi
    label="ui_tars_7b_papo_group_listwise_v4_retrieval_only_ckpt_${step}"
    labels+=("$label")
    echo "===== Prepare checkpoint provenance: $checkpoint ====="
    python scripts/36_prepare_proactive_checkpoint_provenance.py \
      --config config.yaml \
      --training-config "$CONFIG" \
      --checkpoint-dir "$checkpoint"
    echo "===== Evaluate checkpoint: $label ====="
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    MODE="${MODE:-strict_holdout}" \
    LEVELS="${LEVELS:-0,1,2,3}" \
    NUM_SHARDS="${NUM_SHARDS:-4}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}" \
    EVAL_MODEL_LABEL="$label" \
    EVAL_ADAPTER="$checkpoint" \
    REPORT_ROOT="$CHECKPOINT_EVAL_ROOT" \
    bash ui_tars_proactive/run_ui_tars_7b.sh eval_adapter
  done

  summary_dir="$CHECKPOINT_EVAL_ROOT/summary"
  python ui_tars_proactive/summarize_results.py \
    --reports-root "$CHECKPOINT_EVAL_ROOT" \
    --mode "${MODE:-strict_holdout}" \
    --models "${labels[@]}" \
    --output-dir "$summary_dir"
  cat "$summary_dir/${MODE:-strict_holdout}_ui_tars_level_results.md"
}

case "$ACTION" in
  prepare) prepare ;;
  train) train ;;
  probe_eval) probe_eval ;;
  probe_eval_grid) probe_eval_grid ;;
  status) status ;;
  monitor) monitor ;;
  report) report ;;
  eval_checkpoints) eval_checkpoints ;;
  *) echo "Usage: $0 {prepare|train|probe_eval|probe_eval_grid|status|monitor|report|eval_checkpoints}" >&2; exit 2 ;;
esac
