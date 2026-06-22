from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


METRICS = (
    "loss",
    "eval_loss",
    "papo_group_loss",
    "papo_oracle_top1_accuracy",
    "papo_oracle_margin",
    "papo_target_entropy",
    "papo_policy_entropy",
    "eval_papo_group_loss",
    "eval_papo_oracle_top1_accuracy",
    "eval_papo_oracle_margin",
    "eval_papo_target_entropy",
    "eval_papo_policy_entropy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_state(output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in output_dir.glob("**/trainer_state.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((int(state.get("global_step") or 0), path, state))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
    if not candidates:
        return None, None
    _, path, state = max(candidates, key=lambda item: (item[0], str(item[1])))
    return path, state


def metric_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        row = {"step": item.get("step"), "epoch": item.get("epoch")}
        row.update({name: item[name] for name in METRICS if name in item})
        if len(row) > 2:
            rows.append(row)
    return rows


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        observed = [row for row in rows if name in row]
        if not observed:
            continue
        numeric = [float(row[name]) for row in observed]
        summary[name] = {
            "count": len(observed),
            "latest": numeric[-1],
            "latest_step": observed[-1].get("step"),
            "min": min(numeric),
            "min_step": observed[numeric.index(min(numeric))].get("step"),
            "max": max(numeric),
            "max_step": observed[numeric.index(max(numeric))].get("step"),
        }
    return summary


def scan_anomalies(log_text: str, rows: list[dict[str, Any]]) -> list[str]:
    anomalies: list[str] = []
    patterns = {
        "oom": r"out of memory|CUDA OOM",
        "nan_or_inf_log": r"(?:^|[^a-z])(nan|inf)(?:[^a-z]|$)",
        "candidate_group_misalignment": (
            r"metadata does not match candidate count|Malformed PAPO group|"
            r"probabilities must sum to one|exactly one oracle|group.*misalign"
        ),
        "traceback": r"Traceback \(most recent call last\)",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, log_text, flags=re.IGNORECASE | re.MULTILINE):
            anomalies.append(name)
    for row in rows:
        for name in METRICS:
            if name in row and not math.isfinite(float(row[name])):
                anomalies.append(f"nonfinite_metric:{name}:step={row.get('step')}")
    return sorted(set(anomalies))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a retrieval-only PAPO grouped Listwise-v4 smoke run.")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.training_config.resolve()
    output_dir = args.output_dir.resolve()
    log_path = args.log.resolve()
    state_path, state = find_state(output_dir)
    history = list((state or {}).get("log_history") or [])
    rows = metric_rows(history)
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    anomalies = scan_anomalies(log_text, rows)
    completed = any("train_runtime" in item for item in history)
    status = "completed" if completed and not anomalies else "incomplete_or_anomalous"
    report = {
        "status": status,
        "experiment_kind": "retrieval-only smoke experiment",
        "formal_full_v4_complete": False,
        "claim_boundary": "engineering pre-experiment only; not full-v4 and not a formal effect claim",
        "training_config": str(config_path),
        "training_config_sha256": sha256_file(config_path),
        "output_dir": str(output_dir),
        "log": str(log_path),
        "trainer_state": str(state_path) if state_path else None,
        "global_step": (state or {}).get("global_step"),
        "epoch": (state or {}).get("epoch"),
        "completed": completed,
        "metrics": summarize_metrics(rows),
        "anomalies": anomalies,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "group_listwise_v4_smoke_report.json"
    md_path = args.report_dir / "group_listwise_v4_smoke_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# PAPO Grouped Listwise-v4 Retrieval Smoke",
        "",
        "> Engineering pre-experiment only. This is not full-v4 and cannot support a formal effect claim.",
        "",
        f"- Status: `{status}`",
        f"- Completed: `{completed}`",
        f"- Global step: `{report['global_step']}`",
        f"- Epoch: `{report['epoch']}`",
        f"- Anomalies: `{', '.join(anomalies) if anomalies else 'none detected'}`",
        "",
        "| metric | count | latest | latest_step | min | min_step | max | max_step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["metrics"].items():
        lines.append(
            f"| {name} | {values['count']} | {values['latest']:.6f} | {values['latest_step']} | "
            f"{values['min']:.6f} | {values['min_step']} | {values['max']:.6f} | {values['max_step']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
