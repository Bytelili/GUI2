from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .actions import levenshtein_similarity, official_execution_similarity
from .io_utils import manifest_identity_matches, read_csv, read_json, read_jsonl, sha256_file, write_csv, write_json


def score_manifest(manifest: dict[str, Any], annotations_path: str | Path | None = None) -> dict[str, Any]:
    if not manifest_identity_matches(manifest):
        raise ValueError("Experiment manifest identity is missing or changed")
    annotations = load_annotations(annotations_path)
    annotation_path = str(Path(annotations_path).resolve()) if annotations_path else ""
    annotation_sha256 = sha256_file(annotation_path) if annotation_path else ""
    valid_annotation_keys = {
        (str(entry["id"]), str(task["task_id"]))
        for entry in manifest["runs"]
        for task in read_jsonl(entry["tasks_path"])
    }
    unknown_annotation_keys = sorted(set(annotations) - valid_annotation_keys)
    if unknown_annotation_keys:
        raise ValueError(f"Manual annotations reference unknown run/task keys: {unknown_annotation_keys[:10]}")
    reports: list[dict[str, Any]] = []
    for entry in manifest["runs"]:
        run_dir = Path(str(entry["run_dir"]))
        raw_path = run_dir / "raw_results.jsonl"
        rows = read_jsonl(raw_path)
        identifiers = [str(row.get("task_id") or "") for row in rows]
        if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Raw execution results contain duplicate or empty task IDs: {entry['id']}")
        expected = {str(task["task_id"]) for task in read_jsonl(entry["tasks_path"])}
        if set(identifiers) != expected:
            raise ValueError(f"Raw execution result coverage does not match prepared tasks: {entry['id']}")
        run_report_path = run_dir / "run_report.json"
        run_report = read_json(run_report_path) if run_report_path.exists() else {}
        raw_sha256 = sha256_file(raw_path)
        if run_report.get("raw_results_sha256") != raw_sha256:
            raise ValueError(f"Raw execution result hash does not match run report: {entry['id']}")
        scored = [score_result(row, annotations.get((str(entry["id"]), str(row["task_id"])))) for row in rows]
        output = run_dir / "execution_results_scored.csv"
        write_csv(output, scored)
        summary = summarize_scored(str(entry["id"]), scored)
        summary["model_paper_eligible"] = bool(run_report.get("model_paper_eligible"))
        summary["device_paper_eligible"] = bool(run_report.get("device_paper_eligible"))
        summary["paper_eligible"] = bool(
            summary["paper_eligible"]
            and summary["model_paper_eligible"]
            and summary["device_paper_eligible"]
        )
        summary["raw_results_sha256"] = raw_sha256
        summary["annotations_path"] = annotation_path
        summary["annotations_sha256"] = annotation_sha256
        summary["scored_csv"] = str(output)
        summary["scored_csv_sha256"] = sha256_file(output)
        write_json(run_dir / "score_report.json", summary)
        reports.append(summary)
    report = {"status": "completed", "runs": reports}
    write_json(Path(str(manifest["output_root"])) / "score_summary.json", report)
    return report


def score_result(row: dict[str, Any], annotation: dict[str, str] | None) -> dict[str, Any]:
    success = row.get("success")
    verified = bool(row.get("success_verified"))
    source = str(row.get("success_source") or "")
    evidence = str(row.get("success_evidence") or "")
    if annotation and str(annotation.get("success") or "").strip():
        if not str(annotation.get("annotator") or "").strip() or not str(annotation.get("evidence") or "").strip():
            raise ValueError(f"Manual success annotation requires annotator and evidence: {row['task_id']}")
        success = parse_bool(annotation["success"])
        verified = True
        source = f"manual:{annotation['annotator']}"
        evidence = str(annotation["evidence"])
    agent = [str(value) for value in row.get("agent_actions") or []]
    official_agent = [str(value) for value in row.get("official_agent_outputs") or agent]
    golden = [str(value) for value in row.get("golden_actions") or []]
    cross = [str(value) for value in row.get("cross_user_actions") or []]
    up_sim, down_sim, similarity = official_execution_similarity(official_agent, golden, cross)
    strict_up_sim = levenshtein_similarity(agent, golden)
    strict_down_sim = levenshtein_similarity(agent, cross) if cross else 0.0
    strict_similarity = strict_up_sim / max(strict_down_sim, 1e-8)
    over_limit = len(official_agent) > 2.5 * max(len(golden), 1)
    if verified and success and over_limit:
        success = False
        evidence = f"{evidence}; forced failure: exceeded 2.5x golden steps".strip("; ")
    final_observation = row.get("final_observation") if isinstance(row.get("final_observation"), dict) else {}
    return {
        "task_id": row.get("task_id", ""),
        "episode_id": row.get("episode_id", ""),
        "user_id": row.get("user_id", ""),
        "app": row.get("app", ""),
        "screen": final_observation.get("screenshot", ""),
        "intent": row.get("intent", ""),
        "run_id": row.get("run_id", ""),
        "model_id": row.get("model_id", ""),
        "condition": row.get("condition", ""),
        "success": int(bool(success)) if verified else "",
        "success_verified": verified,
        "success_source": source,
        "success_evidence": evidence,
        "origin_step": len(golden),
        "real_step": len(official_agent),
        "step_ratio": len(official_agent) / max(len(golden), 1),
        "up_sim": up_sim,
        "down_sim": down_sim,
        "similarity": similarity,
        "metric_protocol": "FingerTip-20K personalized_execution.py official fuzzy-text formula",
        "strict_action_up_sim": strict_up_sim,
        "strict_action_down_sim": strict_down_sim,
        "strict_action_similarity": strict_similarity,
        "time": float(row.get("time") or 0.0),
        "token": int(row.get("token") or 0),
        "termination_reason": row.get("termination_reason", ""),
        "invalid_actions": int(row.get("invalid_actions") or 0),
        "invalid_action_policy": row.get("invalid_action_policy", ""),
        "official_agent_outputs": json.dumps(official_agent, ensure_ascii=False),
        "agent_actions": json.dumps(agent, ensure_ascii=False),
        "golden_actions": json.dumps(golden, ensure_ascii=False),
        "cross_user_actions": json.dumps(cross, ensure_ascii=False),
    }


def summarize_scored(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [row for row in rows if row["success_verified"]]
    return {
        "run_id": run_id,
        "rows": len(rows),
        "transient_failure_rows": sum(row.get("termination_reason") in {"model_error", "runner_error"} for row in rows),
        "verified_success_rows": len(verified),
        "paper_eligible": (
            bool(rows)
            and len(verified) == len(rows)
            and not any(row.get("termination_reason") in {"model_error", "runner_error"} for row in rows)
        ),
        "success_rate": average([float(row["success"]) for row in verified]),
        "up_sim": average([float(row["up_sim"]) for row in rows]),
        "down_sim": average([float(row["down_sim"]) for row in rows]),
        "similarity": average([float(row["similarity"]) for row in rows]),
        "strict_action_up_sim": average([float(row["strict_action_up_sim"]) for row in rows]),
        "strict_action_down_sim": average([float(row["strict_action_down_sim"]) for row in rows]),
        "strict_action_similarity": average([float(row["strict_action_similarity"]) for row in rows]),
        "step_ratio": average([float(row["step_ratio"]) for row in rows]),
        "time": average([float(row["time"]) for row in rows]),
        "token": average([float(row["token"]) for row in rows]),
    }


def load_annotations(path: str | Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    rows = read_csv(path)
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("run_id") or ""), str(row.get("task_id") or ""))
        if key in output:
            raise ValueError(f"Duplicate manual annotation: {key}")
        output[key] = row
    return output


def parse_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "success", "passed"}:
        return True
    if text in {"0", "false", "no", "failed", "failure"}:
        return False
    raise ValueError(f"Invalid success annotation: {value!r}")


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
