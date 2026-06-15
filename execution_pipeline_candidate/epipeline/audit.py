from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import manifest_identity_matches, read_csv, read_json, read_jsonl, sha256_file, write_json


def audit_manifest(manifest: dict[str, Any], *, require_paper_eligible: bool) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not manifest_identity_matches(manifest):
        issues.append({"issue": "manifest_identity_missing_or_changed"})
    binding = manifest.get("official_reference_audit")
    if require_paper_eligible:
        if not isinstance(binding, dict) or not binding:
            issues.append({"issue": "official_reference_audit_not_bound"})
        elif binding.get("source_only"):
            issues.append({"issue": "official_reference_audit_is_source_only"})
        else:
            binding_path = Path(str(binding.get("path") or ""))
            if not binding_path.is_file() or sha256_file(binding_path) != binding.get("sha256"):
                issues.append({"issue": "official_reference_audit_missing_or_changed"})
    runs: list[dict[str, Any]] = []
    for entry in manifest["runs"]:
        run_id = str(entry["id"])
        run_dir = Path(str(entry["run_dir"]))
        tasks_path = Path(str(entry["tasks_path"]))
        if not tasks_path.exists() or sha256_file(tasks_path) != entry["tasks_sha256"]:
            issues.append({"run_id": run_id, "issue": "task_file_missing_or_changed"})
            continue
        expected = {str(task["task_id"]) for task in read_jsonl(tasks_path)}
        raw_path = run_dir / "raw_results.jsonl"
        score_path = run_dir / "score_report.json"
        if not raw_path.exists():
            issues.append({"run_id": run_id, "issue": "missing_raw_results"})
            continue
        raw = read_jsonl(raw_path)
        run_report_path = run_dir / "run_report.json"
        run_report = read_json(run_report_path) if run_report_path.exists() else {}
        if not run_report_path.exists() or run_report.get("raw_results_sha256") != sha256_file(raw_path):
            issues.append({"run_id": run_id, "issue": "raw_result_hash_missing_or_changed"})
        raw_ids = [str(row.get("task_id") or "") for row in raw]
        actual = set(raw_ids)
        if len(raw_ids) != len(actual):
            issues.append({"run_id": run_id, "issue": "duplicate_raw_task_ids"})
        if actual != expected:
            issues.append(
                {
                    "run_id": run_id,
                    "issue": "raw_result_coverage_mismatch",
                    "missing": len(expected - actual),
                    "extra": len(actual - expected),
                }
            )
        if int(run_report.get("failed") or 0) > 0:
            issues.append(
                {
                    "run_id": run_id,
                    "issue": "retryable_failures_present",
                    "failed": int(run_report["failed"]),
                }
            )
        score = read_json(score_path) if score_path.exists() else {}
        if not score_path.exists():
            issues.append({"run_id": run_id, "issue": "missing_score_report"})
        else:
            scored_path = run_dir / "execution_results_scored.csv"
            if score.get("raw_results_sha256") != sha256_file(raw_path):
                issues.append({"run_id": run_id, "issue": "score_not_bound_to_raw_results"})
            if not scored_path.exists() or score.get("scored_csv_sha256") != sha256_file(scored_path):
                issues.append({"run_id": run_id, "issue": "scored_csv_missing_or_changed"})
            annotations_path = str(score.get("annotations_path") or "")
            annotations_hash = str(score.get("annotations_sha256") or "")
            if annotations_path and (
                not Path(annotations_path).exists()
                or sha256_file(annotations_path) != annotations_hash
            ):
                issues.append({"run_id": run_id, "issue": "annotation_file_missing_or_changed"})
        if require_paper_eligible and not score.get("paper_eligible"):
            issues.append({"run_id": run_id, "issue": "run_not_paper_eligible"})
        runs.append(
            {
                "run_id": run_id,
                "required": bool(entry.get("required")),
                "expected_tasks": len(expected),
                "raw_results": len(raw),
                "verified_success_rows": score.get("verified_success_rows"),
                "paper_eligible": score.get("paper_eligible", False),
                "retryable_failures": run_report.get("failed"),
            }
        )
    entry_by_id = {str(entry["id"]): entry for entry in manifest["runs"]}
    for comparison in manifest.get("comparisons") or []:
        reference_id = str(comparison["reference_run"])
        candidate_id = str(comparison["candidate_run"])
        reference_path = Path(str(entry_by_id[reference_id]["run_dir"])) / "execution_results_scored.csv"
        candidate_path = Path(str(entry_by_id[candidate_id]["run_dir"])) / "execution_results_scored.csv"
        if not reference_path.exists() or not candidate_path.exists():
            issues.append({"comparison_id": comparison["id"], "issue": "missing_comparison_score_file"})
            continue
        reference_rows = read_csv(reference_path)
        candidate_rows = read_csv(candidate_path)
        reference_list = [str(row.get("task_id") or "") for row in reference_rows]
        candidate_list = [str(row.get("task_id") or "") for row in candidate_rows]
        reference_ids = set(reference_list)
        candidate_ids = set(candidate_list)
        if len(reference_list) != len(reference_ids) or len(candidate_list) != len(candidate_ids):
            issues.append({"comparison_id": comparison["id"], "issue": "duplicate_comparison_task_ids"})
        if reference_ids != candidate_ids:
            issues.append(
                {
                    "comparison_id": comparison["id"],
                    "issue": "comparison_coverage_mismatch",
                    "reference_only": len(reference_ids - candidate_ids),
                    "candidate_only": len(candidate_ids - reference_ids),
                }
            )
    report = {
        "status": "passed" if not issues else "failed",
        "require_paper_eligible": require_paper_eligible,
        "runs": runs,
        "issues": issues,
    }
    write_json(Path(str(manifest["output_root"])) / "experiment_audit.json", report)
    return report
