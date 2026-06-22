from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import (  # noqa: E402
    V4ValidationError,
    build_groups,
    build_retrieval_candidate_pools,
    load_candidate_map,
    read_jsonl,
    retrieval_pool_map,
    sha256_file,
    write_json,
    write_jsonl,
)
from papo.proactive_quality_gate_v4 import audit_v4_groups, write_quality_outputs  # noqa: E402


def _validate_imported_candidates(
    *,
    split: str,
    task_path: Path,
    tasks: list[dict],
    candidate_path: Path,
    import_manifest: dict,
) -> tuple[dict[str, list[str]], dict]:
    report = import_manifest.get(split)
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise V4ValidationError(f"Missing passed candidate import report for {split}.")
    expected_ids = {str(task.get("task_id") or "") for task in tasks}
    candidate_map, provenance = load_candidate_map(candidate_path)
    if set(candidate_map) != expected_ids:
        missing = sorted(expected_ids - set(candidate_map))
        extra = sorted(set(candidate_map) - expected_ids)
        raise V4ValidationError(
            f"Imported {split} candidate coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(not values for values in candidate_map.values()):
        raise V4ValidationError(f"Imported {split} candidates contain an empty candidate list.")
    if any(not isinstance(value, dict) for value in provenance.values()):
        raise V4ValidationError(f"Imported {split} candidates contain missing provenance.")
    checks = {
        "task_file_sha256": sha256_file(task_path),
        "output_sha256": sha256_file(candidate_path),
        "task_count": len(tasks),
    }
    for field, actual in checks.items():
        if report.get(field) != actual:
            raise V4ValidationError(
                f"Candidate import report mismatch for {split}.{field}: expected={actual!r}, got={report.get(field)!r}"
            )
    if Path(str(report.get("output") or "")).resolve() != candidate_path.resolve():
        raise V4ValidationError(f"Candidate import output path mismatch for {split}.")
    return candidate_map, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-backed PAPO Listwise-v4 draft groups for local review without publishing a release."
    )
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--eval-candidates", type=Path, required=True)
    parser.add_argument("--candidate-import-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--allow-unavailable-images", action="store_true")
    parser.add_argument("--min-same-user-similarity", type=float, default=0.20)
    parser.add_argument("--positive-same-user-similarity", type=float, default=0.35)
    parser.add_argument("--max-context-similarity", type=float, default=0.75)
    parser.add_argument("--max-global-candidate-frequency", type=int)
    args = parser.parse_args()

    try:
        source_manifest_path = args.workspace / "manifests" / "source_task_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8", errors="strict"))
        import_manifest = json.loads(args.candidate_import_manifest.read_text(encoding="utf-8", errors="strict"))
        train_tasks, eval_tasks = read_jsonl(args.train_tasks), read_jsonl(args.eval_tasks)
        for split, task_path, tasks in (
            ("train", args.train_tasks, train_tasks),
            ("eval", args.eval_tasks, eval_tasks),
        ):
            if source_manifest.get(split, {}).get("sha256") != sha256_file(task_path):
                raise V4ValidationError(f"{split} task SHA256 differs from source audit manifest.")
            if source_manifest.get(split, {}).get("line_count") != len(tasks):
                raise V4ValidationError(f"{split} task count differs from source audit manifest.")

        train_map, train_provenance = _validate_imported_candidates(
            split="train",
            task_path=args.train_tasks,
            tasks=train_tasks,
            candidate_path=args.train_candidates,
            import_manifest=import_manifest,
        )
        eval_map, eval_provenance = _validate_imported_candidates(
            split="eval",
            task_path=args.eval_tasks,
            tasks=eval_tasks,
            candidate_path=args.eval_candidates,
            import_manifest=import_manifest,
        )

        retrieval_kwargs = {
            "min_same_user_similarity": args.min_same_user_similarity,
            "positive_same_user_similarity": args.positive_same_user_similarity,
            "max_context_similarity": args.max_context_similarity,
            "max_global_text_frequency": args.max_global_candidate_frequency,
        }
        train_pool_rows = build_retrieval_candidate_pools(
            train_tasks, train_tasks, split="train", **retrieval_kwargs
        )
        eval_pool_rows = build_retrieval_candidate_pools(
            eval_tasks, train_tasks, split="eval", **retrieval_kwargs
        )
        train_groups = build_groups(
            train_tasks,
            split="train",
            model_candidates=train_map,
            retrieval_candidates=retrieval_pool_map(train_pool_rows),
            synthetic_smoke=False,
        )
        eval_groups = build_groups(
            eval_tasks,
            split="eval",
            model_candidates=eval_map,
            retrieval_candidates=retrieval_pool_map(eval_pool_rows),
            synthetic_smoke=False,
        )

        draft_dir = args.workspace / "intermediate" / "draft_v4"
        report_dir = args.workspace / "reports" / "draft_v4"
        train_group_path = draft_dir / "papo_proactive_train_listwise_v4.draft.groups.json"
        eval_group_path = draft_dir / "papo_proactive_eval_listwise_v4.draft.groups.json"
        train_pool_path = draft_dir / "retrieval_candidate_pool_train_draft_v4.jsonl"
        eval_pool_path = draft_dir / "retrieval_candidate_pool_eval_draft_v4.jsonl"
        write_json(train_group_path, train_groups)
        write_json(eval_group_path, eval_groups)
        write_jsonl(train_pool_path, train_pool_rows)
        write_jsonl(eval_pool_path, eval_pool_rows)

        quality, issues = audit_v4_groups(
            train_groups,
            eval_groups,
            image_roots=args.image_root,
            allow_unavailable_images=args.allow_unavailable_images,
            source_manifest=source_manifest,
        )
        quality_paths = write_quality_outputs(quality, issues, report_dir)
        if quality.get("status") == "failed":
            raise V4ValidationError("Draft quality gate failed; inspect reports/draft_v4 before manual review.")

        manifest = {
            "schema_version": "papo_listwise_v4_draft_manifest",
            "status": "passed_for_manual_review",
            "formal_release": False,
            "requires_manual_review": True,
            "source_task_manifest_sha256": sha256_file(source_manifest_path),
            "candidate_import_manifest_sha256": sha256_file(args.candidate_import_manifest),
            "candidate_inputs": {
                "train": {
                    "path": str(args.train_candidates.resolve()),
                    "sha256": sha256_file(args.train_candidates),
                    "provenance_task_count": len(train_provenance),
                },
                "eval": {
                    "path": str(args.eval_candidates.resolve()),
                    "sha256": sha256_file(args.eval_candidates),
                    "provenance_task_count": len(eval_provenance),
                },
            },
            "group_outputs": {
                "train": {"path": str(train_group_path.resolve()), "sha256": sha256_file(train_group_path)},
                "eval": {"path": str(eval_group_path.resolve()), "sha256": sha256_file(eval_group_path)},
            },
            "group_counts": {"train": len(train_groups), "eval": len(eval_groups)},
            "quality": {"status": quality.get("status"), "reports": quality_paths},
        }
        manifest_path = args.workspace / "manifests" / "listwise_v4_draft_manifest.json"
        write_json(manifest_path, manifest)
    except (OSError, ValueError, V4ValidationError) as error:
        print(f"LISTWISE-V4 DRAFT BUILD FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps({**manifest, "manifest": str(manifest_path.resolve())}, ensure_ascii=False, indent=2))
    print("LISTWISE-V4 DRAFT BUILT FOR MANUAL REVIEW; NO RELEASE WAS PUBLISHED")


if __name__ == "__main__":
    main()
