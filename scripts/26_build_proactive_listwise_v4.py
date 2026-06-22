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
    build_release,
    load_candidate_map,
    read_jsonl,
    retrieval_pool_map,
    sha256_file,
    sha256_json,
    stratified_tasks,
    write_json,
    write_jsonl,
)
from papo.proactive_quality_gate_v4 import audit_v4_groups, write_quality_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated PAPO grouped Listwise-v4 smoke or formal release.")
    parser.add_argument("--train-tasks", type=Path, required=True)
    parser.add_argument("--eval-tasks", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--release-kind", choices=("smoke_v4", "full_v4"), required=True)
    parser.add_argument("--train-candidates", type=Path)
    parser.add_argument("--eval-candidates", type=Path)
    parser.add_argument("--train-reviewed-groups", type=Path)
    parser.add_argument("--eval-reviewed-groups", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--require-local-images", action="store_true")
    parser.add_argument("--min-manual-review-fraction", type=float)
    parser.add_argument("--min-same-user-similarity", type=float, default=0.20)
    parser.add_argument("--positive-same-user-similarity", type=float, default=0.35)
    parser.add_argument("--max-context-similarity", type=float, default=0.75)
    parser.add_argument("--max-global-candidate-frequency", type=int)
    args = parser.parse_args()
    try:
        source_manifest_path = args.workspace / "manifests" / "source_task_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8", errors="strict"))
        if source_manifest.get("train", {}).get("sha256") != sha256_file(args.train_tasks):
            raise V4ValidationError("train task SHA256 differs from source audit manifest")
        if source_manifest.get("eval", {}).get("sha256") != sha256_file(args.eval_tasks):
            raise V4ValidationError("eval task SHA256 differs from source audit manifest")
        if args.release_kind == "full_v4" and args.synthetic_smoke:
            raise V4ValidationError("--synthetic-smoke cannot be combined with --release-kind full_v4")
        if args.release_kind == "full_v4" and (not args.train_candidates or not args.eval_candidates):
            raise V4ValidationError("full_v4 requires imported train and eval UI-TARS candidates")
        if args.release_kind == "full_v4" and (not args.train_reviewed_groups or not args.eval_reviewed_groups):
            raise V4ValidationError("full_v4 requires manually reviewed train and eval group files")
        if args.release_kind == "full_v4" and args.min_manual_review_fraction is None:
            raise V4ValidationError("full_v4 requires an explicit --min-manual-review-fraction")
        if not args.synthetic_smoke and (not args.train_candidates or not args.eval_candidates):
            raise V4ValidationError("missing formal candidates; use --synthetic-smoke only for smoke_v4")

        all_train_tasks = read_jsonl(args.train_tasks)
        all_eval_tasks = read_jsonl(args.eval_tasks)
        train_tasks = stratified_tasks(all_train_tasks, args.train_limit if args.release_kind == "smoke_v4" else 0, args.seed)
        eval_tasks = stratified_tasks(all_eval_tasks, args.eval_limit if args.release_kind == "smoke_v4" else 0, args.seed + 1)
        train_pool_rows = build_retrieval_candidate_pools(
            train_tasks,
            all_train_tasks,
            split="train",
            min_same_user_similarity=args.min_same_user_similarity,
            positive_same_user_similarity=args.positive_same_user_similarity,
            max_context_similarity=args.max_context_similarity,
            max_global_text_frequency=args.max_global_candidate_frequency,
        )
        eval_pool_rows = build_retrieval_candidate_pools(
            eval_tasks,
            all_train_tasks,
            split="eval",
            min_same_user_similarity=args.min_same_user_similarity,
            positive_same_user_similarity=args.positive_same_user_similarity,
            max_context_similarity=args.max_context_similarity,
            max_global_text_frequency=args.max_global_candidate_frequency,
        )
        train_retrieval = retrieval_pool_map(train_pool_rows)
        eval_retrieval = retrieval_pool_map(eval_pool_rows)
        train_map = eval_map = None
        candidate_provenance = None
        if args.train_candidates and args.eval_candidates:
            train_map, train_provenance = load_candidate_map(args.train_candidates)
            eval_map, eval_provenance = load_candidate_map(args.eval_candidates)
            candidate_provenance = {
                "train_file_sha256": sha256_file(args.train_candidates),
                "eval_file_sha256": sha256_file(args.eval_candidates),
                "train_task_count": len(train_provenance),
                "eval_task_count": len(eval_provenance),
                "train_provenance_sha256": sha256_json(train_provenance),
                "eval_provenance_sha256": sha256_json(eval_provenance),
                "train_provenance_sample": next(iter(train_provenance.values()), None),
                "eval_provenance_sample": next(iter(eval_provenance.values()), None),
            }
        train_groups = build_groups(
            train_tasks,
            split="train",
            model_candidates=train_map,
            retrieval_candidates=train_retrieval,
            synthetic_smoke=args.synthetic_smoke,
        )
        eval_groups = build_groups(
            eval_tasks,
            split="eval",
            model_candidates=eval_map,
            retrieval_candidates=eval_retrieval,
            synthetic_smoke=args.synthetic_smoke,
        )
        if args.train_reviewed_groups and args.eval_reviewed_groups:
            train_groups = json.loads(args.train_reviewed_groups.read_text(encoding="utf-8", errors="strict"))
            eval_groups = json.loads(args.eval_reviewed_groups.read_text(encoding="utf-8", errors="strict"))
            if {group.get("task_id") for group in train_groups} != {task.get("task_id") for task in train_tasks}:
                raise V4ValidationError("reviewed train groups do not exactly cover the selected strict tasks")
            if {group.get("task_id") for group in eval_groups} != {task.get("task_id") for task in eval_tasks}:
                raise V4ValidationError("reviewed eval groups do not exactly cover the selected strict tasks")
        intermediate = args.workspace / "intermediate"
        write_jsonl(intermediate / "retrieval_candidate_pool_train_selected_v4.jsonl", train_pool_rows)
        write_jsonl(intermediate / "retrieval_candidate_pool_eval_selected_v4.jsonl", eval_pool_rows)
        write_json(intermediate / "papo_proactive_train_listwise_v4.groups.json", train_groups)
        write_json(intermediate / "papo_proactive_eval_listwise_v4.groups.json", eval_groups)
        quality, issues = audit_v4_groups(
            train_groups,
            eval_groups,
            image_roots=args.image_root,
            allow_unavailable_images=args.synthetic_smoke and not args.require_local_images,
            source_manifest=source_manifest,
            min_manual_review_fraction=args.min_manual_review_fraction or 0.0,
        )
        write_quality_outputs(quality, issues, args.workspace / "reports")
        release = build_release(
            args.workspace,
            train_groups,
            eval_groups,
            release_kind=args.release_kind,
            source_manifest=source_manifest,
            quality_report=quality,
            candidate_provenance=candidate_provenance,
        )
    except (OSError, ValueError, V4ValidationError) as error:
        print(f"LISTWISE-V4 BUILD FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(release, ensure_ascii=False, indent=2))
    print("SYNTHETIC SMOKE RELEASE BUILT; NOT A FORMAL FULL-V4 RELEASE" if args.synthetic_smoke else "FORMAL V4 RELEASE BUILT")


if __name__ == "__main__":
    main()
