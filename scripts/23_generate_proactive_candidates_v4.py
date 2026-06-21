from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import (  # noqa: E402
    V4ValidationError,
    create_candidate_requests,
    import_candidate_results,
    merge_candidate_shards,
    write_json,
)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage PAPO v4 candidate request/export boundary.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    requests = subparsers.add_parser("requests", help="Stage A: create target-free model generation requests.")
    requests.add_argument("--train-tasks", type=Path, required=True)
    requests.add_argument("--eval-tasks", type=Path, required=True)
    requests.add_argument("--workspace", type=Path, required=True)
    requests.add_argument("--base-model", required=True)
    requests.add_argument("--adapter", required=True)
    requests.add_argument("--num-candidates", type=int, default=4)
    requests.add_argument("--temperature", type=float, default=0.8)
    requests.add_argument("--top-p", type=float, default=0.95)
    requests.add_argument("--max-new-tokens", type=int, default=128)
    requests.add_argument("--seed", type=int, default=20260621)
    requests.add_argument("--shard-index", type=int, default=0)
    requests.add_argument("--shard-count", type=int, default=1)
    requests.add_argument("--no-resume", action="store_true")

    imports = subparsers.add_parser("import", help="Stage B: validate and import externally generated candidates.")
    imports.add_argument("--train-tasks", type=Path, required=True)
    imports.add_argument("--eval-tasks", type=Path, required=True)
    imports.add_argument("--workspace", type=Path, required=True)
    imports.add_argument("--train-candidates", type=Path, required=True)
    imports.add_argument("--eval-candidates", type=Path, required=True)
    imports.add_argument("--train-manifest", type=Path, required=True)
    imports.add_argument("--eval-manifest", type=Path, required=True)
    imports.add_argument("--train-manifest-sha256", required=True)
    imports.add_argument("--eval-manifest-sha256", required=True)
    imports.add_argument("--expected-base-model", required=True)
    imports.add_argument("--expected-adapter", required=True)
    merge = subparsers.add_parser("merge", help="Merge all externally generated shards and create a signed manifest.")
    merge.add_argument("--tasks", type=Path, required=True)
    merge.add_argument("--shard", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--base-model", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--decoding-json", required=True)
    merge.add_argument("--candidate-count", type=int, required=True)
    args = parser.parse_args()

    try:
        if args.stage == "requests":
            decoding = {
                "num_candidates": args.num_candidates,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
            }
            reports = {}
            for split, tasks in (("train", args.train_tasks), ("eval", args.eval_tasks)):
                directory = args.workspace / "candidates" / split
                reports[split] = create_candidate_requests(
                    tasks,
                    directory / f"candidate_requests_{split}.jsonl",
                    directory / f"candidate_requests_{split}.manifest.json",
                    split=split,
                    base_model=args.base_model,
                    adapter=args.adapter,
                    decoding=decoding,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    global_seed=args.seed,
                    resume=not args.no_resume,
                    code_commit=_commit(),
                )
            write_json(args.workspace / "manifests" / "candidate_request_manifest.json", reports)
        elif args.stage == "import":
            reports = {}
            for split, tasks, candidates, manifest, manifest_sha256 in (
                ("train", args.train_tasks, args.train_candidates, args.train_manifest, args.train_manifest_sha256),
                ("eval", args.eval_tasks, args.eval_candidates, args.eval_manifest, args.eval_manifest_sha256),
            ):
                reports[split] = import_candidate_results(
                    tasks,
                    candidates,
                    manifest,
                    args.workspace / "candidates" / split / f"ui_tars_sft_{split}_candidates.jsonl",
                    expected_manifest_sha256=manifest_sha256,
                    expected_base_model=args.expected_base_model,
                    expected_adapter=args.expected_adapter,
                )
            write_json(args.workspace / "manifests" / "candidate_import_manifest.json", reports)
        else:
            reports = merge_candidate_shards(
                args.tasks,
                args.shard,
                args.output,
                args.manifest,
                base_model=args.base_model,
                adapter=args.adapter,
                decoding=json.loads(args.decoding_json),
                candidate_count=args.candidate_count,
            )
    except (OSError, ValueError, V4ValidationError) as error:
        print(f"CANDIDATE {args.stage.upper()} FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"CANDIDATE {args.stage.upper()} PASSED")


if __name__ == "__main__":
    main()
