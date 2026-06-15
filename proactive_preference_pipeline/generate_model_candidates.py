from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "LLaMA-Factory" / "src"))

from papo.config import load_config  # noqa: E402
from papo.data_protocol import sha256_file  # noqa: E402
from papo.proactive_adapter import validate_proactive_adapter  # noqa: E402
from papo.proactive_prediction import build_inference_request  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resumable sampled Proactive candidates from clean SFT.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_candidates < 1:
        raise ValueError("num-candidates must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    config = load_config(args.config)
    task_path = Path(args.tasks).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    validate_proactive_adapter(adapter, config)
    tasks = _read_jsonl(task_path)
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    _validate_tasks(tasks)
    identity = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "model_name_or_path": str(config["paths"]["qwen_model_path"]),
        "template": str(config["training"]["template"]),
        "task_path": str(task_path),
        "task_sha256": sha256_file(task_path),
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seed": args.seed,
    }
    _validate_identity(output, identity)
    completed_rows, removed = _prepare_resume(output, {str(task.get("task_id") or "") for task in tasks})
    completed = {str(row.get("task_id") or "") for row in completed_rows}
    pending = [task for task in tasks if str(task.get("task_id") or "") not in completed]
    print(
        f"assigned={len(tasks)}, completed={len(completed)}, removed_for_retry={removed}, pending={len(pending)}, "
        f"shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )
    if not pending:
        _write_manifest(output, identity)
        print("PROACTIVE MODEL CANDIDATE GENERATION COMPLETED")
        return

    from llamafactory.chat import ChatModel

    model = ChatModel(
        {
            "model_name_or_path": str(config["paths"]["qwen_model_path"]),
            "adapter_name_or_path": str(adapter),
            "template": str(config["training"]["template"]),
            "finetuning_type": "lora",
            "stage": "sft",
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
            "trust_remote_code": True,
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    import torch

    for index, task in enumerate(pending, start=1):
        task_id = str(task.get("task_id") or "")
        request = build_inference_request(task)
        started = time.perf_counter()
        candidates: list[str] = []
        errors: list[str] = []
        for sample_index in range(args.num_candidates):
            sample_seed = _sample_seed(args.seed, task_id, sample_index)
            torch.manual_seed(sample_seed)
            try:
                response = model.chat(
                    request["messages"],
                    system=request["system"],
                    images=request["images"],
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )[0]
                text = str(response.response_text or "").strip()
                if text and text not in candidates:
                    candidates.append(text)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                traceback.print_exc()
        row = {
            "task_id": task_id,
            "episode_id": str(task.get("metadata", {}).get("papo_episode_id") or ""),
            "candidates": candidates,
            "requested_candidates": args.num_candidates,
            "unique_candidates": len(candidates),
            "errors": errors,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
        _append_jsonl(output, row)
        print(
            f"[{index}/{len(pending)}] {task_id} | unique={len(candidates)} | errors={len(errors)}",
            flush=True,
        )
    _write_manifest(output, identity)
    print("PROACTIVE MODEL CANDIDATE GENERATION COMPLETED")


def _validate_tasks(tasks: list[dict]) -> None:
    ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Candidate generation tasks contain empty or duplicate task IDs")


def _validate_identity(output: Path, identity: dict) -> None:
    path = output.with_suffix(".identity.json")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Refusing stale candidate generation resume: {path}")
    elif output.exists() and output.stat().st_size:
        raise ValueError(f"Refusing candidate resume without identity: {output}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_seed(seed: int, task_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_resume(path: Path, assigned_ids: set[str]) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    removed = 0
    seen: set[str] = set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Candidate resume contains a corrupt non-final row: {path}") from None
            removed += 1
            continue
        task_id = str(row.get("task_id") or "")
        if task_id not in assigned_ids:
            raise ValueError(f"Candidate resume contains an unassigned task ID: {task_id}")
        if task_id in seen:
            raise ValueError(f"Candidate resume contains a duplicate task ID: {task_id}")
        seen.add(task_id)
        if row.get("candidates"):
            rows.append(row)
        else:
            removed += 1
    if removed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    return rows, removed


def _write_manifest(output: Path, identity: dict) -> None:
    rows = _read_jsonl(output)
    manifest = {
        "status": "passed",
        **identity,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "records": len(rows),
        "records_with_no_candidates": sum(not row.get("candidates") for row in rows),
        "sample_errors": sum(len(row.get("errors") or []) for row in rows),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


if __name__ == "__main__":
    main()
