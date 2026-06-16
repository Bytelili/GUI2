from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "LLaMA-Factory" / "src"))

from papo.config import load_config  # noqa: E402
from papo.data_protocol import sha256_file  # noqa: E402
from papo.proactive_adapter import validate_proactive_adapter  # noqa: E402
from papo.proactive_prediction import (  # noqa: E402
    append_jsonl,
    build_inference_request,
    prediction_record,
    prepare_prediction_resume,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable Proactive predictions for a base or LoRA model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--require-adapter-provenance", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config = load_config(args.config)
    task_path = Path(args.tasks).resolve()
    model_path = Path(args.model).resolve()
    adapter_dir = Path(args.adapter).resolve() if args.adapter else None
    output_path = Path(args.output).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if adapter_dir is not None:
        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Adapter is missing adapter_model.safetensors: {adapter_dir}")
        if args.require_adapter_provenance:
            validate_proactive_adapter(adapter_dir, config)

    _validate_resume_identity(
        config_file=Path(args.config).resolve(),
        task_path=task_path,
        model_path=model_path,
        adapter_dir=adapter_dir,
        output_path=output_path,
        template=args.template,
        model_label=args.model_label,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_new_tokens=args.max_new_tokens,
        image_max_pixels=args.image_max_pixels,
    )
    tasks = read_jsonl(task_path)
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    _validate_assigned_tasks(tasks)
    completed, failed_removed = prepare_prediction_resume(tasks, output_path)
    pending = [task for task in tasks if str(task.get("task_id") or "") not in completed]
    print(
        f"model={args.model_label}, shard={args.shard_index}/{args.num_shards}, "
        f"assigned={len(tasks)}, completed={len(completed)}, "
        f"failed_removed_for_retry={failed_removed}, pending={len(pending)}",
        flush=True,
    )
    if not pending:
        print("No pending tasks.")
        return

    from llamafactory.chat import ChatModel

    chat_args: dict[str, Any] = {
        "model_name_or_path": str(model_path),
        "template": args.template,
        "infer_backend": "huggingface",
        "infer_dtype": "bfloat16",
        "trust_remote_code": True,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "image_max_pixels": args.image_max_pixels,
    }
    if adapter_dir is not None:
        chat_args.update(
            {
                "adapter_name_or_path": str(adapter_dir),
                "finetuning_type": "lora",
                "stage": "sft",
            }
        )
    model = ChatModel(chat_args)

    error_count = 0
    for index, task in enumerate(pending, start=1):
        request = build_inference_request(task)
        start = time.perf_counter()
        try:
            responses = model.chat(
                request["messages"],
                system=request["system"],
                images=request["images"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            response = responses[0]
            row = prediction_record(
                task,
                predicted_intent=response.response_text,
                elapsed_seconds=time.perf_counter() - start,
                prompt_tokens=response.prompt_length,
                response_tokens=response.response_length,
                finish_reason=response.finish_reason,
            )
        except Exception as error:
            error_count += 1
            row = prediction_record(
                task,
                predicted_intent="ERROR",
                elapsed_seconds=time.perf_counter() - start,
                prompt_tokens=0,
                response_tokens=0,
                finish_reason="error",
                error=f"{type(error).__name__}: {error}",
            )
            traceback.print_exc()
        append_jsonl(output_path, row)
        print(
            f"[{index}/{len(pending)}] {row['task_id']} | tokens={row['token']} | "
            f"time={row['time']} | error={bool(row['error'])}",
            flush=True,
        )
    manifest = {
        "status": "failed" if error_count else "completed",
        "model_label": args.model_label,
        "model_path": str(model_path),
        "model_config_sha256": _optional_sha256(model_path / "config.json"),
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "adapter_sha256": _optional_sha256(adapter_dir / "adapter_model.safetensors") if adapter_dir else None,
        "task_path": str(task_path),
        "task_sha256": sha256_file(task_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "template": args.template,
        "max_new_tokens": args.max_new_tokens,
        "image_max_pixels": args.image_max_pixels,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(tasks),
        "errors": error_count,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if error_count:
        raise RuntimeError(f"Shard completed with {error_count} failed predictions; rerun to retry them")


def _validate_resume_identity(
    *,
    config_file: Path,
    task_path: Path,
    model_path: Path,
    adapter_dir: Path | None,
    output_path: Path,
    template: str,
    model_label: str,
    shard_index: int,
    num_shards: int,
    max_new_tokens: int,
    image_max_pixels: int,
) -> None:
    identity = {
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "model_label": model_label,
        "model_path": str(model_path),
        "model_config_sha256": _optional_sha256(model_path / "config.json"),
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "adapter_sha256": _optional_sha256(adapter_dir / "adapter_model.safetensors") if adapter_dir else None,
        "template": template,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "image_max_pixels": image_max_pixels,
        "task_path": str(task_path),
        "task_sha256": sha256_file(task_path),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    identity_path = output_path.with_suffix(".identity.json")
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(f"Refusing stale prediction resume because shard identity changed: {identity_path}")
    elif output_path.exists() and output_path.stat().st_size > 0:
        raise ValueError(f"Refusing prediction resume without an identity file: {output_path}")
    else:
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_assigned_tasks(tasks: list[dict[str, Any]]) -> None:
    task_ids: set[str] = set()
    missing_images: list[str] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in task_ids:
            raise ValueError(f"Assigned shard contains an empty or duplicate task ID: {task_id}")
        task_ids.add(task_id)
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        for image in inputs.get("initial_screenshots", []):
            if not Path(str(image)).is_file():
                missing_images.append(str(image))
    if missing_images:
        raise FileNotFoundError(
            f"Assigned shard contains {len(missing_images)} missing screenshots; first={missing_images[0]}"
        )


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path and path.exists() else None


if __name__ == "__main__":
    main()
