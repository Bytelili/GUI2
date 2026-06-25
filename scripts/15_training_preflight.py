from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
from papo.data_protocol import sha256_file  # noqa: E402
from papo.official_data import read_csv_rows  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    read_jsonish_rows,
    validate_dpo_rows,
    validate_rerank_rows,
    validate_sft_rows,
    validate_weighted_listwise_rows,
)


PROACTIVE_FIXED_TRAIN_DATASETS = {
    "papo_proactive_oracle_sft_train",
    "papo_proactive_dpo_train",
    "papo_proactive_rerank_train",
    "papo_proactive_weighted_listwise_train",
}
PROACTIVE_FIXED_EVAL_DATASETS = {
    "papo_proactive_oracle_sft_eval",
    "papo_proactive_dpo_eval",
    "papo_proactive_rerank_eval",
    "papo_proactive_weighted_listwise_eval",
}
PROACTIVE_FIXED_DATASETS = PROACTIVE_FIXED_TRAIN_DATASETS | PROACTIVE_FIXED_EVAL_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard gate for strict PAPO training and resume.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate without writing a resumable gate.")
    parser.add_argument(
        "--adopt-completed-run",
        action="store_true",
        help="Gate a completed legacy v3 run only after verifying its best checkpoint still exists.",
    )
    args = parser.parse_args()

    project_config = load_config(args.config)
    training_path = Path(args.training_config).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    report = validate_training(
        project_config,
        training_path,
        training,
        adopt_completed_run=args.adopt_completed_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.check_only:
        output_dir = Path(str(training["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        gate_path = output_dir / "papo_preflight.json"
        _validate_resume_gate(output_dir, gate_path, report)
        gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Training gate written: {gate_path}")
    print("STRICT TRAINING PREFLIGHT PASSED")


def validate_training(
    project_config: dict[str, Any],
    training_path: Path,
    training: dict[str, Any],
    *,
    adopt_completed_run: bool = False,
) -> dict[str, Any]:
    dataset_names = _names(training.get("dataset"))
    eval_names = _names(training.get("eval_dataset"))
    is_proactive_fixed = all(name in PROACTIVE_FIXED_DATASETS for name in dataset_names + eval_names)
    if not dataset_names or not eval_names:
        raise ValueError("Formal training requires explicit dataset and eval_dataset")
    if float(training.get("val_size", 0.0) or 0.0) != 0.0:
        raise ValueError("Formal training must use val_size: 0.0 with an explicit temporal eval_dataset")
    output_dir = Path(str(training.get("output_dir") or ""))
    is_v3_dataset = all(name.endswith("_v3") for name in dataset_names + eval_names)
    is_named_clean = "clean_v2" in output_dir.name or "clean_v3" in output_dir.name
    is_legacy_v3_adoption = adopt_completed_run and is_v3_dataset and output_dir.name.endswith("_v3")
    if not is_named_clean and not is_legacy_v3_adoption:
        raise ValueError(f"Formal output_dir must use a clean_v2 or clean_v3 name, got: {output_dir}")
    if training.get("load_best_model_at_end"):
        raise ValueError("load_best_model_at_end must remain false; use the post-training finalizer")
    if training.get("save_steps") != training.get("eval_steps"):
        raise ValueError("save_steps must equal eval_steps so every evaluated checkpoint is preserved")
    save_total_limit = training.get("save_total_limit")
    if save_total_limit not in {None, 0} and not adopt_completed_run:
        raise ValueError("save_total_limit must be unset so the best evaluated checkpoint cannot be pruned")
    if adopt_completed_run and save_total_limit not in {None, 0, 1, 2, 3}:
        raise ValueError("Completed-run adoption only permits save_total_limit values 1, 2, or 3")

    dataset_dir = Path(str(training["dataset_dir"]))
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    train_rows, train_hashes = _load_datasets(dataset_dir, info, dataset_names)
    eval_rows, eval_hashes = _load_datasets(dataset_dir, info, eval_names)
    protocol_id = str(project_config["data"]["protocol"]["protocol_id"])
    if is_proactive_fixed:
        manifest_path = None
        _validate_proactive_fixed_rows(dataset_names, train_rows, "train")
        _validate_proactive_fixed_rows(eval_names, eval_rows, "eval")
        _validate_proactive_fixed_no_overlap(train_rows, eval_rows)
    else:
        protocol_dir = config_path(project_config, "paths.protocol_dir")
        manifest_path = protocol_dir / "protocol_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "passed":
            raise ValueError("Protocol manifest status is not passed")
        _verify_protocol_hashes(project_config, protocol_dir, manifest)
        _validate_rows(train_rows, "train", protocol_id)
        _validate_rows(eval_rows, "eval", protocol_id)
        _validate_no_leakage(project_config, dataset_names + eval_names, train_rows, eval_rows)
    _validate_adapter(training)
    completed_run = _validate_completed_run(output_dir) if adopt_completed_run else None

    return {
        "status": "passed",
        "protocol_id": protocol_id,
        "training_config": str(training_path),
        "training_config_sha256": sha256_file(training_path),
        "output_dir": str(output_dir),
        "datasets": dataset_names,
        "eval_datasets": eval_names,
        "dataset_hashes": {**train_hashes, **eval_hashes},
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "protocol_manifest_sha256": sha256_file(manifest_path) if manifest_path is not None else None,
        "adapter_provenance": _adapter_provenance(training),
        "completed_run_adoption": completed_run,
    }


def _verify_protocol_hashes(project_config: dict[str, Any], protocol_dir: Path, manifest: dict[str, Any]) -> None:
    official_root = config_path(project_config, "paths.official_root")
    for filename, expected in manifest["source_hashes"].items():
        actual = sha256_file(official_root / filename)
        if actual != expected:
            raise ValueError(f"Official source changed after protocol build: {filename}")
    for name, record in manifest["files"].items():
        path = protocol_dir / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Protocol file changed after manifest build: {name}")


def _load_datasets(
    dataset_dir: Path,
    info: dict[str, Any],
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for name in names:
        if name not in info:
            raise KeyError(f"Dataset is missing from dataset_info.json: {name}")
        path = dataset_dir / info[name]["file_name"]
        data = read_jsonish_rows(path)
        if not data:
            raise ValueError(f"Formal dataset is empty: {name}")
        rows.extend(data)
        hashes[name] = sha256_file(path)
    return rows, hashes


def _validate_proactive_fixed_rows(dataset_names: list[str], rows: list[dict[str, Any]], partition: str) -> None:
    failures: list[str] = []
    for name in dataset_names:
        if name.endswith("_oracle_sft_train") or name.endswith("_oracle_sft_eval"):
            report = validate_sft_rows(rows)
            if not report.get("passed"):
                failures.extend(report.get("issues", []))
            if report.get("prompt_tag_count"):
                failures.append(f"{name} still contains [system]/[user] tags")
        elif name.endswith("_rerank_train") or name.endswith("_rerank_eval"):
            report = validate_rerank_rows(rows)
            if not report.get("passed"):
                failures.extend(report.get("issues", []))
            prompt_failures = _count_bad_prompts(rows)
            if prompt_failures:
                failures.append(f"{name} has {prompt_failures} dirty prompts")
        elif name.endswith("_weighted_listwise_train") or name.endswith("_weighted_listwise_eval"):
            report = validate_weighted_listwise_rows(rows)
            if not report.get("passed"):
                failures.extend(report.get("issues", []))
            prompt_failures = _count_bad_prompts(rows)
            if prompt_failures:
                failures.append(f"{name} has {prompt_failures} dirty prompts")
        elif name.endswith("_dpo_train") or name.endswith("_dpo_eval"):
            report = validate_dpo_rows(rows)
            if not report.get("passed"):
                failures.extend(report.get("issues", []))
            prompt_failures = _count_bad_conversation_prompts(rows)
            if prompt_failures:
                failures.append(f"{name} has {prompt_failures} dirty prompts")
    if failures:
        raise ValueError(f"{partition} proactive_fixed validation failed: {failures[:5]}")


def _validate_proactive_fixed_no_overlap(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    train_ids = {_proactive_fixed_row_id(row) for row in train_rows}
    eval_ids = {_proactive_fixed_row_id(row) for row in eval_rows}
    overlap = {item for item in train_ids & eval_ids if item}
    if overlap:
        raise ValueError(f"proactive_fixed train/eval overlap detected: {len(overlap)} rows")


def _validate_rows(rows: list[dict[str, Any]], partition: str, protocol_id: str) -> None:
    bad_partition = 0
    bad_protocol = 0
    missing_episode = 0
    non_temporal_context = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        bad_partition += metadata.get("partition") != partition
        bad_protocol += metadata.get("protocol_id") != protocol_id
        missing_episode += not bool(metadata.get("papo_episode_id"))
        target_time = _episode_time(str(metadata.get("papo_episode_id") or ""))
        for key in [
            "history_episode_ids",
            "same_user_reference_episode_ids",
            "cross_user_reference_episode_ids",
        ]:
            non_temporal_context += sum(
                not _episode_time(str(item)) < target_time
                for item in metadata.get(key, [])
                if target_time and _episode_time(str(item))
            )
    if bad_partition or bad_protocol or missing_episode or non_temporal_context:
        raise ValueError(
            f"{partition} provenance validation failed: bad_partition={bad_partition}, "
            f"bad_protocol={bad_protocol}, missing_episode={missing_episode}, "
            f"non_temporal_context={non_temporal_context}"
        )


def _validate_no_leakage(
    project_config: dict[str, Any],
    dataset_names: list[str],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    is_proactive = all("proactive" in name for name in dataset_names)
    is_execution = all("execution" in name for name in dataset_names)
    if not (is_proactive or is_execution):
        raise ValueError(f"Mixed-track formal training is not supported by this gate: {dataset_names}")

    protocol = project_config["data"]["protocol"]
    test_name = protocol["proactive_test_split"] if is_proactive else protocol["execution_test_split"]
    test_rows = read_csv_rows(config_path(project_config, "paths.official_root") / test_name)
    test_ids = {f"{row.get('user_id', '')}__{row.get('time', '')}" for row in test_rows}
    train_ids = {_episode_id(row) for row in train_rows}
    eval_ids = {_episode_id(row) for row in eval_rows}
    if train_ids & eval_ids:
        raise ValueError(f"Train/eval target overlap detected: {len(train_ids & eval_ids)} episodes")
    if train_ids & test_ids or eval_ids & test_ids:
        raise ValueError(
            f"Official same-track test target leakage detected: "
            f"train={len(train_ids & test_ids)}, eval={len(eval_ids & test_ids)}"
        )

    context_keys = (
        ["history_episode_ids"]
        if is_proactive
        else ["same_user_reference_episode_ids", "cross_user_reference_episode_ids"]
    )
    leaked_context: set[str] = set()
    for row in train_rows + eval_rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for key in context_keys:
            leaked_context.update(str(item) for item in metadata.get(key, []) if str(item) in test_ids)
    if leaked_context:
        raise ValueError(f"Official same-track test context leakage detected: {len(leaked_context)} episodes")


def _validate_adapter(training: dict[str, Any]) -> None:
    adapter = training.get("adapter_name_or_path")
    if not adapter:
        return
    provenance_path = Path(str(adapter)) / "papo_training_provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Upstream adapter has no clean training provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "passed":
        raise ValueError(f"Upstream adapter provenance is not passed: {provenance_path}")


def _adapter_provenance(training: dict[str, Any]) -> dict[str, Any] | None:
    adapter = training.get("adapter_name_or_path")
    if not adapter:
        return None
    path = Path(str(adapter)) / "papo_training_provenance.json"
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_resume_gate(output_dir: Path, gate_path: Path, report: dict[str, Any]) -> None:
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints and not gate_path.exists() and report.get("completed_run_adoption") is None:
        raise ValueError(f"Refusing to resume checkpoints without a strict preflight gate: {output_dir}")
    if gate_path.exists():
        previous = json.loads(gate_path.read_text(encoding="utf-8"))
        keys = ["protocol_id", "training_config_sha256", "dataset_hashes", "adapter_provenance"]
        changed = [key for key in keys if previous.get(key) != report.get(key)]
        if changed:
            raise ValueError(f"Refusing stale resume because gated inputs changed: {changed}")


def _validate_completed_run(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "trainer_state.json"
    adapter_path = output_dir / "adapter_model.safetensors"
    if not state_path.exists() or not adapter_path.exists():
        raise FileNotFoundError(
            "Completed-run adoption requires root trainer_state.json and adapter_model.safetensors: "
            f"{output_dir}"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state.get("global_step", 0) or 0)
    records = {
        int(item["step"]): float(item["eval_loss"])
        for item in state.get("log_history", [])
        if "step" in item and "eval_loss" in item
    }
    if global_step <= 0 or not records:
        raise ValueError(f"Completed-run adoption found no finished training state or eval_loss: {output_dir}")

    best_step, best_loss = min(records.items(), key=lambda item: (item[1], item[0]))
    best_checkpoint = output_dir / f"checkpoint-{best_step}"
    if best_step == global_step and not best_checkpoint.is_dir():
        best_checkpoint = output_dir
    if not (best_checkpoint / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"The globally best evaluated checkpoint-{best_step} was pruned and cannot be adopted: {output_dir}"
        )
    return {
        "status": "verified",
        "global_step": global_step,
        "best_step": best_step,
        "best_eval_loss": best_loss,
        "best_checkpoint": str(best_checkpoint),
    }


def _episode_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("papo_episode_id") or "")


def _episode_time(episode_id: str) -> str:
    return episode_id.split("__", 1)[1] if "__" in episode_id else ""


def _proactive_fixed_row_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        metadata.get("group_id")
        or metadata.get("task_id")
        or metadata.get("papo_episode_id")
        or metadata.get("user_id")
    )


def _count_bad_prompts(rows: list[dict[str, Any]]) -> int:
    bad = 0
    for row in rows:
        messages = row.get("messages") or []
        if len(messages) < 2:
            bad += 1
            continue
        prompt = str(messages[1].get("value") or messages[1].get("content") or "")
        if "[system]" in prompt.lower() or "[user]" in prompt.lower():
            bad += 1
    return bad


def _count_bad_conversation_prompts(rows: list[dict[str, Any]]) -> int:
    bad = 0
    for row in rows:
        messages = row.get("conversations") or []
        if len(messages) < 2:
            bad += 1
            continue
        prompt = str(messages[1].get("value") or messages[1].get("content") or "")
        if "[system]" in prompt.lower() or "[user]" in prompt.lower():
            bad += 1
    return bad


def _names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
