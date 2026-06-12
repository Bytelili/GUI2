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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard gate for strict PAPO training and resume.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate without writing a resumable gate.")
    args = parser.parse_args()

    project_config = load_config(args.config)
    training_path = Path(args.training_config).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    report = validate_training(project_config, training_path, training)
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
) -> dict[str, Any]:
    dataset_names = _names(training.get("dataset"))
    eval_names = _names(training.get("eval_dataset"))
    if not dataset_names or not eval_names:
        raise ValueError("Formal training requires explicit dataset and eval_dataset")
    if float(training.get("val_size", 0.0) or 0.0) != 0.0:
        raise ValueError("Formal training must use val_size: 0.0 with an explicit temporal eval_dataset")
    output_dir = Path(str(training.get("output_dir") or ""))
    if "clean_v2" not in output_dir.name:
        raise ValueError(f"Formal output_dir must use a clean_v2 name, got: {output_dir}")
    if training.get("load_best_model_at_end"):
        raise ValueError("load_best_model_at_end must remain false; use the post-training finalizer")
    if training.get("save_steps") != training.get("eval_steps"):
        raise ValueError("save_steps must equal eval_steps so every evaluated checkpoint is preserved")
    if training.get("save_total_limit") not in {None, 0}:
        raise ValueError("save_total_limit must be unset so the best evaluated checkpoint cannot be pruned")

    protocol_dir = config_path(project_config, "paths.protocol_dir")
    manifest_path = protocol_dir / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("Protocol manifest status is not passed")
    _verify_protocol_hashes(project_config, protocol_dir, manifest)

    dataset_dir = Path(str(training["dataset_dir"]))
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    train_rows, train_hashes = _load_datasets(dataset_dir, info, dataset_names)
    eval_rows, eval_hashes = _load_datasets(dataset_dir, info, eval_names)
    protocol_id = str(project_config["data"]["protocol"]["protocol_id"])
    _validate_rows(train_rows, "train", protocol_id)
    _validate_rows(eval_rows, "eval", protocol_id)
    _validate_no_leakage(project_config, dataset_names + eval_names, train_rows, eval_rows)
    _validate_adapter(training)

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
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "adapter_provenance": _adapter_provenance(training),
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
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            raise ValueError(f"Formal dataset is empty: {name}")
        rows.extend(data)
        hashes[name] = sha256_file(path)
    return rows, hashes


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
    if checkpoints and not gate_path.exists():
        raise ValueError(f"Refusing to resume checkpoints without a strict preflight gate: {output_dir}")
    if gate_path.exists():
        previous = json.loads(gate_path.read_text(encoding="utf-8"))
        keys = ["protocol_id", "training_config_sha256", "dataset_hashes", "adapter_provenance"]
        changed = [key for key in keys if previous.get(key) != report.get(key)]
        if changed:
            raise ValueError(f"Refusing stale resume because gated inputs changed: {changed}")


def _episode_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("papo_episode_id") or "")


def _episode_time(episode_id: str) -> str:
    return episode_id.split("__", 1)[1] if "__" in episode_id else ""


def _names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
