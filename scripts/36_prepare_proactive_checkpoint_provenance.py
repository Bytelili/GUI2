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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write clean proactive provenance into a preserved checkpoint so it can be evaluated post-hoc."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    project_config = load_config(args.config)
    training_path = Path(args.training_config).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(str(training["output_dir"])).resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if checkpoint_dir != output_dir and checkpoint_dir.parent != output_dir:
        raise ValueError(f"Checkpoint directory is outside output_dir: {checkpoint_dir}")
    if not (checkpoint_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Checkpoint has no adapter_model.safetensors: {checkpoint_dir}")

    dataset_dir = Path(str(training["dataset_dir"])).resolve()
    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_names = _names(training.get("dataset"))
    eval_names = _names(training.get("eval_dataset"))
    if not dataset_names or not eval_names:
        raise ValueError("Checkpoint evaluation requires explicit dataset and eval_dataset in the training config")

    dataset_hashes: dict[str, str] = {}
    train_rows = 0
    eval_rows = 0
    for name in dataset_names:
        path = dataset_dir / dataset_info[name]["file_name"]
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset_hashes[name] = sha256_file(path)
        train_rows += len(data)
    for name in eval_names:
        path = dataset_dir / dataset_info[name]["file_name"]
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset_hashes[name] = sha256_file(path)
        eval_rows += len(data)

    protocol_id = str(project_config["data"]["protocol"]["protocol_id"])
    manifest_path = config_path(project_config, "paths.protocol_dir") / "protocol_manifest.json"
    provenance = {
        "status": "passed",
        "protocol_id": protocol_id,
        "training_config": str(training_path),
        "training_config_sha256": sha256_file(training_path),
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "datasets": dataset_names,
        "eval_datasets": eval_names,
        "dataset_hashes": dataset_hashes,
        "dataset_info_sha256": sha256_file(dataset_info_path),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "adapter_provenance": None,
        "source_checkpoint": str(checkpoint_dir),
        "stable_model_dir": str(checkpoint_dir),
        "checkpoint_eval_only": True,
    }
    provenance_path = checkpoint_dir / "papo_training_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(f"Wrote: {provenance_path}")


def _names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Unsupported dataset declaration: {value!r}")


if __name__ == "__main__":
    main()
