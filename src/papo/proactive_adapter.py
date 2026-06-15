from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import config_path
from .data_protocol import sha256_file


def validate_proactive_adapter(adapter_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    provenance_path = adapter_dir / "papo_training_provenance.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not provenance_path.exists() or not adapter_path.exists():
        raise FileNotFoundError(f"Finalized adapter is incomplete: {adapter_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "passed":
        raise ValueError(f"Finalized adapter provenance is not passed: {provenance_path}")

    train_datasets = set(provenance.get("datasets") or [])
    eval_datasets = set(provenance.get("eval_datasets") or [])
    if not train_datasets or any(not str(name).startswith("papo_proactive_train_") for name in train_datasets):
        raise ValueError(f"Finalized adapter is not a clean Proactive adapter: {provenance_path}")
    if not eval_datasets or any(not str(name).startswith("papo_proactive_eval_") for name in eval_datasets):
        raise ValueError(f"Finalized adapter has no clean Proactive eval provenance: {provenance_path}")

    expected_protocol = str(config["data"]["protocol"]["protocol_id"])
    if provenance.get("protocol_id") != expected_protocol:
        raise ValueError(
            f"Finalized adapter protocol mismatch: expected={expected_protocol}, actual={provenance.get('protocol_id')}"
        )
    manifest_path = config_path(config, "paths.protocol_dir") / "protocol_manifest.json"
    if sha256_file(manifest_path) != provenance.get("protocol_manifest_sha256"):
        raise ValueError(f"Finalized adapter protocol manifest changed after training: {manifest_path}")

    dataset_dir = config_path(config, "paths.llamafactory_data_dir")
    dataset_info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    for name, expected_hash in dict(provenance.get("dataset_hashes") or {}).items():
        if name not in dataset_info:
            raise KeyError(f"Finalized adapter dataset is missing from dataset_info.json: {name}")
        dataset_path = dataset_dir / dataset_info[name]["file_name"]
        if sha256_file(dataset_path) != expected_hash:
            raise ValueError(f"Finalized adapter dataset changed after training: {name}")
    return provenance
