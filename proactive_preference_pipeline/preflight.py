from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ppipeline.io_utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard gate for Proactive preference optimization artifacts.")
    parser.add_argument("--manifest", default="data/proactive_preference/preference_manifest.json")
    parser.add_argument("--training-config", required=True)
    args = parser.parse_args()
    report = validate_preference_training(Path(args.manifest), Path(args.training_config))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PROACTIVE PREFERENCE PREFLIGHT PASSED")


def validate_preference_training(manifest_path: Path, training_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    training_path = training_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("Preference manifest status is not passed")
    if manifest.get("method") != "proactive_personalized_preference_v1":
        raise ValueError(f"Unexpected preference method: {manifest.get('method')}")
    for partition in ["train", "eval"]:
        report = manifest.get("partitions", {}).get(partition, {})
        if report.get("outside_train_reference_episodes") != 0 or report.get("temporal_violations") != 0:
            raise ValueError(f"Preference {partition} candidate provenance is not clean")

    train_names = _names(training.get("dataset"))
    eval_names = _names(training.get("eval_dataset"))
    if not train_names or not eval_names:
        raise ValueError("Preference training requires explicit train and eval datasets")
    expected_partition = {
        **{name: "train" for name in train_names},
        **{name: "eval" for name in eval_names},
    }
    dataset_records = manifest.get("datasets", {})
    verified: dict[str, dict[str, Any]] = {}
    for name, partition in expected_partition.items():
        if not name.startswith(f"papo_proactive_{partition}_"):
            raise ValueError(f"Preference dataset partition mismatch: {name}")
        if name not in dataset_records:
            raise KeyError(f"Preference dataset is not bound to the manifest: {name}")
        record = dataset_records[name]
        path = Path(str(record["path"]))
        if not path.exists():
            raise FileNotFoundError(f"Preference dataset is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"Preference dataset changed after audited build: {name}")
        verified[name] = {"path": str(path), "sha256": actual, "rows": record["rows"]}

    output_dir = Path(str(training.get("output_dir") or ""))
    if "proactive_preference" not in output_dir.name or "clean_v2" not in output_dir.name:
        raise ValueError(f"Unsafe preference output directory: {output_dir}")
    stage = str(training.get("stage") or "")
    if stage == "sft":
        if not training.get("use_papo_listwise") or any(not name.endswith("_listwise") for name in expected_partition):
            raise ValueError("Preference SFT stage must use PAPO Listwise datasets and objective")
    elif stage == "dpo":
        if training.get("pref_loss") != "papo" or any(not name.endswith("_dpo") for name in expected_partition):
            raise ValueError("Preference DPO stage must use PAPO-DPO datasets and objective")
    else:
        raise ValueError(f"Unsupported preference training stage: {stage}")
    return {
        "status": "passed",
        "method": manifest["method"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_config": str(training_path),
        "training_config_sha256": sha256_file(training_path),
        "stage": stage,
        "output_dir": str(output_dir),
        "datasets": verified,
    }


def _names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
