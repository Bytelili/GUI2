from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LLaMA-Factory YAML files from config.yaml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "configs/llamafactory/generated"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, stage, dataset in [
        ("proactive_sft", "sft", "papo_proactive_sft"),
        ("execution_sft", "sft", "papo_execution_sft"),
        ("execution_listwise", "sft", "papo_execution_listwise"),
        ("execution_dpo", "dpo", "papo_execution_dpo"),
    ]:
        rendered = _training_config(config, name, stage, dataset)
        (output / f"{name}.yaml").write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
        print(f"wrote: {output / f'{name}.yaml'}")


def _training_config(config: dict[str, Any], name: str, stage: str, dataset: str) -> dict[str, Any]:
    training = config["training"]
    section = training[name]
    checkpoint_root = _portable_path(config, "checkpoint_root")
    logging_root = _portable_path(config, "logging_root")
    model_path = str(config.get("paths", {}).get("qwen_model_path") or training["model_name_or_path"])
    result: dict[str, Any] = {
        "model_name_or_path": model_path,
        "image_max_pixels": training["image_max_pixels"],
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": training["lora_rank"],
        "lora_target": "all",
        "dataset_dir": _portable_path(config, "llamafactory_data_dir"),
        "dataset": dataset,
        "template": training["template"],
        "cutoff_len": training["cutoff_len"],
        "val_size": 0.05,
        "output_dir": _join_path(checkpoint_root, name),
        "logging_dir": _join_path(logging_root, name),
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "report_to": "tensorboard",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": section["learning_rate"],
        "num_train_epochs": section["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "gradient_checkpointing": True,
    }
    if stage == "dpo":
        result["pref_beta"] = section["pref_beta"]
        result["pref_loss"] = section.get("pref_loss", "sigmoid")
        result["adapter_name_or_path"] = _join_path(checkpoint_root, "execution_listwise")
    if section.get("use_papo_listwise", False):
        result["use_papo_listwise"] = True
        result["adapter_name_or_path"] = _join_path(checkpoint_root, "execution_sft")
    return result


def _portable_path(config: dict[str, Any], key: str) -> str:
    value = str(config["paths"][key])
    if value.startswith("/"):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(config["_project_root"]) / path)


def _join_path(root: str, child: str) -> str:
    if root.startswith("/"):
        return posixpath.join(root, child)
    return str(Path(root) / child)


if __name__ == "__main__":
    main()
