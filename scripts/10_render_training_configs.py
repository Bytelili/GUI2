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


DATASETS = {
    "proactive_sft": ("sft", "papo_proactive_train_sft", "papo_proactive_eval_sft"),
    "proactive_oracle_sft_fixed": ("sft", "papo_proactive_oracle_sft_train", "papo_proactive_oracle_sft_eval"),
    "proactive_dpo_fixed": ("dpo", "papo_proactive_dpo_train", "papo_proactive_dpo_eval"),
    "proactive_rerank_fixed": ("sft", "papo_proactive_rerank_train", "papo_proactive_rerank_eval"),
    (
        "proactive_weighted_listwise_fixed"
    ): ("sft", "papo_proactive_weighted_listwise_train", "papo_proactive_weighted_listwise_eval"),
    "execution_sft": ("sft", "papo_execution_train_sft", "papo_execution_eval_sft"),
    "execution_listwise": ("sft", "papo_execution_train_listwise", "papo_execution_eval_listwise"),
    "execution_dpo": ("dpo", "papo_execution_train_dpo", "papo_execution_eval_dpo"),
}


def _default_adapter_name(name: str, stage: str, section: dict[str, Any]) -> str | None:
    explicit = section.get("adapter_name_or_path")
    if explicit:
        return str(explicit)
    if name == "proactive_dpo_fixed":
        return "proactive_oracle_sft_fixed_clean_v2_best"
    if name == "proactive_weighted_listwise_fixed":
        return "proactive_oracle_sft_fixed_clean_v2_best"
    if name == "execution_dpo":
        return "execution_listwise_clean_v2_best"
    if name == "execution_listwise":
        return "execution_sft_clean_v2_best"
    return None


def _resolve_adapter_path(checkpoint_root: str, adapter_name: str) -> str:
    adapter_path = Path(adapter_name)
    if adapter_path.is_absolute():
        return str(adapter_path)
    return _join_path(checkpoint_root, adapter_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render strict LLaMA-Factory YAML files from config.yaml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "configs/llamafactory/generated"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, (stage, dataset, eval_dataset) in DATASETS.items():
        rendered = _training_config(config, name, stage, dataset, eval_dataset)
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
        print(f"wrote: {path}")


def _training_config(
    config: dict[str, Any],
    name: str,
    stage: str,
    dataset: str,
    eval_dataset: str,
) -> dict[str, Any]:
    training = config["training"]
    section = training[name]
    checkpoint_root = _portable_path(config, "checkpoint_root")
    logging_root = _portable_path(config, "logging_root")
    model_path = str(config.get("paths", {}).get("qwen_model_path") or training["model_name_or_path"])
    run_name = str(section.get("output_name") or f"{name}_clean_v2")
    rendered_stage = str(section.get("stage", stage))
    eval_steps = int(section["eval_steps"])
    save_steps = int(section.get("save_steps", eval_steps))
    result: dict[str, Any] = {
        "model_name_or_path": model_path,
        "image_max_pixels": training["image_max_pixels"],
        "trust_remote_code": True,
        "stage": rendered_stage,
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_rank": int(section.get("lora_rank", training["lora_rank"])),
        "lora_alpha": int(section.get("lora_alpha", training.get("lora_alpha", 32))),
        "lora_dropout": float(section.get("lora_dropout", training.get("lora_dropout", 0.0))),
        "lora_target": "all",
        "dataset_dir": _portable_path(config, "llamafactory_data_dir"),
        "dataset": dataset,
        "eval_dataset": eval_dataset,
        "template": training["template"],
        "cutoff_len": training["cutoff_len"],
        "val_size": 0.0,
        "output_dir": _join_path(checkpoint_root, run_name),
        "logging_dir": _join_path(logging_root, run_name),
        "logging_steps": int(section.get("logging_steps", 10)),
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps if rendered_stage != "dpo" else eval_steps,
        "save_only_model": False,
        "load_best_model_at_end": False,
        "plot_loss": True,
        "report_to": "tensorboard",
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(section["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(section["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(section["gradient_accumulation_steps"]),
        "learning_rate": section["learning_rate"],
        "num_train_epochs": section["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": bool(section["gradient_checkpointing"]),
        "dataloader_num_workers": 8,
        "preprocessing_num_workers": 16,
        "ddp_timeout": 7200,
    }
    adapter_name = _default_adapter_name(name, rendered_stage, section)
    if rendered_stage == "dpo":
        result["pref_beta"] = section["pref_beta"]
        result["pref_loss"] = section.get("pref_loss", "sigmoid")
        if adapter_name:
            result["adapter_name_or_path"] = _resolve_adapter_path(checkpoint_root, adapter_name)
    if section.get("use_papo_listwise", False):
        result["use_papo_listwise"] = True
        if adapter_name:
            result["adapter_name_or_path"] = _resolve_adapter_path(checkpoint_root, adapter_name)
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
