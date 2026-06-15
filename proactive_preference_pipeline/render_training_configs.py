from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path
from typing import Any

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Proactive preference optimization training configs.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "configs/llamafactory/generated"))
    parser.add_argument(
        "--sft-adapter",
        default="/home/dumike/zyy/GUI2/LLaMA-Factory/saves/papo/proactive_sft_clean_v2_best",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = str(config["paths"]["checkpoint_root"])
    logging_root = str(config["paths"]["logging_root"])
    listwise_best = _join(checkpoint_root, "proactive_preference_listwise_clean_v2_best")
    configs = {
        "proactive_preference_listwise.yaml": _base(
            config,
            stage="sft",
            train_dataset="papo_proactive_train_listwise",
            eval_dataset="papo_proactive_eval_listwise",
            output_dir=_join(checkpoint_root, "proactive_preference_listwise_clean_v2"),
            logging_dir=_join(logging_root, "proactive_preference_listwise_clean_v2"),
            adapter=args.sft_adapter,
            batch_size=2,
            accumulation=8,
            learning_rate=1e-5,
            epochs=2,
            eval_steps=50,
            extras={"use_papo_listwise": True},
        ),
        "proactive_preference_dpo.yaml": _base(
            config,
            stage="dpo",
            train_dataset="papo_proactive_train_dpo",
            eval_dataset="papo_proactive_eval_dpo",
            output_dir=_join(checkpoint_root, "proactive_preference_dpo_clean_v2"),
            logging_dir=_join(logging_root, "proactive_preference_dpo_clean_v2"),
            adapter=listwise_best,
            batch_size=1,
            accumulation=16,
            learning_rate=5e-6,
            epochs=2,
            eval_steps=25,
            extras={"pref_beta": 0.1, "pref_loss": "papo"},
        ),
    }
    for filename, value in configs.items():
        path = output / filename
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        print(f"wrote: {path}")
    print("PROACTIVE PREFERENCE TRAINING CONFIGS RENDERED")


def _base(
    config: dict[str, Any],
    *,
    stage: str,
    train_dataset: str,
    eval_dataset: str,
    output_dir: str,
    logging_dir: str,
    adapter: str,
    batch_size: int,
    accumulation: int,
    learning_rate: float,
    epochs: int,
    eval_steps: int,
    extras: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    return {
        "model_name_or_path": str(config["paths"]["qwen_model_path"]),
        "adapter_name_or_path": adapter,
        "image_max_pixels": training["image_max_pixels"],
        "trust_remote_code": True,
        "stage": stage,
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_rank": training["lora_rank"],
        "lora_target": "all",
        "dataset_dir": str(config["paths"]["llamafactory_data_dir"]),
        "dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "template": training["template"],
        "cutoff_len": training["cutoff_len"],
        "val_size": 0.0,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_only_model": False,
        "load_best_model_at_end": False,
        "plot_loss": True,
        "report_to": "tensorboard",
        "overwrite_output_dir": False,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 8,
        "preprocessing_num_workers": 16,
        "ddp_timeout": 7200,
        **extras,
    }


def _join(root: str, child: str) -> str:
    return posixpath.join(root, child) if root.startswith("/") else str(Path(root) / child)


if __name__ == "__main__":
    main()
