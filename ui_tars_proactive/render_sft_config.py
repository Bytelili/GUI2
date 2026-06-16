from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.data_protocol import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a clean-v2 Proactive SFT config for UI-TARS.")
    parser.add_argument("--source", default="configs/llamafactory/generated/proactive_sft.yaml")
    parser.add_argument("--model", default="/home/dumike/zyy/GUI/backbone/UI-TARS-7B")
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--output", default="configs/llamafactory/generated/ui_tars_7b_proactive_sft.yaml")
    parser.add_argument(
        "--output-dir",
        default="/home/dumike/zyy/GUI2/LLaMA-Factory/saves/papo/ui_tars_7b_proactive_sft_clean_v2",
    )
    parser.add_argument(
        "--logging-dir",
        default="/home/dumike/zyy/GUI2/runs/papo/ui_tars_7b_proactive_sft_clean_v2",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    source = _resolve(args.source)
    target = _resolve(args.output)
    model = Path(args.model)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update(
        {
            "model_name_or_path": str(model),
            "template": args.template,
            "output_dir": args.output_dir,
            "logging_dir": args.logging_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "gradient_checkpointing": True,
            "save_total_limit": None,
            "load_best_model_at_end": False,
            "overwrite_output_dir": False,
            "bf16": True,
            "tf32": True,
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report: dict[str, Any] = {
        "status": "rendered",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "target": str(target),
        "model": str(model),
        "model_exists": model.is_dir(),
        "model_config_sha256": _optional_sha256(model / "config.json"),
        "template": args.template,
        "output_dir": args.output_dir,
        "effective_global_batch_4gpu": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * 4
        ),
        "key_training_args": {
            key: config.get(key)
            for key in [
                "dataset",
                "eval_dataset",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "learning_rate",
                "gradient_checkpointing",
                "eval_steps",
                "save_steps",
            ]
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("UI-TARS PROACTIVE SFT CONFIG RENDERED")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


if __name__ == "__main__":
    main()
