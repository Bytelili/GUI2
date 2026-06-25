from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_json  # noqa: E402
from papo.llamafactory_export import dataset_info  # noqa: E402
from papo.proactive_fixed_export import read_jsonish_rows  # noqa: E402


DATA_FILES = [
    "proactive_dpo_train.jsonl",
    "proactive_dpo_eval.jsonl",
    "proactive_oracle_sft_train.jsonl",
    "proactive_oracle_sft_eval.jsonl",
    "proactive_rerank_train.jsonl",
    "proactive_rerank_eval.jsonl",
    "proactive_weighted_listwise_train.jsonl",
    "proactive_weighted_listwise_eval.jsonl",
]

TRAINING_FILES = {
    "proactive_oracle_sft_fixed.yaml": {
        "dataset": "papo_proactive_oracle_sft_train",
        "eval_dataset": "papo_proactive_oracle_sft_eval",
        "stage": "sft",
    },
    "proactive_dpo_fixed.yaml": {
        "dataset": "papo_proactive_dpo_train",
        "eval_dataset": "papo_proactive_dpo_eval",
        "stage": "dpo",
    },
    "proactive_rerank_fixed.yaml": {
        "dataset": "papo_proactive_rerank_train",
        "eval_dataset": "papo_proactive_rerank_eval",
        "stage": "sft",
    },
    "proactive_weighted_listwise_fixed.yaml": {
        "dataset": "papo_proactive_weighted_listwise_train",
        "eval_dataset": "papo_proactive_weighted_listwise_eval",
        "stage": "sft",
        "use_papo_listwise": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight proactive_fixed formal training chain.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    out_path = Path(args.out)
    info = dataset_info()

    report: dict[str, Any] = {
        "status": "passed",
        "data_dir": str(data_dir),
        "config_dir": str(config_dir),
        "data_files": {},
        "training_files": {},
    }
    failures: list[str] = []

    for name in DATA_FILES:
        path = data_dir / name
        if not path.exists():
            failures.append(f"missing_data::{name}")
            continue
        rows = read_jsonish_rows(path)[:8]
        report["data_files"][name] = {
            "exists": True,
            "rows_checked": len(rows),
            "preview_ok": bool(rows),
        }
        if not rows:
            failures.append(f"empty_data::{name}")

    base_dir = data_dir.parent
    for filename, expected in TRAINING_FILES.items():
        path = config_dir / filename
        if not path.exists():
            failures.append(f"missing_config::{filename}")
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset_name = str(cfg.get("dataset") or "")
        eval_name = str(cfg.get("eval_dataset") or "")
        stage = str(cfg.get("stage") or "")
        item_report = {
            "dataset": dataset_name,
            "eval_dataset": eval_name,
            "stage": stage,
            "use_papo_listwise": cfg.get("use_papo_listwise"),
        }
        report["training_files"][filename] = item_report

        if dataset_name != expected["dataset"]:
            failures.append(f"{filename}::dataset")
        if eval_name != expected["eval_dataset"]:
            failures.append(f"{filename}::eval_dataset")
        if stage != expected["stage"]:
            failures.append(f"{filename}::stage")
        if expected.get("use_papo_listwise") is True and cfg.get("use_papo_listwise") is not True:
            failures.append(f"{filename}::use_papo_listwise")
        if filename == "proactive_oracle_sft_fixed.yaml" and stage != "sft":
            failures.append(f"{filename}::oracle_stage")
        if filename == "proactive_rerank_fixed.yaml" and stage != "sft":
            failures.append(f"{filename}::rerank_stage")
        if filename == "proactive_dpo_fixed.yaml" and stage != "dpo":
            failures.append(f"{filename}::dpo_stage")

        for name in [dataset_name, eval_name]:
            if name not in info:
                failures.append(f"{filename}::missing_dataset_info::{name}")
                continue
            dataset_path = base_dir / str(info[name]["file_name"])
            if not dataset_path.exists():
                failures.append(f"{filename}::missing_dataset_file::{name}")
                continue
            sample_rows = read_jsonish_rows(dataset_path)[:8]
            if not sample_rows:
                failures.append(f"{filename}::empty_dataset_file::{name}")

    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
