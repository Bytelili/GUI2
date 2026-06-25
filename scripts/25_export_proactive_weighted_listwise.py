from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import (  # noqa: E402
    WeightedListwiseExportConfig,
    export_weighted_listwise_rows,
    read_wide_csv,
    split_rows_by_user_time,
    write_jsonl_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export weighted proactive listwise ablation data from train_wide.csv.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_eval", required=True)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--min_context_prob", type=float, default=0.02)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_wide_csv(args.input)
    train_rows, eval_rows, split_report = split_rows_by_user_time(rows, args.eval_ratio, "user_time")
    config = WeightedListwiseExportConfig(
        temperature=args.temperature,
        min_context_prob=args.min_context_prob,
    )
    train_export, train_report = export_weighted_listwise_rows(train_rows, config)
    eval_export, eval_report = export_weighted_listwise_rows(eval_rows, config)
    write_jsonl_dataset(Path(args.out_train), train_export)
    write_jsonl_dataset(Path(args.out_eval), eval_export)
    print(
        json.dumps(
            {
                "config": {
                    "temperature": config.temperature,
                    "min_context_prob": config.min_context_prob,
                    "min_oracle_prob": config.min_oracle_prob,
                    "max_oracle_prob": config.max_oracle_prob,
                },
                "split": split_report,
                "train": train_report,
                "eval": eval_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
