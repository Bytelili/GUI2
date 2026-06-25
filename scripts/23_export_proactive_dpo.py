from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import (  # noqa: E402
    DPOExportConfig,
    export_dpo_rows,
    read_wide_csv,
    split_rows_by_user_time,
    write_jsonl_dataset,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export proactive DPO data from train_wide.csv.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_eval", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_pairs_per_row", type=int, default=2)
    parser.add_argument("--min_reward_gap", type=float, default=0.05)
    parser.add_argument("--min_char_similarity", type=float, default=0.20)
    parser.add_argument("--max_char_similarity", type=float, default=0.98)
    parser.add_argument("--same_user_min_similarity", type=float, default=0.45)
    parser.add_argument("--same_user_max_similarity", type=float, default=0.95)
    args = parser.parse_args()

    rows = read_wide_csv(args.input)
    train_rows, eval_rows, split_report = split_rows_by_user_time(rows, args.eval_ratio, "user_time")
    config = DPOExportConfig(
        max_pairs_per_row=args.max_pairs_per_row,
        min_reward_gap=args.min_reward_gap,
        min_char_similarity=args.min_char_similarity,
        max_char_similarity=args.max_char_similarity,
        same_user_min_similarity=args.same_user_min_similarity,
        same_user_max_similarity=args.same_user_max_similarity,
    )
    train_export, train_report = export_dpo_rows(train_rows, config)
    eval_export, eval_report = export_dpo_rows(eval_rows, config)
    write_jsonl_dataset(Path(args.out_train), train_export)
    write_jsonl_dataset(Path(args.out_eval), eval_export)
    payload = {
        "config": {
            "max_pairs_per_row": config.max_pairs_per_row,
            "min_reward_gap": config.min_reward_gap,
            "min_char_similarity": config.min_char_similarity,
            "max_char_similarity": config.max_char_similarity,
            "same_user_min_similarity": config.same_user_min_similarity,
            "same_user_max_similarity": config.same_user_max_similarity,
        },
        "split": split_report,
        "train": train_report,
        "eval": eval_report,
    }
    write_report(Path(args.report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
