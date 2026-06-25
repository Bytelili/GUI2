from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import (  # noqa: E402
    export_oracle_sft_rows,
    read_wide_csv,
    split_rows_by_user_time,
    write_jsonl_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export oracle-only proactive SFT data from train_wide.csv.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_eval", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--split_by", default="user_time")
    args = parser.parse_args()

    rows = read_wide_csv(args.input)
    train_rows, eval_rows, split_report = split_rows_by_user_time(rows, args.eval_ratio, args.split_by)
    train_export, train_report = export_oracle_sft_rows(train_rows)
    eval_export, eval_report = export_oracle_sft_rows(eval_rows)

    write_jsonl_dataset(Path(args.out_train), train_export)
    write_jsonl_dataset(Path(args.out_eval), eval_export)
    print(json.dumps({"split": split_report, "train": train_report, "eval": eval_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
