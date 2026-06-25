from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import (  # noqa: E402
    RerankExportConfig,
    export_rerank_rows,
    read_wide_csv,
    split_rows_by_user_time,
    write_jsonl_dataset,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export proactive rerank data from train_wide.csv.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_eval", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--shuffle_candidates", default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_candidates", type=int, default=2)
    args = parser.parse_args()

    rows = read_wide_csv(args.input)
    train_rows, eval_rows, split_report = split_rows_by_user_time(rows, args.eval_ratio, "user_time")
    config = RerankExportConfig(
        min_candidates=args.min_candidates,
        shuffle_candidates=str(args.shuffle_candidates).lower() not in {"false", "0", "no"},
        seed=args.seed,
    )
    train_export, train_report = export_rerank_rows(train_rows, config)
    eval_export, eval_report = export_rerank_rows(eval_rows, config)
    write_jsonl_dataset(Path(args.out_train), train_export)
    write_jsonl_dataset(Path(args.out_eval), eval_export)
    payload = {
        "config": {
            "min_candidates": config.min_candidates,
            "shuffle_candidates": config.shuffle_candidates,
            "seed": config.seed,
        },
        "split": split_report,
        "train": train_report,
        "eval": eval_report,
    }
    write_report(Path(args.report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
