from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one execution baseline variant for LLaMA-Factory SFT.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--variant", default="official_icl")
    parser.add_argument("--raw_root", default=r"D:\0608DataSet\Raw")
    parser.add_argument("--asset_prefix", default="RawDataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--dataset_info", default="")
    args = parser.parse_args()

    rows = [row for row in _read_jsonl(args.prompts) if row.get("variant") == args.variant]
    if args.limit > 0:
        rows = rows[: args.limit]
    exported = [
        {
            "messages": [
                {"role": "user", "content": "<image>" + str(row.get("prompt") or "")},
                {"role": "assistant", "content": str(row.get("target_action") or "")},
            ],
            "images": [_relative_asset(str(row.get("image") or ""), args.raw_root, args.asset_prefix)],
            "metadata": {
                "variant": args.variant,
                "episode_id": row.get("episode_id", ""),
                "step_id": row.get("step_id", ""),
            },
        }
        for row in rows
        if row.get("image") and row.get("target_action")
    ]
    write_json(args.out, exported)
    if args.dataset_info and args.dataset_name:
        _register_dataset(args.dataset_info, args.dataset_name, Path(args.out).name)
    print(f"variant: {args.variant}")
    print(f"rows: {len(exported)}")
    print(f"wrote: {args.out}")


def _relative_asset(path: str, raw_root: str, asset_prefix: str) -> str:
    source = Path(path)
    try:
        relative = source.resolve().relative_to(Path(raw_root).resolve()).as_posix()
        return f"{asset_prefix.strip('/')}/{relative}"
    except ValueError:
        return source.as_posix()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _register_dataset(path: str, dataset_name: str, file_name: str) -> None:
    info_path = Path(path)
    data = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    data[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    write_json(info_path, data)
    print(f"registered dataset: {dataset_name} in {info_path}")


if __name__ == "__main__":
    main()
