from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local PAPO LLaMA-Factory datasets.")
    parser.add_argument("--dataset_dir", default="LLaMA-Factory/data/papo")
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset names. By default all datasets in dataset_info.json are validated.",
    )
    parser.add_argument("--check_images", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if selected:
        missing = [name for name in selected if name not in info]
        if missing:
            raise KeyError(f"Datasets are missing from dataset_info.json: {missing}")
        info = {name: info[name] for name in selected}
    total = 0
    missing_images: list[str] = []
    listwise_sums: dict[str, float] = {}
    for name, config in info.items():
        path = root / config["file_name"]
        rows = _load_rows(path)
        total += len(rows)
        for index, row in enumerate(rows):
            prompt = _prompt_text(row)
            images = [str(item) for item in row.get("images", [])]
            if prompt.count("<image>") != len(images):
                raise ValueError(f"{name}[{index}] image marker count does not match images")
            if config.get("ranking") and (not row.get("chosen") or not row.get("rejected")):
                raise ValueError(f"{name}[{index}] is missing chosen or rejected")
            weight_column = config.get("columns", {}).get("preference_weight")
            if weight_column:
                weight = float(row.get(weight_column, 0.0))
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"{name}[{index}] has an invalid preference weight")
            target_column = config.get("columns", {}).get("preference_target")
            if target_column:
                target = float(row.get(target_column, 0.0))
                if not math.isfinite(target) or not 0.5 < target <= 1.0:
                    raise ValueError(f"{name}[{index}] has an invalid preference target")
            listwise_column = config.get("columns", {}).get("listwise_weight")
            if listwise_column:
                weight = float(row.get(listwise_column, 0.0))
                if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
                    raise ValueError(f"{name}[{index}] has an invalid listwise weight")
                metadata = row.get("metadata", {})
                group = str(
                    metadata.get("group_id")
                    or metadata.get("preference_group_id")
                    or metadata.get("task_id")
                    or f"{metadata.get('tree_id', '')}::{metadata.get('node_id', '')}"
                )
                listwise_sums[group] = listwise_sums.get(group, 0.0) + weight
            if args.check_images:
                missing_images.extend(image for image in images if not (root / image).exists())
        print(f"{name}: {len(rows)} rows")
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} image paths are missing; first: {missing_images[0]}")
    invalid_groups = [group for group, value in listwise_sums.items() if abs(value - 1.0) > 1e-6]
    if invalid_groups:
        raise ValueError(f"{len(invalid_groups)} listwise groups do not sum to one; first: {invalid_groups[0]}")
    print(f"validation passed: {total} rows")


def _prompt_text(row: dict[str, Any]) -> str:
    return "".join(
        str(message.get("content") or message.get("value") or "")
        for message in row.get("messages", []) + row.get("conversations", [])
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


if __name__ == "__main__":
    main()
