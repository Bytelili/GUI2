from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.io import write_json  # noqa: E402
from papo.proactive_fixed_export import read_jsonish_rows  # noqa: E402


FILE_NAMES = [
    "proactive_dpo_train.jsonl",
    "proactive_dpo_eval.jsonl",
    "proactive_oracle_sft_train.jsonl",
    "proactive_oracle_sft_eval.jsonl",
    "proactive_rerank_train.jsonl",
    "proactive_rerank_eval.jsonl",
    "proactive_weighted_listwise_train.jsonl",
    "proactive_weighted_listwise_eval.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check proactive_fixed_clean image path resolution.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--llamafactory_data_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    project_root = Path(args.project_root).resolve()
    llamafactory_data_dir = Path(args.llamafactory_data_dir).resolve()
    out_path = Path(args.out)

    report: dict[str, Any] = {
        "status": "passed",
        "data_dir": str(data_dir),
        "project_root": str(project_root),
        "llamafactory_data_dir": str(llamafactory_data_dir),
        "files": {},
        "summary": {},
    }
    all_images = 0
    existing_images = 0
    resolution_counter: Counter[str] = Counter()
    missing_examples: list[dict[str, Any]] = []

    for name in FILE_NAMES:
        path = data_dir / name
        if not path.exists():
            report["files"][name] = {"status": "missing"}
            report["status"] = "failed"
            continue
        rows = read_jsonish_rows(path)
        file_images = 0
        file_existing = 0
        file_counter: Counter[str] = Counter()
        file_missing: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            images = row.get("images")
            if not isinstance(images, list):
                continue
            for image in images:
                image_text = str(image or "").strip()
                if not image_text:
                    continue
                file_images += 1
                resolved_by, resolved_path = resolve_image(image_text, project_root, llamafactory_data_dir)
                file_counter[resolved_by] += 1
                if resolved_by != "missing":
                    file_existing += 1
                elif len(file_missing) < 10:
                    file_missing.append(
                        {
                            "row_index": index,
                            "image": image_text,
                            "checked_paths": candidate_paths(image_text, project_root, llamafactory_data_dir),
                        }
                    )
        all_images += file_images
        existing_images += file_existing
        resolution_counter.update(file_counter)
        missing_examples.extend(file_missing[: max(0, 10 - len(missing_examples))])
        report["files"][name] = {
            "status": "passed",
            "rows": len(rows),
            "image_count": file_images,
            "resolved_count": file_existing,
            "resolved_ratio": (file_existing / file_images) if file_images else 1.0,
            "resolution_distribution": dict(file_counter),
            "missing_examples": file_missing,
        }

    report["summary"] = {
        "image_count": all_images,
        "resolved_count": existing_images,
        "resolved_ratio": (existing_images / all_images) if all_images else 1.0,
        "resolution_distribution": dict(resolution_counter),
        "missing_examples": missing_examples,
    }
    if report["status"] == "passed" and report["summary"]["resolved_ratio"] < 0.95:
        report["status"] = "warning"

    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


def candidate_paths(image_path: str, project_root: Path, llamafactory_data_dir: Path) -> list[str]:
    candidates = [Path(image_path)]
    candidates.append(project_root / image_path)
    candidates.append(llamafactory_data_dir / image_path)
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def resolve_image(image_path: str, project_root: Path, llamafactory_data_dir: Path) -> tuple[str, Path | None]:
    original = Path(image_path)
    if original.exists():
        return "original", original
    project_candidate = project_root / image_path
    if project_candidate.exists():
        return "project_root", project_candidate
    llamafactory_candidate = llamafactory_data_dir / image_path
    if llamafactory_candidate.exists():
        return "llamafactory_data_dir", llamafactory_candidate
    return "missing", None


if __name__ == "__main__":
    main()
