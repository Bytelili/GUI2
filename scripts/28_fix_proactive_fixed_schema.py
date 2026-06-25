from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import load_config  # noqa: E402
from papo.io import write_json, write_jsonl  # noqa: E402
from papo.llamafactory_export import dataset_info  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    SFT_SYSTEM_PROMPT,
    clean_prompt_text,
    compute_soft_target,
    read_jsonish_rows,
    relativize_image_paths,
)


FILE_SPECS = {
    "proactive_oracle_sft_train.jsonl": "sft",
    "proactive_oracle_sft_eval.jsonl": "sft",
    "proactive_dpo_train.jsonl": "dpo",
    "proactive_dpo_eval.jsonl": "dpo",
    "proactive_rerank_train.jsonl": "rerank",
    "proactive_rerank_eval.jsonl": "rerank",
    "proactive_weighted_listwise_train.jsonl": "listwise",
    "proactive_weighted_listwise_eval.jsonl": "listwise",
}

FIXED_DATASET_NAMES = {
    "papo_proactive_oracle_sft_train",
    "papo_proactive_oracle_sft_eval",
    "papo_proactive_dpo_train",
    "papo_proactive_dpo_eval",
    "papo_proactive_rerank_train",
    "papo_proactive_rerank_eval",
    "papo_proactive_weighted_listwise_train",
    "papo_proactive_weighted_listwise_eval",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean proactive_fixed schema into proactive_fixed_clean.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--llamafactory_data_dir", default="")
    parser.add_argument("--dpo_beta", type=float, default=None)
    parser.add_argument("--min_target", type=float, default=None)
    parser.add_argument("--max_target", type=float, default=None)
    parser.add_argument("--make_image_relative", default="")
    parser.add_argument("--image_root", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    fixed_cfg = dict(config.get("proactive_fixed", {}))

    input_dir = Path(args.input_dir or fixed_cfg.get("output_dir") or "data/proactive_fixed")
    output_dir = Path(args.output_dir or fixed_cfg.get("clean_output_dir") or "data/proactive_fixed_clean")
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    report_path = Path(
        args.report
        or fixed_cfg.get("clean_report_dir")
        or (output_dir.parent.parent / "reports" / "proactive_fixed_clean")
    )
    if report_path.suffix.lower() != ".json":
        report_path = report_path / "schema_fix_report.json"
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    llamafactory_data_dir_value = args.llamafactory_data_dir or config.get("paths", {}).get("llamafactory_data_dir") or ""
    llamafactory_data_dir = Path(llamafactory_data_dir_value) if llamafactory_data_dir_value else None
    if llamafactory_data_dir is not None and not llamafactory_data_dir.is_absolute():
        llamafactory_data_dir = PROJECT_ROOT / llamafactory_data_dir

    dpo_beta = float(args.dpo_beta if args.dpo_beta is not None else fixed_cfg.get("dpo_beta", 0.1))
    min_target = float(args.min_target if args.min_target is not None else fixed_cfg.get("min_target", 0.55))
    max_target = float(args.max_target if args.max_target is not None else fixed_cfg.get("max_target", 0.98))
    make_image_relative = _parse_bool(
        args.make_image_relative if args.make_image_relative != "" else fixed_cfg.get("make_image_relative", False)
    )
    image_root = str(args.image_root or fixed_cfg.get("image_root") or "")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    files_report: dict[str, Any] = {}
    total_rows = 0
    total_image_rewrites = 0
    total_prompt_rewrites = 0
    warnings: list[str] = []

    for filename, kind in FILE_SPECS.items():
        source = input_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing proactive_fixed input file: {source}")
        rows = read_jsonish_rows(source)
        cleaned: list[dict[str, Any]] = []
        file_image_rewrites = 0
        file_prompt_rewrites = 0
        for row in rows:
            fixed_row, image_rewrites, prompt_rewrites, row_warnings = _clean_row(
                row=row,
                kind=kind,
                dpo_beta=dpo_beta,
                min_target=min_target,
                max_target=max_target,
                make_image_relative=make_image_relative,
                image_root=image_root,
            )
            cleaned.append(fixed_row)
            file_image_rewrites += image_rewrites
            file_prompt_rewrites += prompt_rewrites
            warnings.extend(row_warnings)
        write_jsonl(output_dir / filename, cleaned)
        files_report[filename] = {
            "rows": len(cleaned),
            "image_rewrites": file_image_rewrites,
            "prompt_rewrites": file_prompt_rewrites,
        }
        total_rows += len(cleaned)
        total_image_rewrites += file_image_rewrites
        total_prompt_rewrites += file_prompt_rewrites

    mirrored_dir = None
    if llamafactory_data_dir is not None:
        mirrored_dir = _mirror_to_llamafactory(output_dir, llamafactory_data_dir)

    report = {
        "status": "passed",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "llamafactory_data_dir": str(llamafactory_data_dir) if llamafactory_data_dir is not None else None,
        "mirrored_dir": str(mirrored_dir) if mirrored_dir else None,
        "config": {
            "dpo_beta": dpo_beta,
            "min_target": min_target,
            "max_target": max_target,
            "make_image_relative": make_image_relative,
            "image_root": image_root,
        },
        "rows": total_rows,
        "image_rewrites": total_image_rewrites,
        "prompt_rewrites": total_prompt_rewrites,
        "files": files_report,
        "warnings": warnings[:200],
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _clean_row(
    *,
    row: dict[str, Any],
    kind: str,
    dpo_beta: float,
    min_target: float,
    max_target: float,
    make_image_relative: bool,
    image_root: str,
) -> tuple[dict[str, Any], int, int, list[str]]:
    fixed = dict(row)
    warnings: list[str] = []
    prompt_rewrites = 0

    images = [str(item) for item in row.get("images", [])]
    image_rewrites = 0
    if make_image_relative and image_root:
        images, image_rewrites = relativize_image_paths(images, image_root)
    fixed["images"] = images

    if kind == "dpo":
        conversations = list(row.get("conversations") or [])
        user_prompt = _first_message_text(conversations, {"user", "human"})
        system_text, user_text = clean_prompt_text(user_prompt, SFT_SYSTEM_PROMPT)
        prompt_rewrites = int(user_prompt != user_text)
        fixed["conversations"] = [
            {"from": "system", "value": system_text},
            {"from": "human", "value": user_text},
        ]
        fixed["chosen"] = {"from": "gpt", "value": _message_text(row.get("chosen") or {})}
        fixed["rejected"] = {"from": "gpt", "value": _message_text(row.get("rejected") or {})}
        metadata = dict(row.get("metadata") or {})
        reward_gap = _reward_gap(row)
        old_target = float(row.get("papo_target_probability") or 1.0)
        if reward_gap is not None:
            fixed["papo_target_probability"] = compute_soft_target(
                reward_gap,
                beta=dpo_beta,
                min_target=min_target,
                max_target=max_target,
            )
            metadata["reward_gap"] = reward_gap
            metadata["old_papo_target_probability"] = old_target
            metadata["soft_target_beta"] = dpo_beta
        else:
            fixed["papo_target_probability"] = old_target
            warnings.append(f"missing_reward_gap::{metadata.get('task_id') or 'unknown'}")
        fixed["papo_weight"] = _resolve_weight(row, reward_gap)
        fixed["metadata"] = metadata
        return fixed, image_rewrites, prompt_rewrites, warnings

    messages = list(row.get("messages") or [])
    assistant_value = _last_message_text(messages, {"assistant", "gpt"})
    user_prompt = _first_message_text(messages, {"user", "human"})
    system_text, user_text = clean_prompt_text(user_prompt, SFT_SYSTEM_PROMPT if kind != "rerank" else SFT_SYSTEM_PROMPT)
    prompt_rewrites = int(user_prompt != user_text)
    fixed["messages"] = [
        {"from": "system", "value": system_text},
        {"from": "human", "value": user_text},
        {"from": "gpt", "value": assistant_value},
    ]
    fixed["metadata"] = dict(row.get("metadata") or {})
    return fixed, image_rewrites, prompt_rewrites, warnings


def _mirror_to_llamafactory(output_dir: Path, llamafactory_data_dir: Path) -> Path:
    target_dir = llamafactory_data_dir / "proactive_fixed_clean"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.jsonl"):
        (target_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    dataset_info_path = llamafactory_data_dir / "dataset_info.json"
    info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    info.update({key: value for key, value in dataset_info().items() if key in FIXED_DATASET_NAMES})
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_dir


def _reward_gap(row: dict[str, Any]) -> float | None:
    metadata = dict(row.get("metadata") or {})
    for value in (metadata.get("reward_gap"), row.get("reward_gap")):
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    oracle = _as_float(metadata.get("oracle_reward_total"))
    negative = _as_float(metadata.get("negative_reward_total"))
    if oracle is not None and negative is not None:
        return oracle - negative
    return None


def _resolve_weight(row: dict[str, Any], reward_gap: float | None) -> float:
    existing = _as_float(row.get("papo_weight"))
    if existing is not None:
        return min(max(existing, 0.5), 3.0)
    if reward_gap is None:
        return 1.0
    return min(max(0.5 + 2.0 * reward_gap, 0.5), 3.0)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("from") or message.get("role") or "").strip().lower()


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("value") or message.get("content") or "")


def _first_message_text(messages: list[dict[str, Any]], roles: set[str]) -> str:
    for message in messages:
        if _message_role(message) in roles:
            return _message_text(message)
    return ""


def _last_message_text(messages: list[dict[str, Any]], roles: set[str]) -> str:
    for message in reversed(messages):
        if _message_role(message) in roles:
            return _message_text(message)
    return ""


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
