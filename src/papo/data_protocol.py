from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .official_data import read_csv_rows


PROTOCOL_FILES = {
    "proactive_train_targets": "proactive_train_targets.csv",
    "proactive_eval_targets": "proactive_eval_targets.csv",
    "proactive_history": "proactive_history.csv",
    "execution_train_targets": "execution_train_targets.csv",
    "execution_eval_targets": "execution_eval_targets.csv",
    "execution_references": "execution_references.csv",
}


def build_formal_protocol(
    official_root: str | Path,
    output_dir: str | Path,
    *,
    source_train_split: str,
    proactive_test_split: str,
    execution_test_split: str,
    validation_fraction: float,
    min_validation_per_user: int,
    protocol_id: str,
) -> dict[str, Any]:
    official_root = Path(official_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = official_root / source_train_split
    proactive_test_path = official_root / proactive_test_split
    execution_test_path = official_root / execution_test_split
    source_rows = read_csv_rows(source_path)
    proactive_test_rows = read_csv_rows(proactive_test_path)
    execution_test_rows = read_csv_rows(execution_test_path)

    proactive_test_keys = episode_keys(proactive_test_rows)
    execution_test_keys = episode_keys(execution_test_rows)

    proactive_candidates = exclude_keys(source_rows, proactive_test_keys)
    execution_candidates = exclude_keys(source_rows, execution_test_keys)
    proactive_train, proactive_eval = chronological_user_split(
        proactive_candidates,
        validation_fraction=validation_fraction,
        min_validation_per_user=min_validation_per_user,
    )
    execution_train, execution_eval = chronological_user_split(
        execution_candidates,
        validation_fraction=validation_fraction,
        min_validation_per_user=min_validation_per_user,
    )

    datasets = {
        "proactive_train_targets": proactive_train,
        "proactive_eval_targets": proactive_eval,
        "proactive_history": proactive_train,
        "execution_train_targets": execution_train,
        "execution_eval_targets": execution_eval,
        "execution_references": execution_train,
    }
    for name, rows in datasets.items():
        write_csv_rows(output_dir / PROTOCOL_FILES[name], rows, source_rows)

    manifest: dict[str, Any] = {
        "protocol_id": protocol_id,
        "status": "passed",
        "policy": {
            "source_train_split": source_train_split,
            "proactive_test_split": proactive_test_split,
            "execution_test_split": execution_test_split,
            "validation_fraction": validation_fraction,
            "min_validation_per_user": min_validation_per_user,
            "validation_policy": "latest_per_user",
            "proactive_history_policy": "train_partition_only_and_strictly_before_target",
            "execution_reference_policy": "train_partition_only_and_strictly_before_target",
            "same_track_official_test_policy": "hard_exclude",
            "cross_track_overlap_policy": "report_only_for_separate_models",
        },
        "source_hashes": {
            source_train_split: sha256_file(source_path),
            proactive_test_split: sha256_file(proactive_test_path),
            execution_test_split: sha256_file(execution_test_path),
        },
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "excluded_same_track_episode_keys": {
            "proactive": len(episode_keys(source_rows) & proactive_test_keys),
            "execution": len(episode_keys(source_rows) & execution_test_keys),
        },
        "cross_track_episode_key_warnings": {
            "proactive_train_vs_execution_test": len(episode_keys(proactive_train) & execution_test_keys),
            "execution_train_vs_suggestion_test": len(episode_keys(execution_train) & proactive_test_keys),
        },
        "files": {},
    }

    _validate_protocol(datasets, proactive_test_keys, execution_test_keys)
    for name, filename in PROTOCOL_FILES.items():
        path = output_dir / filename
        manifest["files"][name] = {
            "path": filename,
            "sha256": sha256_file(path),
            "rows": len(datasets[name]),
        }

    manifest_path = output_dir / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def chronological_user_split(
    rows: list[dict[str, str]],
    *,
    validation_fraction: float,
    min_validation_per_user: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if min_validation_per_user < 1:
        raise ValueError("min_validation_per_user must be at least one")

    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_user[str(row.get("user_id") or "")].append(row)

    train: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    for user_id in sorted(by_user):
        user_rows = sorted(by_user[user_id], key=lambda row: (str(row.get("time") or ""), _stable_row(row)))
        if len(user_rows) < 2:
            train.extend(user_rows)
            continue
        holdout = max(min_validation_per_user, math.ceil(len(user_rows) * validation_fraction))
        holdout = min(holdout, len(user_rows) - 1)
        train.extend(user_rows[:-holdout])
        evaluation.extend(user_rows[-holdout:])
    return train, evaluation


def episode_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("user_id") or "").strip(), str(row.get("time") or "").strip()


def episode_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {episode_key(row) for row in rows}


def exclude_keys(rows: list[dict[str, str]], excluded: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if episode_key(row) not in excluded]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_rows(path: str | Path, rows: list[dict[str, str]], schema_rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list((rows or schema_rows or [{}])[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(
    datasets: dict[str, list[dict[str, str]]],
    proactive_test_keys: set[tuple[str, str]],
    execution_test_keys: set[tuple[str, str]],
) -> None:
    proactive_train_keys = episode_keys(datasets["proactive_train_targets"])
    proactive_eval_keys = episode_keys(datasets["proactive_eval_targets"])
    execution_train_keys = episode_keys(datasets["execution_train_targets"])
    execution_eval_keys = episode_keys(datasets["execution_eval_targets"])

    checks = {
        "proactive train/eval overlap": proactive_train_keys & proactive_eval_keys,
        "execution train/eval overlap": execution_train_keys & execution_eval_keys,
        "proactive train/test overlap": proactive_train_keys & proactive_test_keys,
        "proactive eval/test overlap": proactive_eval_keys & proactive_test_keys,
        "execution train/test overlap": execution_train_keys & execution_test_keys,
        "execution eval/test overlap": execution_eval_keys & execution_test_keys,
    }
    failures = {name: values for name, values in checks.items() if values}
    if failures:
        summary = ", ".join(f"{name}={len(values)}" for name, values in failures.items())
        raise ValueError(f"Formal data protocol validation failed: {summary}")

    if episode_keys(datasets["proactive_history"]) != proactive_train_keys:
        raise ValueError("Proactive history must exactly equal the proactive train partition")
    if episode_keys(datasets["execution_references"]) != execution_train_keys:
        raise ValueError("Execution references must exactly equal the execution train partition")


def _stable_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)
