from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .conditions import validate_source_tasks
from .io_utils import sha256_file, write_json, write_jsonl


def prepare_official_execution_tasks(
    project_config_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 0,
    target_split: str = "",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    from papo.config import config_path, load_config
    from papo.data_protocol import episode_keys
    from papo.official_data import read_csv_rows
    from papo.tasks import build_personalized_execution_tasks

    config = load_config(project_config_path)
    protocol_id = str(config["data"]["protocol"]["protocol_id"])
    official_root = config_path(config, "paths.official_root")
    protocol_dir = config_path(config, "paths.protocol_dir")
    raw_root = config_path(config, "paths.raw_root")
    configured_split = str(config["data"]["protocol"]["execution_test_split"])
    selected_split = target_split or configured_split
    if selected_split not in {configured_split, "sampled_test_execution.csv"}:
        raise ValueError(f"Unsupported official execution target split: {selected_split}")
    target_path = official_root / selected_split
    if not target_path.is_file():
        raise FileNotFoundError(f"Official execution target split does not exist: {target_path}")
    reference_path = protocol_dir / "execution_references.csv"
    profile_path = official_root / "user_profile.csv"
    references = config["papo"]["references"]
    tasks = build_personalized_execution_tasks(
        target_path,
        reference_path,
        profile_path,
        raw_root,
        limit=limit,
        require_complete=bool(config["data"]["require_complete"]),
        same_user_top_k=int(references["same_user_top_k"]),
        cross_user_top_k=int(references["cross_user_top_k"]),
        intent_similarity_threshold=float(references["intent_similarity_threshold"]),
        exclude_same_intent=bool(references["exclude_same_intent"]),
        provenance={
            "partition": "official_test",
            "protocol_id": protocol_id,
            "target_split": target_path.name,
            "reference_split": reference_path.name,
        },
    )
    validate_source_tasks(tasks)
    target_keys = episode_keys(read_csv_rows(target_path))
    reference_keys = episode_keys(read_csv_rows(reference_path))
    if target_keys & reference_keys:
        raise ValueError("Official execution targets overlap strict reference partition")
    output = Path(output_path).resolve()
    write_jsonl(output, tasks)
    report = {
        "status": "passed",
        "protocol_id": protocol_id,
        "target_path": str(target_path),
        "target_sha256": sha256_file(target_path),
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "official_target_rows": len(target_keys),
        "complete_tasks": len(tasks),
        "excluded_incomplete_targets": len(target_keys) - len(tasks),
        "target_reference_overlap": 0,
        "official_reference_implementation": "FingerTip-20K-main/personalized_execution.py",
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report
