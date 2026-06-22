from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import verify_release  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for the retrieval-only grouped Listwise-v4 smoke.")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--allow-unavailable-images", action="store_true")
    args = parser.parse_args()

    config_path = args.training_config.resolve()
    release = args.release_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    runtime_blockers: list[str] = []
    warnings: list[str] = []

    required = {
        "use_papo_group_listwise": True,
        "use_papo_listwise": False,
        "packing": False,
        "papo_allow_nonformal_smoke": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            blockers.append(f"config:{key}: expected={expected!r}, actual={config.get(key)!r}")
    if config.get("dataset") != "papo_proactive_train_listwise_v4":
        blockers.append("config: unexpected grouped train dataset")
    if config.get("eval_dataset") != "papo_proactive_eval_listwise_v4":
        blockers.append("config: unexpected grouped eval dataset")

    verification = verify_release(release)
    if verification["status"] != "passed":
        blockers.extend(f"release:{item}" for item in verification.get("errors", []))
    manifest = json.loads((release / "listwise_v4_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_status") != "synthetic_smoke_not_for_formal_training":
        blockers.append("release: expected the unchanged non-formal smoke status")
    if manifest.get("formal_full_v4_complete") is not False:
        blockers.append("release: formal_full_v4_complete must remain false")
    if manifest.get("candidate_provenance") is not None:
        warnings.append("release contains candidate provenance; this preflight was designed for retrieval-only smoke")

    group_counts: dict[str, int] = {}
    candidate_count = 0
    unavailable_images: set[str] = set()
    for split in ("train", "eval"):
        path = release / f"papo_proactive_{split}_listwise_v4.json"
        rows = json.loads(path.read_text(encoding="utf-8", errors="strict"))
        group_counts[split] = len(rows)
        for row in rows:
            candidates = row.get("candidates")
            probabilities = row.get("target_distribution")
            oracle_index = row.get("oracle_index")
            if not isinstance(candidates, list) or len(candidates) < 2:
                blockers.append(f"data:{split}:{row.get('group_id')}: invalid candidates")
                continue
            if not isinstance(probabilities, list) or len(probabilities) != len(candidates):
                blockers.append(f"data:{split}:{row.get('group_id')}: probability alignment")
                continue
            if abs(sum(float(value) for value in probabilities) - 1.0) > 1e-6:
                blockers.append(f"data:{split}:{row.get('group_id')}: probability sum")
            if not isinstance(oracle_index, int) or not 0 <= oracle_index < len(candidates):
                blockers.append(f"data:{split}:{row.get('group_id')}: oracle index")
            elif float(probabilities[oracle_index]) + 1e-8 < max(float(value) for value in probabilities):
                blockers.append(f"data:{split}:{row.get('group_id')}: oracle probability")
            candidate_count += len(candidates)
            unavailable_images.update(str(path) for path in row.get("images", []) if not Path(str(path)).is_file())

    if unavailable_images:
        message = f"images: unavailable={len(unavailable_images)}"
        if args.allow_unavailable_images:
            warnings.append(message)
        else:
            blockers.append(message)

    for key in ("model_name_or_path", "adapter_name_or_path", "dataset_dir"):
        value = config.get(key)
        if not value or not Path(str(value)).exists():
            runtime_blockers.append(f"runtime:{key}: unavailable: {value}")
    if shutil.which("llamafactory-cli") is None and importlib.util.find_spec("llamafactory") is None:
        runtime_blockers.append("runtime: LLaMA-Factory is not installed or importable")
    try:
        import torch

        if not torch.cuda.is_available():
            runtime_blockers.append("runtime: CUDA is unavailable")
        else:
            total_memory = min(
                torch.cuda.get_device_properties(index).total_memory for index in range(torch.cuda.device_count())
            )
            if total_memory < 20 * 1024**3:
                runtime_blockers.append(f"runtime: minimum GPU memory is only {total_memory / 1024**3:.2f} GiB")
    except ImportError:
        runtime_blockers.append("runtime: torch is not installed")

    if blockers:
        status = "blocked"
    elif runtime_blockers and args.data_only:
        status = "data_passed_runtime_blocked"
    elif runtime_blockers:
        status = "blocked"
    else:
        status = "passed"
    report: dict[str, Any] = {
        "status": status,
        "experiment_kind": "retrieval-only smoke experiment",
        "formal_full_v4_complete": False,
        "training_config": str(config_path),
        "release_dir": str(release),
        "group_counts": group_counts,
        "candidate_count": candidate_count,
        "blockers": blockers,
        "runtime_blockers": runtime_blockers,
        "warnings": warnings,
        "required_flags": required,
        "claim_boundary": "engineering pre-experiment only; not full-v4",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "blocked" or (runtime_blockers and not args.data_only):
        raise SystemExit(1)
    print("PAPO GROUP LISTWISE-V4 SMOKE PREFLIGHT PASSED WITH DECLARED BOUNDARY")


if __name__ == "__main__":
    main()
