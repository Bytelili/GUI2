from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import verify_release  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for retrieval-only grouped Listwise-v4.")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--allow-unavailable-images", action="store_true")
    args = parser.parse_args()

    config_path, release = args.training_config.resolve(), args.release_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((release / "listwise_v4_manifest.json").read_text(encoding="utf-8", errors="strict"))
    blockers: list[str] = []
    runtime_blockers: list[str] = []
    warnings: list[str] = []
    required = {
        "use_papo_group_listwise": True,
        "use_papo_listwise": False,
        "packing": False,
        "papo_allow_nonformal_smoke": False,
        "papo_allow_nonformal_retrieval": True,
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
        blockers.extend(f"release:{value}" for value in verification.get("errors", []))
    if manifest.get("release_kind") != "retrieval_only_v4":
        blockers.append("release: release_kind must be retrieval_only_v4")
    if manifest.get("release_status") != "retrieval_only_not_for_formal_training":
        blockers.append("release: invalid retrieval-only release status")
    if manifest.get("formal_full_v4_complete") is not False or manifest.get("training_claim_allowed") is not False:
        blockers.append("release: non-formal claim boundary changed")
    if manifest.get("candidate_provenance") is not None:
        blockers.append("release: model candidate provenance must remain null")

    group_counts: dict[str, int] = {}
    group_sizes: Counter[int] = Counter()
    candidate_sources: Counter[str] = Counter()
    candidate_count = 0
    unavailable_images: set[str] = set()
    for split in ("train", "eval"):
        rows = json.loads((release / f"papo_proactive_{split}_listwise_v4.json").read_text(encoding="utf-8"))
        group_counts[split] = len(rows)
        if len(rows) != int((manifest.get("group_counts") or {}).get(split, -1)):
            blockers.append(f"data:{split}: manifest group count mismatch")
        for row in rows:
            candidates = row.get("candidates")
            probabilities = row.get("target_distribution")
            oracle_index = row.get("oracle_index")
            if not isinstance(candidates, list) or len(candidates) not in (2, 3):
                blockers.append(f"data:{split}:{row.get('group_id')}: group size")
                continue
            if not isinstance(probabilities, list) or len(probabilities) != len(candidates):
                blockers.append(f"data:{split}:{row.get('group_id')}: probability alignment")
                continue
            if abs(sum(float(value) for value in probabilities) - 1.0) > 1e-6:
                blockers.append(f"data:{split}:{row.get('group_id')}: probability sum")
            if not isinstance(oracle_index, int) or not 0 <= oracle_index < len(candidates):
                blockers.append(f"data:{split}:{row.get('group_id')}: oracle index")
            sources = [str(item.get("source") or "") for item in candidates]
            if any(source.startswith("cross_user") or source == "ui_tars_sft" for source in sources):
                blockers.append(f"data:{split}:{row.get('group_id')}: forbidden candidate source")
            candidate_sources.update(sources)
            group_sizes[len(candidates)] += 1
            candidate_count += len(candidates)
            unavailable_images.update(str(value) for value in row.get("images", []) if not Path(str(value)).is_file())

    if unavailable_images:
        message = f"images: unavailable={len(unavailable_images)}"
        (warnings if args.allow_unavailable_images else blockers).append(message)
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
        elif torch.cuda.device_count() < 4:
            runtime_blockers.append(f"runtime: expected four GPUs, found {torch.cuda.device_count()}")
        else:
            minimum = min(torch.cuda.get_device_properties(index).total_memory for index in range(4))
            if minimum < 40 * 1024**3:
                runtime_blockers.append(f"runtime: minimum GPU memory is only {minimum / 1024**3:.2f} GiB")
    except ImportError:
        runtime_blockers.append("runtime: torch is not installed")

    status = "blocked" if blockers or (runtime_blockers and not args.data_only) else "passed"
    if not blockers and runtime_blockers and args.data_only:
        status = "data_passed_runtime_blocked"
    report = {
        "status": status,
        "experiment_kind": "full-scale history-retrieval-only grouped Listwise-v4 engineering experiment",
        "formal_full_v4_complete": False,
        "training_config": str(config_path),
        "release_dir": str(release),
        "group_counts": group_counts,
        "group_size_distribution": dict(group_sizes),
        "candidate_count": candidate_count,
        "candidate_sources": dict(candidate_sources),
        "blockers": blockers,
        "runtime_blockers": runtime_blockers,
        "warnings": warnings,
        "claim_boundary": "no model-generated candidates; not full-v4 and not a formal effect claim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == "blocked":
        raise SystemExit(1)
    print("PAPO RETRIEVAL-ONLY GROUPED LISTWISE-V4 PREFLIGHT PASSED WITH DECLARED BOUNDARY")


if __name__ == "__main__":
    main()
