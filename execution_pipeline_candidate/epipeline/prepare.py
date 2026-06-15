from __future__ import annotations

from pathlib import Path
from typing import Any

from .conditions import apply_condition
from .io_utils import read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl


def prepare_experiment(config_path: str | Path) -> dict[str, Any]:
    source = Path(config_path).resolve()
    config = read_json(source)
    validate_config(config)
    tasks_path = Path(str(config["tasks_path"])).resolve()
    tasks = read_jsonl(tasks_path)
    validate_task_provenance(tasks, str(config["protocol_id"]))
    output_root = Path(str(config["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    official_reference_audit = load_official_reference_audit(config.get("official_reference_audit_path"))
    model_by_id = {str(model["id"]): {**model, "identity": model_identity(config, model)} for model in config["models"]}
    prepared: dict[str, dict[str, Any]] = {}
    for run in config["runs"]:
        run_id = str(run["id"])
        transformed, condition_manifest = apply_condition(tasks, str(run["condition"]), seed=int(config["seed"]))
        prepared[run_id] = {
            "run": run,
            "model": model_by_id[str(run["model"])],
            "tasks": transformed,
            "condition_manifest": condition_manifest,
        }
    align_comparison_components(prepared, config.get("comparisons") or [])
    run_entries: list[dict[str, Any]] = []
    for run in config["runs"]:
        run_id = str(run["id"])
        model = prepared[run_id]["model"]
        transformed = prepared[run_id]["tasks"]
        condition_manifest = prepared[run_id]["condition_manifest"]
        if run.get("required") and not transformed:
            raise ValueError(f"Required run has zero eligible tasks: {run_id}")
        run_dir = output_root / "runs" / run_id
        run_tasks = run_dir / "tasks.jsonl"
        write_jsonl(run_tasks, transformed)
        entry = {
            **run,
            "model_spec": model,
            "model_identity": model["identity"],
            "run_dir": str(run_dir),
            "tasks_path": str(run_tasks),
            "tasks_sha256": sha256_file(run_tasks),
            "task_count": len(transformed),
            "condition_manifest": condition_manifest,
        }
        write_json(run_dir / "run_plan.json", entry)
        run_entries.append(entry)
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "experiment_id": str(config["experiment_id"]),
        "protocol_id": str(config["protocol_id"]),
        "config_path": str(source),
        "config_sha256": sha256_file(source),
        "tasks_path": str(tasks_path),
        "tasks_sha256": sha256_file(tasks_path),
        "source_task_count": len(tasks),
        "output_root": str(output_root),
        "seed": int(config["seed"]),
        "max_steps": int(config["max_steps"]),
        "max_invalid_actions": int(config["max_invalid_actions"]),
        "invalid_action_policy": str(config.get("invalid_action_policy") or "bounded_retry"),
        "inference": config["inference"],
        "device": config["device"],
        "success_rules_path": (
            str(Path(str(config["success_rules_path"])).resolve())
            if config.get("success_rules_path")
            else ""
        ),
        "official_reference_audit": official_reference_audit,
        "comparisons": config.get("comparisons") or [],
        "runs": run_entries,
    }
    manifest["identity_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "identity_sha256"})
    write_json(output_root / "experiment_manifest.json", manifest)
    return manifest


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "protocol_id",
        "tasks_path",
        "output_root",
        "seed",
        "max_steps",
        "max_invalid_actions",
        "inference",
        "device",
        "models",
        "runs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment config is missing keys: {sorted(missing)}")
    if not Path(str(config["tasks_path"])).is_file():
        raise FileNotFoundError(f"Execution task file does not exist: {config['tasks_path']}")
    if int(config["max_steps"]) < 1 or int(config["max_invalid_actions"]) < 1:
        raise ValueError("max_steps and max_invalid_actions must be positive")
    if str(config.get("invalid_action_policy") or "bounded_retry") not in {"bounded_retry", "official_wait"}:
        raise ValueError("invalid_action_policy must be bounded_retry or official_wait")
    models = config["models"]
    runs = config["runs"]
    if not isinstance(models, list) or not isinstance(runs, list) or not models or not runs:
        raise ValueError("models and runs must be non-empty lists")
    model_ids = [str(model.get("id") or "") for model in models if isinstance(model, dict)]
    run_ids = [str(run.get("id") or "") for run in runs if isinstance(run, dict)]
    if any(not value for value in model_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("Model IDs must be unique and non-empty")
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("Run IDs must be unique and non-empty")
    for run in runs:
        if str(run.get("model") or "") not in model_ids:
            raise ValueError(f"Run references an unknown model: {run}")
    comparisons = config.get("comparisons") or []
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list")
    known_runs = set(run_ids)
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError(f"Invalid comparison: {comparison!r}")
        if str(comparison.get("reference_run") or "") not in known_runs:
            raise ValueError(f"Comparison references an unknown reference run: {comparison}")
        if str(comparison.get("candidate_run") or "") not in known_runs:
            raise ValueError(f"Comparison references an unknown candidate run: {comparison}")


def load_official_reference_audit(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official reference audit does not exist: {path}")
    report = read_json(path)
    if report.get("status") != "passed":
        raise ValueError(f"Official reference audit is not passed: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_only": bool(report.get("source_only")),
        "official_source_sha256": str((report.get("source") or {}).get("sha256") or ""),
        "dataset_hashes": {
            str(row.get("name") or ""): str(row.get("project_sha256") or "")
            for row in report.get("datasets") or []
            if isinstance(row, dict)
        },
    }


def validate_task_provenance(tasks: list[dict[str, Any]], protocol_id: str) -> None:
    bad_protocol = []
    target_not_guarded = []
    identifiers = [str(task.get("task_id") or "") for task in tasks]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Execution task IDs must be unique and non-empty")
    for task in tasks:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        identifier = str(task.get("task_id") or "")
        if str(metadata.get("protocol_id") or "") != protocol_id:
            bad_protocol.append(identifier)
        if metadata.get("target_actions_are_evaluation_only") is not True:
            target_not_guarded.append(identifier)
    if bad_protocol or target_not_guarded:
        raise ValueError(
            "Execution task provenance validation failed: "
            f"bad_protocol={len(bad_protocol)}, target_not_evaluation_only={len(target_not_guarded)}"
        )


def model_identity(config: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    adapter = str(model.get("adapter") or "")
    if not adapter:
        return {"kind": "base", "model_id": str(model["id"]), "adapter": ""}
    adapter_dir = Path(adapter).resolve()
    adapter_path = adapter_dir / "adapter_model.safetensors"
    provenance_path = adapter_dir / "papo_training_provenance.json"
    if not adapter_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Execution adapter or provenance is incomplete: {adapter_dir}")
    provenance = read_json(provenance_path)
    if provenance.get("status") != "passed":
        raise ValueError(f"Execution adapter provenance is not passed: {provenance_path}")
    if str(provenance.get("protocol_id") or "") != str(config["protocol_id"]):
        raise ValueError(f"Execution adapter protocol mismatch: {adapter_dir}")
    datasets = [str(value) for value in provenance.get("datasets") or []]
    if not datasets or any("execution" not in value for value in datasets):
        raise ValueError(f"Adapter is not provenanced from execution datasets: {adapter_dir}")
    return {
        "kind": "adapter",
        "model_id": str(model["id"]),
        "adapter": str(adapter_dir),
        "adapter_sha256": sha256_file(adapter_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "protocol_id": provenance["protocol_id"],
        "datasets": datasets,
    }


def align_comparison_components(prepared: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    neighbors: dict[str, set[str]] = {run_id: set() for run_id in prepared}
    for comparison in comparisons:
        left = str(comparison["reference_run"])
        right = str(comparison["candidate_run"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    visited: set[str] = set()
    for start in sorted(neighbors):
        if start in visited or not neighbors[start]:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        visited.update(component)
        task_maps = {
            run_id: {str(task["task_id"]): task for task in prepared[run_id]["tasks"]}
            for run_id in component
        }
        common = set.intersection(*(set(values) for values in task_maps.values()))
        if not common:
            raise ValueError(f"Comparison component has zero shared eligible tasks: {sorted(component)}")
        for run_id in component:
            original = prepared[run_id]["tasks"]
            prepared[run_id]["tasks"] = [task for task in original if str(task["task_id"]) in common]
            prepared[run_id]["condition_manifest"]["comparison_component"] = sorted(component)
            prepared[run_id]["condition_manifest"]["comparison_aligned_tasks"] = len(common)
            prepared[run_id]["condition_manifest"]["excluded_for_comparison_alignment"] = len(original) - len(common)
