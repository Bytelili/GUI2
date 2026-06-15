from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from .backends import create_model_backend
from .devices import Observation, create_device_backend
from .io_utils import manifest_identity_matches, read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from .success import load_success_rules, verify_success


def run_entry(manifest: dict[str, Any], entry: dict[str, Any], *, limit: int = 0) -> dict[str, Any]:
    if not manifest_identity_matches(manifest):
        raise ValueError("Experiment manifest identity is missing or changed")
    run_dir = Path(str(entry["run_dir"]))
    tasks_path = Path(str(entry["tasks_path"]))
    if sha256_file(tasks_path) != entry["tasks_sha256"]:
        raise ValueError(f"Run task file changed after preparation: {tasks_path}")
    tasks = read_jsonl(tasks_path)
    if limit > 0:
        tasks = tasks[:limit]
    identity = {
        "experiment_identity_sha256": manifest["identity_sha256"],
        "run_id": entry["id"],
        "model_spec": entry["model_spec"],
        "model_identity": entry["model_identity"],
        "tasks_sha256": entry["tasks_sha256"],
        "inference": manifest["inference"],
        "device": manifest["device"],
        "max_steps": manifest["max_steps"],
        "max_invalid_actions": manifest["max_invalid_actions"],
        "invalid_action_policy": manifest["invalid_action_policy"],
    }
    identity_path = run_dir / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError(f"Refusing stale resume because run identity changed: {identity_path}")
    write_json(identity_path, identity)
    raw_task_dir = run_dir / "raw_tasks"
    raw_task_dir.mkdir(parents=True, exist_ok=True)
    model = create_model_backend(manifest["inference"], entry["model_spec"])
    device = create_device_backend(manifest["device"])
    rules = load_success_rules(manifest.get("success_rules_path"))
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        result_path = raw_task_dir / f"{safe_name(str(task['task_id']))}.json"
        if result_path.exists():
            previous = read_json(result_path)
            if previous.get("status") == "completed":
                results.append(previous)
                continue
        result = execute_task(
            task,
            model=model,
            device=device,
            rules=rules,
            task_dir=raw_task_dir / safe_name(str(task["task_id"])),
            max_steps=int(manifest["max_steps"]),
            max_invalid_actions=int(manifest["max_invalid_actions"]),
            invalid_action_policy=str(manifest["invalid_action_policy"]),
            run_id=str(entry["id"]),
            model_id=str(entry["model"]),
            condition=str(entry["condition"]),
        )
        write_json(result_path, result)
        results.append(result)
        print(
            f"[{index}/{len(tasks)}] {entry['id']} {task['task_id']} "
            f"termination={result['termination_reason']} verified={result['success_verified']}",
            flush=True,
        )
    write_jsonl(run_dir / "raw_results.jsonl", results)
    report = {
        "status": "failed" if any(result.get("status") != "completed" for result in results) else "completed",
        "run_id": entry["id"],
        "tasks": len(tasks),
        "completed": sum(result.get("status") == "completed" for result in results),
        "failed": sum(result.get("status") != "completed" for result in results),
        "verified_success_labels": sum(bool(result.get("success_verified")) for result in results),
        "model_paper_eligible": bool(model.paper_eligible),
        "device_paper_eligible": bool(device.paper_eligible),
        "raw_results_sha256": sha256_file(run_dir / "raw_results.jsonl"),
    }
    write_json(run_dir / "run_report.json", report)
    return report


def execute_task(
    task: dict[str, Any],
    *,
    model: Any,
    device: Any,
    rules: dict[str, Any],
    task_dir: Path,
    max_steps: int,
    max_invalid_actions: int,
    invalid_action_policy: str,
    run_id: str,
    model_id: str,
    condition: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    previous_actions: list[str] = []
    model_outputs: list[str] = []
    trace: list[dict[str, Any]] = []
    invalid_actions = 0
    termination = "step_limit"
    final_observation = Observation("", "", 0)
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    golden_actions = [str(value) for value in target.get("actions") or []]
    official_limit = max(1, math.floor(2.5 * len(golden_actions)))
    hard_limit = min(max_steps, official_limit)
    try:
        observation = device.reset(task, task_dir)
        final_observation = observation
        for step_index in range(hard_limit):
            prompt_history = model_outputs if invalid_action_policy == "official_wait" else previous_actions
            prediction = model.predict(
                task,
                screenshot=observation.screenshot,
                xml_path=observation.xml_path,
                previous_actions=prompt_history,
            )
            step = {
                "step_index": step_index,
                "observation": {
                    "screenshot": observation.screenshot,
                    "xml_path": observation.xml_path,
                },
                "prediction": prediction.raw_text,
                "parsed_action": prediction.action.to_dict(),
                "time": prediction.elapsed_seconds,
                "token": prediction.total_tokens,
                "error": prediction.error,
            }
            trace.append(step)
            model_outputs.append(prediction.raw_text)
            if prediction.error:
                termination = "model_error"
                break
            if not prediction.action.valid:
                invalid_actions += 1
                if invalid_action_policy == "official_wait":
                    wait_action = parse_wait_action()
                    previous_actions.append(wait_action.raw)
                    device.act(wait_action)
                    observation = device.observe(task_dir, step_index + 1)
                    final_observation = observation
                    continue
                if invalid_actions >= max_invalid_actions:
                    termination = "invalid_action_limit"
                    break
                continue
            previous_actions.append(prediction.action.raw)
            if prediction.action.action_type == "finished":
                termination = "finished"
                break
            device.act(prediction.action)
            observation = device.observe(task_dir, step_index + 1)
            final_observation = observation
        else:
            termination = "step_limit"
    except Exception as error:
        termination = "runner_error"
        trace.append({"step_index": len(trace), "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            device.close_task(task)
        except Exception as error:
            termination = "runner_error"
            trace.append({"step_index": len(trace), "error": f"close_task: {type(error).__name__}: {error}"})
    if termination in {"model_error", "runner_error"}:
        success = {
            "success": None,
            "success_verified": False,
            "success_source": "",
            "success_evidence": f"Success cannot be verified after transient failure: {termination}.",
        }
    else:
        success = verify_success(task, final_observation.xml_path, rules)
    return {
        "status": "failed" if termination in {"model_error", "runner_error"} else "completed",
        "run_id": run_id,
        "model_id": model_id,
        "condition": condition,
        "task_id": str(task.get("task_id") or ""),
        "episode_id": str((task.get("metadata") or {}).get("papo_episode_id") or ""),
        "user_id": str((task.get("input") or {}).get("user_id") or ""),
        "app": str((task.get("input") or {}).get("app") or ""),
        "intent": str((task.get("input") or {}).get("instruction") or ""),
        "agent_actions": previous_actions,
        "official_agent_outputs": model_outputs,
        "golden_actions": golden_actions,
        "cross_user_actions": cross_actions(task),
        "origin_step": len(golden_actions),
        "real_step": len(model_outputs),
        "official_step_limit": official_limit,
        "termination_reason": termination,
        "invalid_actions": invalid_actions,
        "invalid_action_policy": invalid_action_policy,
        "time": sum(float(step.get("time") or 0.0) for step in trace),
        "wall_time": time.perf_counter() - started,
        "token": sum(int(step.get("token") or 0) for step in trace),
        "final_observation": {
            "screenshot": final_observation.screenshot,
            "xml_path": final_observation.xml_path,
        },
        **success,
        "trace": trace,
    }


def cross_actions(task: dict[str, Any]) -> list[str]:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    evaluation_actions = metadata.get("evaluation_cross_user_actions")
    if isinstance(evaluation_actions, list):
        return [str(value) for value in evaluation_actions]
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    references = inputs.get("cross_user_action_references")
    references = references if isinstance(references, list) else []
    if not references or not isinstance(references[0], dict):
        return []
    return [str(value) for value in references[0].get("actions") or []]


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean[:120] or sha256_json(value)[:16]


def parse_wait_action() -> Any:
    from .actions import parse_action

    return parse_action("wait()")
