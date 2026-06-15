from __future__ import annotations

import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Any

from .io_utils import manifest_identity_matches, read_jsonl, sha256_file, write_json
from .success import load_success_rules


def preflight_manifest(manifest: dict[str, Any], *, check_device_connection: bool = True) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not manifest_identity_matches(manifest):
        errors.append({"issue": "manifest_identity_missing_or_changed"})
    validate_official_reference_binding(manifest.get("official_reference_audit"), errors, warnings)
    task_apps: set[str] = set()
    task_ids: set[str] = set()
    for entry in manifest["runs"]:
        tasks_path = Path(str(entry["tasks_path"]))
        if not tasks_path.is_file() or sha256_file(tasks_path) != entry["tasks_sha256"]:
            errors.append({"run_id": entry["id"], "issue": "task_file_missing_or_changed"})
            continue
        tasks = read_jsonl(tasks_path)
        ids = [str(task.get("task_id") or "") for task in tasks]
        if len(ids) != len(set(ids)):
            errors.append({"run_id": entry["id"], "issue": "duplicate_task_ids"})
        if len(tasks) != int(entry["task_count"]):
            errors.append({"run_id": entry["id"], "issue": "task_count_changed"})
        task_ids.update(ids)
        task_apps.update(str((task.get("input") or {}).get("app") or "") for task in tasks)
        validate_model_identity(entry, errors)
    inference = manifest["inference"]
    prompt_style = str(inference.get("prompt_style") or "training_aligned")
    if prompt_style not in {"training_aligned", "official_reference"}:
        errors.append({"issue": "unsupported_prompt_style", "prompt_style": prompt_style})
    if str(inference.get("backend") or "") == "llamafactory":
        base_model = Path(str(inference.get("base_model") or ""))
        project_root = Path(__file__).resolve().parents[2]
        if not base_model.is_dir():
            errors.append({"issue": "base_model_missing", "path": str(base_model)})
        if not (project_root / "LLaMA-Factory" / "src" / "llamafactory").is_dir():
            errors.append({"issue": "llamafactory_source_missing"})
    elif str(inference.get("backend") or "") != "replay":
        errors.append({"issue": "unsupported_inference_backend", "backend": inference.get("backend")})
    device_info = validate_device(manifest["device"], task_apps, errors, warnings, check_device_connection)
    rules = load_success_rules(manifest.get("success_rules_path"))
    valid_rule_ids: set[str] = set()
    for identifier, rule in rules.items():
        rule_errors = validate_success_rule(identifier, rule)
        if rule_errors:
            errors.extend(rule_errors)
        else:
            valid_rule_ids.add(str(identifier))
    unknown_rules = sorted(set(rules) - task_ids)
    if unknown_rules:
        warnings.append({"issue": "success_rules_for_unknown_tasks", "count": len(unknown_rules), "examples": unknown_rules[:10]})
    covered = task_ids & valid_rule_ids
    missing_rules = sorted(task_ids - valid_rule_ids)
    if missing_rules:
        warnings.append(
            {
                "issue": "tasks_require_manual_success_review",
                "count": len(missing_rules),
                "examples": missing_rules[:10],
            }
        )
    report = {
        "status": "passed" if not errors else "failed",
        "check_device_connection": check_device_connection,
        "runs": len(manifest["runs"]),
        "unique_tasks": len(task_ids),
        "apps": sorted(task_apps),
        "device_info": device_info,
        "success_rule_coverage": {
            "covered": len(covered),
            "missing": len(missing_rules),
            "fraction": len(covered) / len(task_ids) if task_ids else 0.0,
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(Path(str(manifest["output_root"])) / "preflight_report.json", report)
    return report


def validate_success_rule(identifier: str, rule: Any) -> list[dict[str, Any]]:
    if not isinstance(rule, dict):
        return [{"issue": "invalid_success_rule", "task_id": identifier, "reason": "rule_not_object"}]
    errors: list[dict[str, Any]] = []
    if not str(rule.get("source") or "").strip():
        errors.append({"issue": "invalid_success_rule", "task_id": identifier, "reason": "missing_source"})
    predicates = ("xml_contains_all", "xml_contains_any", "xml_not_contains")
    has_predicate = False
    for name in predicates:
        values = rule.get(name)
        if values is None:
            continue
        if not isinstance(values, list) or any(not str(value).strip() for value in values):
            errors.append({"issue": "invalid_success_rule", "task_id": identifier, "reason": f"invalid_{name}"})
        elif values:
            has_predicate = True
    if not has_predicate:
        errors.append({"issue": "invalid_success_rule", "task_id": identifier, "reason": "missing_predicate"})
    return errors


def validate_official_reference_binding(
    binding: Any,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    if not isinstance(binding, dict) or not binding:
        warnings.append({"issue": "official_reference_audit_not_bound"})
        return
    path = Path(str(binding.get("path") or ""))
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        errors.append({"issue": "official_reference_audit_missing_or_changed"})
        return
    if binding.get("source_only"):
        warnings.append({"issue": "official_reference_audit_is_source_only"})


def validate_model_identity(entry: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    identity = entry["model_identity"]
    if identity.get("kind") != "adapter":
        return
    adapter_dir = Path(str(identity["adapter"]))
    adapter_path = adapter_dir / "adapter_model.safetensors"
    provenance_path = Path(str(identity["provenance_path"]))
    if not adapter_path.is_file() or sha256_file(adapter_path) != identity["adapter_sha256"]:
        errors.append({"run_id": entry["id"], "issue": "adapter_missing_or_changed"})
    if not provenance_path.is_file() or sha256_file(provenance_path) != identity["provenance_sha256"]:
        errors.append({"run_id": entry["id"], "issue": "adapter_provenance_missing_or_changed"})


def validate_device(
    device: dict[str, Any],
    task_apps: set[str],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    check_connection: bool,
) -> dict[str, Any]:
    backend = str(device.get("backend") or "")
    if backend == "replay":
        return {"backend": "replay"}
    if backend != "adb":
        errors.append({"issue": "unsupported_device_backend", "backend": backend})
        return {"backend": backend}
    text_mode = str(device.get("text_input_mode") or "adb_keyboard")
    hierarchy_backend = str(device.get("hierarchy_backend") or "adb_uiautomator_dump")
    if text_mode not in {"adb_keyboard", "input_text", "uiautomator2"}:
        errors.append({"issue": "unsupported_text_input_mode", "mode": text_mode})
    if hierarchy_backend not in {"adb_uiautomator_dump", "uiautomator2"}:
        errors.append({"issue": "unsupported_hierarchy_backend", "backend": hierarchy_backend})
    if "uiautomator2" in {text_mode, hierarchy_backend} and importlib.util.find_spec("uiautomator2") is None:
        errors.append({"issue": "uiautomator2_missing"})
    hooks = device.get("app_hooks") if isinstance(device.get("app_hooks"), dict) else {}
    missing_hooks = sorted(app for app in task_apps if app not in hooks)
    if device.get("require_app_hook", True) and missing_hooks:
        errors.append({"issue": "missing_required_app_hooks", "count": len(missing_hooks), "apps": missing_hooks})
    elif missing_hooks:
        warnings.append({"issue": "missing_optional_app_hooks", "count": len(missing_hooks), "apps": missing_hooks})
    for app, hook in hooks.items():
        if not isinstance(hook, dict):
            errors.append({"issue": "invalid_app_hook", "app": app})
            continue
        if device.get("require_app_hook", True) and not hook.get("before_task"):
            errors.append({"issue": "empty_required_before_task_hook", "app": app})
        for phase in ("before_task", "after_task"):
            commands = hook.get(phase) or []
            if not isinstance(commands, list) or any(
                not isinstance(command, list)
                or not command
                or any(not isinstance(part, str) for part in command)
                for command in commands
            ):
                errors.append({"issue": "invalid_app_hook_commands", "app": app, "phase": phase})
    adb_value = str(device.get("adb_path") or "adb")
    adb_path = shutil.which(adb_value) or (adb_value if Path(adb_value).is_file() else "")
    if not adb_path:
        errors.append({"issue": "adb_executable_missing", "path": adb_value})
        return {"backend": "adb", "adb_path": adb_value}
    info: dict[str, Any] = {"backend": "adb", "adb_path": adb_path, "serial": str(device.get("serial") or "")}
    if check_connection:
        serial = str(device.get("serial") or "")
        try:
            state = run_adb_command(adb_path, serial, ["get-state"]).strip()
            info["state"] = state
            if state != "device":
                errors.append({"issue": "adb_device_not_ready", "state": state})
            properties = {
                "manufacturer": "ro.product.manufacturer",
                "model": "ro.product.model",
                "device": "ro.product.device",
                "android_version": "ro.build.version.release",
                "build_fingerprint": "ro.build.fingerprint",
            }
            for name, prop in properties.items():
                info[name] = run_adb_command(adb_path, serial, ["shell", "getprop", prop]).strip()
            info["wm_size"] = run_adb_command(adb_path, serial, ["shell", "wm", "size"]).strip()
            info["wm_density"] = run_adb_command(adb_path, serial, ["shell", "wm", "density"]).strip()
            installed = {}
            for app in sorted(task_apps):
                package_path = run_adb_command(adb_path, serial, ["shell", "pm", "path", app]).strip()
                installed[app] = package_path
                if not package_path:
                    errors.append({"issue": "task_app_not_installed", "app": app})
            info["task_app_package_paths"] = installed
        except (subprocess.SubprocessError, OSError) as error:
            errors.append({"issue": "adb_connection_failed", "error": f"{type(error).__name__}: {error}"})
    return info


def run_adb_command(adb_path: str, serial: str, arguments: list[str]) -> str:
    command = [adb_path]
    if serial:
        command.extend(["-s", serial])
    command.extend(arguments)
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    return completed.stdout
