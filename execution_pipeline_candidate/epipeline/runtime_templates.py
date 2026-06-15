from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_file, write_json


PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")


def build_runtime_templates(tasks_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(tasks_path).resolve()
    tasks = read_jsonl(source)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    apps = sorted({str((task.get("input") or {}).get("app") or "") for task in tasks})
    invalid_apps = [app for app in apps if not PACKAGE_PATTERN.fullmatch(app)]
    hooks = {
        app: {
            "before_task": [
                ["shell", "am", "force-stop", app],
                ["shell", "monkey", "-p", app, "-c", "android.intent.category.LAUNCHER", "1"],
            ],
            "after_task": [["shell", "am", "force-stop", app]],
        }
        for app in apps
        if app not in invalid_apps
    }
    rules = {
        str(task["task_id"]): {
            "source": "TODO: human-authored final-state rule with evidence",
            "xml_contains_all": [],
            "xml_contains_any": [],
            "xml_not_contains": [],
            "_review_context": {
                "app": str((task.get("input") or {}).get("app") or ""),
                "instruction": str((task.get("input") or {}).get("instruction") or ""),
            },
        }
        for task in tasks
    }
    hooks_path = output / "app_hooks.template.json"
    rules_path = output / "success_rules.template.json"
    write_json(hooks_path, {"schema_version": 1, "app_hooks": hooks})
    write_json(rules_path, {"schema_version": 1, "rules": rules})
    report = {
        "status": "passed" if tasks and not invalid_apps else "failed",
        "tasks_path": str(source),
        "tasks_sha256": sha256_file(source),
        "tasks": len(tasks),
        "apps": len(apps),
        "valid_package_apps": len(hooks),
        "invalid_package_apps": invalid_apps,
        "app_hooks_template": str(hooks_path),
        "success_rules_template": str(rules_path),
        "warning": "Templates must be reviewed. Empty success predicates are intentionally rejected by preflight.",
    }
    write_json(output / "runtime_template_report.json", report)
    return report
