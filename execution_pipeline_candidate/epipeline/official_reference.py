from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .actions import OFFICIAL_ACTION_SEPARATOR
from .io_utils import sha256_file, write_json


REFERENCE_FILES = [
    "total.csv",
    "train_set.csv",
    "test_execution.csv",
    "test_suggestion.csv",
    "user_profile.csv",
]


def audit_official_reference(
    reference_root: str | Path,
    project_official_root: str | Path,
    output_path: str | Path,
    *,
    source_only: bool = False,
) -> dict[str, Any]:
    reference = Path(reference_root).resolve()
    project = Path(project_official_root).resolve()
    source_path = reference / "personalized_execution.py"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {"path": str(source_path), "exists": source_path.is_file()}
    if source_path.is_file():
        source = source_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        functions = sorted(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        expected_functions = {
            "text_similarity",
            "get_previous_actions",
            "get_different_type_actions",
            "extract_nodes_to_csv",
            "get_prompt",
            "text_parser",
            "main",
        }
        markers = {
            "official_fuzzy_similarity": "fuzz.ratio(text1, text2) / 100",
            "official_down_zero_fallback": "dif_similarity = 0.4",
            "official_sim2_ratio": "final_sim = similarity/dif_similarity",
            "official_2_5x_limit": "step <= 2.5*line_count",
            "official_output_columns": "'up_sim', 'down_sim', 'similarity'",
            "official_uiautomator2": "import uiautomator2 as u2",
            "official_scroll_down_semantics": '"down": (x, y - distance)',
            "official_prompt_screen_size": "Screen_width_height:",
            "official_prompt_actions_reference": "Actions_reference:",
            "official_uiautomator2_text_input": "d.send_keys",
        }
        missing_functions = sorted(expected_functions - set(functions))
        missing_markers = sorted(name for name, marker in markers.items() if marker not in source)
        if missing_functions:
            errors.append({"issue": "official_reference_functions_missing", "values": missing_functions})
        if missing_markers:
            errors.append({"issue": "official_reference_contract_markers_missing", "values": missing_markers})
        action_separator = extract_official_action_separator(tree)
        if action_separator != OFFICIAL_ACTION_SEPARATOR:
            errors.append(
                {
                    "issue": "official_action_separator_mismatch",
                    "expected_codepoints": codepoints(OFFICIAL_ACTION_SEPARATOR),
                    "actual_codepoints": codepoints(action_separator),
                }
            )
        source_report.update(
            {
                "sha256": sha256_file(source_path),
                "functions": functions,
                "classes": classes,
                "contract_markers": {name: marker in source for name, marker in markers.items()},
                "action_separator": action_separator,
                "action_separator_codepoints": codepoints(action_separator),
            }
        )
    else:
        errors.append({"issue": "official_personalized_execution_source_missing"})
    datasets = []
    for name in REFERENCE_FILES:
        reference_path = reference / name
        project_path = project / name
        row = {
            "name": name,
            "reference_path": str(reference_path),
            "project_path": str(project_path),
            "reference_exists": reference_path.is_file(),
            "project_exists": project_path.is_file(),
        }
        if reference_path.is_file():
            row["reference_sha256"] = sha256_file(reference_path)
        if project_path.is_file():
            row["project_sha256"] = sha256_file(project_path)
        row["hash_match"] = bool(
            row["reference_exists"]
            and row["project_exists"]
            and row.get("reference_sha256") == row.get("project_sha256")
        )
        if not row["reference_exists"]:
            errors.append({"issue": "official_reference_dataset_file_missing", "name": name})
        elif not row["project_exists"]:
            target = warnings if source_only else errors
            target.append({"issue": "project_official_dataset_file_missing", "name": name})
        elif not row["hash_match"]:
            target = warnings if source_only else errors
            target.append({"issue": "official_dataset_hash_mismatch", "name": name})
        datasets.append(row)
    sampled = reference / "sampled_test_execution.csv"
    if not sampled.is_file():
        warnings.append({"issue": "official_sampled_execution_split_missing"})
    report = {
        "status": "passed" if not errors else "failed",
        "reference_root": str(reference),
        "project_official_root": str(project),
        "source_only": source_only,
        "source": source_report,
        "datasets": datasets,
        "sampled_test_execution": {
            "path": str(sampled),
            "exists": sampled.is_file(),
            "sha256": sha256_file(sampled) if sampled.is_file() else "",
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_path, report)
    return report


def extract_official_action_separator(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "big_string" for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "join"
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
        ):
            return value.func.value.value
    return ""


def codepoints(value: str) -> list[str]:
    return [f"U+{ord(character):04X}" for character in value]
