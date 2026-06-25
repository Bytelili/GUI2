from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
from papo.io import read_jsonl  # noqa: E402
from papo.llamafactory_export import (  # noqa: E402
    attach_prior_actions,
    export_execution_dpo,
    export_execution_listwise,
    export_execution_sft,
    export_proactive_sft,
)
from papo.official_data import read_csv_rows  # noqa: E402


PROACTIVE_DATASET = "papo_proactive_sft"
PROACTIVE_FIXED_SFT_DATASETS = {
    "papo_proactive_oracle_sft_train",
    "papo_proactive_rerank_train",
}
PROACTIVE_FIXED_DPO_DATASETS = {
    "papo_proactive_dpo_train",
}
PROACTIVE_FIXED_LISTWISE_DATASETS = {
    "papo_proactive_weighted_listwise_train",
}
PROACTIVE_FIXED_DATASETS = (
    PROACTIVE_FIXED_SFT_DATASETS
    | PROACTIVE_FIXED_DPO_DATASETS
    | PROACTIVE_FIXED_LISTWISE_DATASETS
)
EXECUTION_DATASETS = {
    "papo_execution_sft",
    "papo_execution_listwise",
    "papo_execution_dpo",
}
CONTAMINATION_MARKERS = {
    "CONTAMINATED_HISTORY_DO_NOT_USE.txt",
    "CONTAMINATED_DO_NOT_USE.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Block formal training unless provenance checks pass.")
    parser.add_argument("--project-config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--require-approval", action="store_true")
    args = parser.parse_args()

    project = load_config(args.project_config)
    train_path = Path(args.train_config).resolve()
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    dataset = str(train.get("dataset") or "")
    report_path = Path(args.report).resolve() if args.report else _default_report(dataset)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    audit = Audit(project, train_path, train, dataset, args.sample_count)
    try:
        audit.run()
    except Exception as exc:
        audit.check("preflight_internal_error", False, f"{type(exc).__name__}: {exc}")
    report = audit.report()

    approval_path = _approval_path(dataset)
    if args.approve and report["passed"]:
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "approved_data_hashes": report["data_hashes"],
                    "approved_report": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report["approval"] = {"status": "created", "path": str(approval_path)}

    if args.require_approval:
        _verify_approval(report, approval_path, audit)

    report["passed"] = not audit.failures
    report["failures"] = audit.failures
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report, report_path)
    if not report["passed"]:
        raise SystemExit(1)


class Audit:
    def __init__(
        self,
        project: dict[str, Any],
        train_path: Path,
        train: dict[str, Any],
        dataset: str,
        sample_count: int,
    ) -> None:
        self.project = project
        self.train_path = train_path
        self.train = train
        self.dataset = dataset
        self.sample_count = sample_count
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}
        self.samples: list[dict[str, Any]] = []
        self.data_paths: set[Path] = set()

    def run(self) -> None:
        self._check_supported_dataset()
        self._check_training_config()
        if self.dataset == PROACTIVE_DATASET:
            self._audit_proactive()
        elif self.dataset in PROACTIVE_FIXED_DATASETS:
            self._audit_proactive_fixed()
        elif self.dataset in EXECUTION_DATASETS:
            self._audit_execution()
        self._check_contaminated_dependencies()
        self._check_resume_manifest()

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "train_config": str(self.train_path),
            "passed": not self.failures,
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": self.checks,
            "stats": self.stats,
            "samples": self.samples,
            "data_hashes": {
                str(path): _sha256(path)
                for path in sorted(self.data_paths, key=str)
                if path.is_file()
            },
        }

    def check(self, name: str, passed: bool, details: Any) -> None:
        self.checks.append({"name": name, "passed": passed, "details": details})
        if not passed:
            self.failures.append(f"{name}: {details}")

    def warn(self, name: str, details: Any) -> None:
        self.checks.append({"name": name, "passed": True, "warning": True, "details": details})
        self.warnings.append(f"{name}: {details}")

    def _check_supported_dataset(self) -> None:
        supported = {PROACTIVE_DATASET, *EXECUTION_DATASETS, *PROACTIVE_FIXED_DATASETS}
        self.check("supported_dataset", self.dataset in supported, self.dataset)

    def _check_training_config(self) -> None:
        required = [
            "model_name_or_path",
            "dataset_dir",
            "dataset",
            "output_dir",
            "eval_strategy",
            "eval_steps",
            "save_strategy",
            "save_steps",
            "seed",
            "data_seed",
        ]
        missing = [key for key in required if self.train.get(key) is None]
        self.check("training_config_required_fields", not missing, {"missing": missing})
        self.check(
            "step_evaluation_enabled",
            self.train.get("eval_strategy") == "steps" and int(self.train.get("eval_steps", 0)) > 0,
            {"eval_strategy": self.train.get("eval_strategy"), "eval_steps": self.train.get("eval_steps")},
        )
        self.check(
            "every_evaluation_checkpoint_saved",
            self.train.get("save_strategy") == "steps"
            and int(self.train.get("save_steps", -1)) == int(self.train.get("eval_steps", -2)),
            {"save_steps": self.train.get("save_steps"), "eval_steps": self.train.get("eval_steps")},
        )
        save_total_limit = self.train.get("save_total_limit")
        output_dir = str(self.train.get("output_dir") or "")
        clean_v3_run = "clean_v3" in output_dir
        allow_small_limit = clean_v3_run and save_total_limit in (1, 2, 3)
        self.check(
            "best_checkpoint_cannot_be_pruned",
            save_total_limit in (None, 0) or allow_small_limit,
            {
                "save_total_limit": save_total_limit,
                "output_dir": output_dir,
                "policy": "clean_v3_allow_small_limit" if allow_small_limit else "strict_no_prune",
            },
        )
        has_eval_dataset = bool(self.train.get("eval_dataset"))
        has_val_size = 0.0 < float(self.train.get("val_size", 0.0)) < 1.0
        self.check(
            "validation_enabled",
            has_eval_dataset or has_val_size,
            {"val_size": self.train.get("val_size"), "eval_dataset": self.train.get("eval_dataset")},
        )
        self.data_paths.add(self.train_path)

    def _audit_proactive(self) -> None:
        data = self.project["data"]
        self.check(
            "proactive_target_split_is_train_only",
            data.get("suggestion_split") == "train_set.csv",
            {"suggestion_split": data.get("suggestion_split")},
        )
        self.check(
            "proactive_history_split_is_train_only",
            data.get("suggestion_history_split") == "train_set.csv",
            {"suggestion_history_split": data.get("suggestion_history_split")},
        )
        official = config_path(self.project, "paths.official_root")
        train_path = official / "train_set.csv"
        suggestion_test_path = official / "test_suggestion.csv"
        execution_test_path = official / "test_execution.csv"
        task_path = config_path(self.project, "paths.task_dir") / "proactive_config.jsonl"
        dataset_path = _dataset_path(self.train, self.dataset)
        for path in [train_path, suggestion_test_path, execution_test_path, task_path, dataset_path]:
            self.data_paths.add(path)
        self.check("proactive_artifacts_exist", all(path.is_file() for path in [task_path, dataset_path]), {
            "tasks": str(task_path),
            "dataset": str(dataset_path),
        })
        if not task_path.is_file() or not dataset_path.is_file():
            return

        train_keys = _row_keys(read_csv_rows(train_path))
        suggestion_test_keys = _row_keys(read_csv_rows(suggestion_test_path))
        execution_test_keys = _row_keys(read_csv_rows(execution_test_path))
        tasks = read_jsonl(task_path)
        target_keys: set[tuple[str, str]] = set()
        history_keys: list[tuple[str, str]] = []
        future = 0
        wrong_metadata = 0
        repeated_target_history = 0
        for task in tasks:
            inputs = task.get("input", {})
            metadata = task.get("metadata", {})
            target = task.get("target", {})
            target_key = _key(inputs)
            target_keys.add(target_key)
            if metadata.get("target_split") != "train_set.csv" or metadata.get("history_split") != "train_set.csv":
                wrong_metadata += 1
            histories = inputs.get("previous_intents", [])
            for history in histories:
                history_key = _key(history)
                history_keys.append(history_key)
                future += int(history_key[1] >= target_key[1])
                repeated_target_history += int(
                    _intent(history) != "" and _intent(history) == str(target.get("intent") or "").strip()
                )

        outside_train_targets = target_keys - train_keys
        suggestion_test_targets = target_keys & suggestion_test_keys
        history_outside_train = [key for key in history_keys if key not in train_keys]
        history_suggestion_test = [key for key in history_keys if key in suggestion_test_keys]
        history_execution_test = [key for key in history_keys if key in execution_test_keys]
        self.stats["proactive"] = {
            "tasks": len(tasks),
            "unique_targets": len(target_keys),
            "history_entries": len(history_keys),
            "history_length_distribution": Counter(
                len(task.get("input", {}).get("previous_intents", [])) for task in tasks
            ).most_common(),
            "legitimate_repeated_target_history": repeated_target_history,
        }
        self.check("proactive_targets_subset_train", not outside_train_targets, {"count": len(outside_train_targets)})
        self.check("proactive_targets_exclude_test_suggestion", not suggestion_test_targets, {
            "count": len(suggestion_test_targets)
        })
        self.check("proactive_histories_subset_train", not history_outside_train, {
            "count": len(history_outside_train)
        })
        self.check("proactive_histories_exclude_test_suggestion", not history_suggestion_test, {
            "count": len(history_suggestion_test)
        })
        if history_execution_test:
            self.warn("proactive_cross_track_test_execution_overlap", {
                "count": len(history_execution_test),
                "interpretation": "Allowed for Proactive-only training; disclose before any shared/joint model experiment.",
            })
        self.check("proactive_histories_strictly_past", future == 0, {"count": future})
        self.check("proactive_task_provenance_metadata", wrong_metadata == 0, {"count": wrong_metadata})

        exported = json.loads(dataset_path.read_text(encoding="utf-8"))
        regenerated = export_proactive_sft(
            tasks,
            config_path(self.project, "paths.raw_root"),
            str(self.project["paths"]["asset_prefix"]),
        )
        self.check("proactive_export_matches_audited_tasks", exported == regenerated, {
            "exported_rows": len(exported),
            "regenerated_rows": len(regenerated),
        })
        self.samples = [_proactive_sample(task) for task in tasks[: self.sample_count]]

    def _audit_proactive_fixed(self) -> None:
        dataset_path = _dataset_path(self.train, self.dataset)
        eval_dataset = str(self.train.get("eval_dataset") or "")
        eval_path = _dataset_path(self.train, eval_dataset) if eval_dataset else None
        self.data_paths.add(dataset_path)
        if eval_path is not None:
            self.data_paths.add(eval_path)
        self.check(
            "proactive_fixed_datasets_exist",
            dataset_path.is_file() and (eval_path is None or eval_path.is_file()),
            {"dataset": str(dataset_path), "eval_dataset": str(eval_path) if eval_path else ""},
        )
        if not dataset_path.is_file() or (eval_path is not None and not eval_path.is_file()):
            return

        train_rows = _load_rows(dataset_path)
        eval_rows = _load_rows(eval_path) if eval_path is not None else []
        self.stats["proactive_fixed"] = {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_dataset": self.dataset,
            "eval_dataset": eval_dataset,
        }
        self.check("proactive_fixed_train_non_empty", len(train_rows) > 0, {"rows": len(train_rows)})
        if eval_path is not None:
            self.check("proactive_fixed_eval_non_empty", len(eval_rows) > 0, {"rows": len(eval_rows)})

        if self.dataset in PROACTIVE_FIXED_SFT_DATASETS:
            valid = all(
                isinstance(row.get("messages"), list)
                and len(row.get("messages", [])) >= 3
                and str(
                    row.get("messages", [])[-1].get("value")
                    or row.get("messages", [])[-1].get("content")
                    or ""
                ).strip()
                for row in train_rows[: min(len(train_rows), 100)]
            )
            self.check("proactive_fixed_sft_shape", valid, {"sampled_rows": min(len(train_rows), 100)})
        elif self.dataset in PROACTIVE_FIXED_DPO_DATASETS:
            valid = all(
                row.get("chosen")
                and row.get("rejected")
                and str((row.get("chosen") or {}).get("value") or (row.get("chosen") or {}).get("content") or "").strip()
                != str((row.get("rejected") or {}).get("value") or (row.get("rejected") or {}).get("content") or "").strip()
                for row in train_rows[: min(len(train_rows), 100)]
            )
            self.check("proactive_fixed_dpo_shape", valid, {"sampled_rows": min(len(train_rows), 100)})
        elif self.dataset in PROACTIVE_FIXED_LISTWISE_DATASETS:
            valid = all(float(row.get("papo_listwise_weight", 0.0)) > 0.0 for row in train_rows[: min(len(train_rows), 100)])
            self.check("proactive_fixed_listwise_shape", valid, {"sampled_rows": min(len(train_rows), 100)})

        self.samples = [
            {
                "metadata": row.get("metadata", {}),
                "images": row.get("images", []),
                "message_count": len(row.get("messages", []) or row.get("conversations", [])),
            }
            for row in train_rows[: self.sample_count]
        ]

    def _audit_execution(self) -> None:
        data = self.project["data"]
        self.check(
            "execution_target_split_is_train_only",
            data.get("execution_split") == "train_set.csv",
            {"execution_split": data.get("execution_split")},
        )
        self.check(
            "execution_reference_split_is_train_only",
            data.get("execution_reference_split") == "train_set.csv",
            {"execution_reference_split": data.get("execution_reference_split")},
        )
        self.check(
            "execution_raw_retrieval_is_train_only",
            set(data.get("retrieval_splits") or []) == {"train_set.csv"},
            {"retrieval_splits": data.get("retrieval_splits")},
        )
        official = config_path(self.project, "paths.official_root")
        train_path = official / "train_set.csv"
        suggestion_test_path = official / "test_suggestion.csv"
        execution_test_path = official / "test_execution.csv"
        task_path = config_path(self.project, "paths.task_dir") / "execution_config.jsonl"
        dataset_path = _dataset_path(self.train, self.dataset)
        work = config_path(self.project, "paths.work_dir")
        steps_path = work / "papo_steps.jsonl"
        listwise_path = work / "papo_listwise_targets.jsonl"
        pairs_path = work / "papo_dpo_pairs.jsonl"
        required_artifacts = [task_path, dataset_path, steps_path]
        if self.dataset == "papo_execution_listwise":
            required_artifacts.append(listwise_path)
        if self.dataset == "papo_execution_dpo":
            required_artifacts.append(pairs_path)
        for path in [
            train_path,
            suggestion_test_path,
            execution_test_path,
            *required_artifacts,
        ]:
            self.data_paths.add(path)
        self.check("execution_artifacts_exist", all(path.is_file() for path in required_artifacts), {
            "required": [str(path) for path in required_artifacts],
        })
        if not all(path.is_file() for path in required_artifacts):
            return

        train_keys = _row_keys(read_csv_rows(train_path))
        execution_test_keys = _row_keys(read_csv_rows(execution_test_path))
        suggestion_test_keys = _row_keys(read_csv_rows(suggestion_test_path))
        tasks = read_jsonl(task_path)
        steps = read_jsonl(steps_path)
        targets = {_key(task.get("input", {})) for task in tasks}
        references: list[tuple[str, str]] = []
        future = 0
        wrong_metadata = 0
        for task in tasks:
            inputs = task.get("input", {})
            target_key = _key(inputs)
            metadata = task.get("metadata", {})
            if metadata.get("target_split") != "train_set.csv" or metadata.get("reference_split") != "train_set.csv":
                wrong_metadata += 1
            for name in ["same_user_action_references", "cross_user_action_references"]:
                for reference in inputs.get(name, []):
                    ref_key = _key(reference)
                    references.append(ref_key)
                    future += int(ref_key[1] >= target_key[1])

        self.stats["execution"] = {
            "tasks": len(tasks),
            "unique_targets": len(targets),
            "reference_entries": len(references),
        }
        self.check("execution_targets_subset_train", targets <= train_keys, {"outside_count": len(targets - train_keys)})
        self.check("execution_targets_exclude_test_execution", not (targets & execution_test_keys), {
            "count": len(targets & execution_test_keys)
        })
        if targets & suggestion_test_keys:
            self.warn("execution_cross_track_test_suggestion_target_overlap", {
                "count": len(targets & suggestion_test_keys),
                "interpretation": "Allowed for Execution-only training; disclose before any shared/joint model experiment.",
            })
        self.check("execution_references_subset_train", all(key in train_keys for key in references), {
            "outside_count": sum(key not in train_keys for key in references)
        })
        self.check("execution_references_exclude_test_execution", all(key not in execution_test_keys for key in references), {
            "count": sum(key in execution_test_keys for key in references)
        })
        suggestion_reference_overlap = sum(key in suggestion_test_keys for key in references)
        if suggestion_reference_overlap:
            self.warn("execution_cross_track_test_suggestion_reference_overlap", {
                "count": suggestion_reference_overlap,
                "interpretation": "Allowed for Execution-only training; disclose before any shared/joint model experiment.",
            })
        self.check("execution_references_strictly_past", future == 0, {"count": future})
        self.check("execution_task_provenance_metadata", wrong_metadata == 0, {"count": wrong_metadata})
        step_episode_keys = {
            tuple(str(step.get("episode_id") or "").split("__", 1))
            for step in steps
            if "__" in str(step.get("episode_id") or "")
        }
        self.check("execution_raw_steps_subset_train", step_episode_keys <= train_keys, {
            "outside_count": len(step_episode_keys - train_keys)
        })
        self.check("execution_raw_steps_exclude_test_execution", not (step_episode_keys & execution_test_keys), {
            "count": len(step_episode_keys & execution_test_keys)
        })

        attach_prior_actions(tasks, steps)
        exported = json.loads(dataset_path.read_text(encoding="utf-8"))
        raw_root = config_path(self.project, "paths.raw_root")
        asset_prefix = str(self.project["paths"]["asset_prefix"])
        if self.dataset == "papo_execution_sft":
            regenerated = export_execution_sft(tasks, steps, raw_root, asset_prefix)
        elif self.dataset == "papo_execution_listwise":
            regenerated = export_execution_listwise(
                tasks, steps, read_jsonl(listwise_path), raw_root, asset_prefix
            )
        else:
            regenerated = export_execution_dpo(
                tasks, steps, read_jsonl(pairs_path), raw_root, asset_prefix
            )
        self.check("execution_export_matches_audited_artifacts", exported == regenerated, {
            "exported_rows": len(exported),
            "regenerated_rows": len(regenerated),
        })
        self.samples = [_execution_sample(task) for task in tasks[: self.sample_count]]

    def _check_contaminated_dependencies(self) -> None:
        paths = [Path(str(self.train.get("output_dir") or ""))]
        adapter = self.train.get("adapter_name_or_path")
        if adapter:
            paths.append(Path(str(adapter)))
            adapter_model = Path(str(adapter)) / "adapter_model.safetensors"
            if adapter_model.is_file():
                self.data_paths.add(adapter_model)
        contaminated = []
        for path in paths:
            for marker in CONTAMINATION_MARKERS:
                if (path / marker).exists():
                    contaminated.append(str(path / marker))
        self.check("no_contaminated_model_dependency", not contaminated, {"markers": contaminated})
        if self.dataset in {PROACTIVE_DATASET, "papo_execution_sft"}:
            self.check("sft_starts_from_base_model", not adapter, {"adapter_name_or_path": adapter})
        if self.dataset in {
            "papo_execution_listwise",
            "papo_execution_dpo",
            "papo_proactive_dpo_train",
            "papo_proactive_weighted_listwise_train",
        }:
            expected = {
                "papo_execution_listwise": "papo_execution_sft",
                "papo_execution_dpo": "papo_execution_listwise",
                "papo_proactive_dpo_train": "papo_proactive_oracle_sft_train",
                "papo_proactive_weighted_listwise_train": "papo_proactive_oracle_sft_train",
            }[self.dataset]
            manifest = Path(str(adapter or "")) / "TRAINING_PREFLIGHT_PASS.json"
            valid_manifest = False
            manifest_dataset = None
            if manifest.is_file():
                upstream = json.loads(manifest.read_text(encoding="utf-8"))
                manifest_dataset = upstream.get("dataset")
                valid_manifest = bool(upstream.get("passed")) and manifest_dataset == expected
                self.data_paths.add(manifest)
            self.check("upstream_adapter_has_clean_provenance", valid_manifest, {
                "expected_dataset": expected,
                "manifest": str(manifest),
                "manifest_dataset": manifest_dataset,
            })
        if self.dataset in PROACTIVE_FIXED_SFT_DATASETS:
            self.check("sft_starts_from_base_model", not adapter, {"adapter_name_or_path": adapter})

    def _check_resume_manifest(self) -> None:
        output = Path(str(self.train.get("output_dir") or ""))
        manifest = output / "TRAINING_PREFLIGHT_PASS.json"
        if not manifest.is_file():
            self.check("resume_data_manifest", True, {"status": "new_output"})
            return
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        current = {
            str(path): _sha256(path)
            for path in sorted(self.data_paths, key=str)
            if path.is_file() and path != self.train_path
        }
        old = {
            path: digest
            for path, digest in previous.get("data_hashes", {}).items()
            if path != str(self.train_path)
        }
        self.check("resume_data_manifest", current == old, {
            "status": "matched" if current == old else "changed",
            "previous_count": len(old),
            "current_count": len(current),
        })


def _verify_approval(report: dict[str, Any], path: Path, audit: Audit) -> None:
    if not path.is_file():
        audit.check("explicit_preflight_approval", False, {"missing": str(path)})
        return
    approval = json.loads(path.read_text(encoding="utf-8"))
    audit.check(
        "explicit_preflight_approval",
        approval.get("approved_data_hashes") == report.get("data_hashes"),
        {"path": str(path), "hashes_match": approval.get("approved_data_hashes") == report.get("data_hashes")},
    )


def _dataset_path(train: dict[str, Any], dataset: str) -> Path:
    root = Path(str(train["dataset_dir"]))
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    return root / info[dataset]["file_name"]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _row_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {_key(row) for row in rows}


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("user_id") or "").strip(), str(row.get("time") or "").strip())


def _intent(row: dict[str, Any]) -> str:
    return str(row.get("intent") or row.get("intentDescription") or "").strip()


def _proactive_sample(task: dict[str, Any]) -> dict[str, Any]:
    inputs = task.get("input", {})
    histories = inputs.get("previous_intents", [])
    return {
        "task_id": task.get("task_id"),
        "target_key": list(_key(inputs)),
        "target": task.get("target", {}).get("intent"),
        "history_count": len(histories),
        "history_keys": [list(_key(item)) for item in histories],
        "metadata": task.get("metadata", {}),
    }


def _execution_sample(task: dict[str, Any]) -> dict[str, Any]:
    inputs = task.get("input", {})
    refs = [
        item
        for name in ["same_user_action_references", "cross_user_action_references"]
        for item in inputs.get(name, [])
    ]
    return {
        "task_id": task.get("task_id"),
        "target_key": list(_key(inputs)),
        "reference_keys": [list(_key(item)) for item in refs],
        "metadata": task.get("metadata", {}),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_report(dataset: str) -> Path:
    return PROJECT_ROOT / "reports" / "training_preflight" / f"{dataset}_latest.json"


def _approval_path(dataset: str) -> Path:
    return PROJECT_ROOT / "reports" / "training_preflight" / "approved" / f"{dataset}.json"


def _print_summary(report: dict[str, Any], path: Path) -> None:
    print("===== Formal training preflight =====")
    print("Dataset:", report["dataset"])
    for check in report["checks"]:
        status = "WARN" if check.get("warning") else ("PASS" if check["passed"] else "FAIL")
        print(f"[{status}] {check['name']}: {check['details']}")
    print("Report:", path)
    print("Result:", "PASS" if report["passed"] else "FAIL")


if __name__ == "__main__":
    main()
