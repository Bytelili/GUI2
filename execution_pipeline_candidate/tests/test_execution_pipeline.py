from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epipeline.actions import OFFICIAL_ACTION_SEPARATOR, official_action_text, official_execution_similarity, parse_action  # noqa: E402
from epipeline.analysis import analyze_manifest  # noqa: E402
from epipeline.audit import audit_manifest  # noqa: E402
from epipeline.backends import Prediction  # noqa: E402
from epipeline.conditions import apply_condition  # noqa: E402
from epipeline.devices import AdbDevice, Observation  # noqa: E402
from epipeline.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from epipeline.preflight import preflight_manifest, validate_device  # noqa: E402
from epipeline.prepare import prepare_experiment  # noqa: E402
from epipeline.prompts import build_prompt  # noqa: E402
from epipeline.runner import execute_task, run_entry  # noqa: E402
from epipeline.runtime_templates import build_runtime_templates  # noqa: E402
from epipeline.scoring import score_manifest, score_result  # noqa: E402
from epipeline.success import verify_success  # noqa: E402


class ExecutionPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.xml = self.root / "screen.xml"
        self.xml.write_text(
            '<hierarchy><node text="done" class="android.widget.TextView" bounds="[0,0][100,100]" /></hierarchy>',
            encoding="utf-8",
        )
        self.image = self.root / "screen.png"
        self.image.write_bytes(b"not-a-real-image")
        self.tasks = self.root / "tasks.jsonl"
        write_jsonl(self.tasks, [self.task("execution__1__20250103", "1", "30")])
        self.rules = self.root / "rules.json"
        write_json(
            self.rules,
            {
                "rules": {
                    "execution__1__20250103": {
                        "source": "test rule",
                        "xml_contains_all": ["done"],
                    }
                }
            },
        )
        self.output = self.root / "output"
        self.config = self.root / "config.json"
        write_json(
            self.config,
            {
                "schema_version": 1,
                "experiment_id": "test",
                "protocol_id": "strict-test",
                "tasks_path": str(self.tasks),
                "output_root": str(self.output),
                "seed": 42,
                "max_steps": 10,
                "max_invalid_actions": 2,
                "inference": {"backend": "replay"},
                "device": {"backend": "replay"},
                "success_rules_path": str(self.rules),
                "models": [{"id": "base", "adapter": ""}],
                "runs": [
                    {
                        "id": "base__correct_full",
                        "model": "base",
                        "condition": "correct_full_history",
                        "required": True,
                    },
                    {
                        "id": "base__no_history",
                        "model": "base",
                        "condition": "no_history",
                        "required": True,
                    },
                ],
                "comparisons": [
                    {
                        "id": "full_vs_none",
                        "reference_run": "base__no_history",
                        "candidate_run": "base__correct_full",
                    }
                ],
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_action_parser_is_strict(self) -> None:
        self.assertTrue(parse_action("click(coordinates=(1,2), content='x')").valid)
        self.assertTrue(parse_action("type(text='hello')").valid)
        self.assertFalse(parse_action("click()").valid)
        self.assertFalse(parse_action("wait() finished()").valid)

    def test_prompt_matches_training_shape_and_hides_target(self) -> None:
        task = self.task("execution__1__20250103", "1", "30")
        task["target"]["secret"] = "DO_NOT_LEAK"
        prompt = build_prompt(task, str(self.image), str(self.xml), [])
        self.assertTrue(prompt.startswith("<image>Predict exactly one next Android action"))
        self.assertIn("Relevant same-user reference actions", prompt)
        self.assertNotIn("DO_NOT_LEAK", prompt)
        official_prompt = build_prompt(task, str(self.image), str(self.xml), [], style="official_reference")
        self.assertIn("Screen_width_height:", official_prompt)
        self.assertIn("Actions_reference:", official_prompt)
        self.assertNotIn("DO_NOT_LEAK", official_prompt)

    def test_replay_pipeline_is_resumable_and_not_paper_eligible(self) -> None:
        manifest = prepare_experiment(self.config)
        self.assertEqual(len(manifest["runs"]), 2)
        preflight = preflight_manifest(manifest)
        self.assertEqual(preflight["status"], "passed")
        self.assertEqual(preflight["success_rule_coverage"]["fraction"], 1.0)
        for entry in manifest["runs"]:
            report = run_entry(manifest, entry)
            self.assertEqual(report["completed"], 1)
            self.assertFalse(report["model_paper_eligible"])
            first = read_jsonl(Path(entry["run_dir"]) / "raw_results.jsonl")
            run_entry(manifest, entry)
            second = read_jsonl(Path(entry["run_dir"]) / "raw_results.jsonl")
            self.assertEqual(first, second)
        score = score_manifest(manifest)
        self.assertTrue(all(not run["paper_eligible"] for run in score["runs"]))
        analysis = analyze_manifest(manifest, bootstrap_samples=200)
        self.assertEqual(len(analysis["paired_comparisons"]), 10)
        self.assertTrue(all(row["user_clusters"] == 1 for row in analysis["paired_comparisons"]))
        self.assertTrue(
            all(row["macro_user_ci95_low"] == row["macro_user_ci95_high"] for row in analysis["paired_comparisons"])
        )
        audit = audit_manifest(manifest, require_paper_eligible=False)
        self.assertEqual(audit["status"], "passed")
        strict_audit = audit_manifest(manifest, require_paper_eligible=True)
        self.assertEqual(strict_audit["status"], "failed")
        self.assertIn("official_reference_audit_not_bound", {row["issue"] for row in strict_audit["issues"]})

    def test_no_history_hides_but_preserves_evaluation_cross_reference(self) -> None:
        manifest = prepare_experiment(self.config)
        entry = next(row for row in manifest["runs"] if row["condition"] == "no_history")
        task = read_jsonl(entry["tasks_path"])[0]
        self.assertEqual(task["input"]["cross_user_action_references"], [])
        self.assertEqual(task["metadata"]["evaluation_cross_user_actions"], ["press_back()"])
        prompt = build_prompt(task, str(self.image), str(self.xml), [])
        self.assertNotIn("press_back()", prompt)

    def test_audit_rejects_duplicate_results_and_retryable_failures(self) -> None:
        manifest = prepare_experiment(self.config)
        for entry in manifest["runs"]:
            run_entry(manifest, entry)
        score_manifest(manifest)
        entry = manifest["runs"][0]
        raw_path = Path(entry["run_dir"]) / "raw_results.jsonl"
        rows = read_jsonl(raw_path)
        write_jsonl(raw_path, rows + rows)
        report_path = Path(entry["run_dir"]) / "run_report.json"
        report = read_json(report_path)
        report["failed"] = 1
        report["raw_results_sha256"] = sha256_file(raw_path)
        write_json(report_path, report)
        audit = audit_manifest(manifest, require_paper_eligible=False)
        issues = {row["issue"] for row in audit["issues"] if row.get("run_id") == entry["id"]}
        self.assertIn("duplicate_raw_task_ids", issues)
        self.assertIn("retryable_failures_present", issues)

    def test_prepare_rejects_duplicate_task_ids(self) -> None:
        task = self.task("execution__1__20250103", "1", "30")
        write_jsonl(self.tasks, [task, task])
        with self.assertRaisesRegex(ValueError, "unique and non-empty"):
            prepare_experiment(self.config)

    def test_preflight_rejects_changed_task_file(self) -> None:
        manifest = prepare_experiment(self.config)
        entry = manifest["runs"][0]
        write_jsonl(entry["tasks_path"], [])
        report = preflight_manifest(manifest)
        self.assertEqual(report["status"], "failed")
        self.assertIn("task_file_missing_or_changed", {row["issue"] for row in report["errors"]})

    def test_manifest_change_is_rejected_across_pipeline(self) -> None:
        manifest = prepare_experiment(self.config)
        manifest["max_steps"] = 999
        preflight = preflight_manifest(manifest)
        self.assertEqual(preflight["status"], "failed")
        self.assertIn("manifest_identity_missing_or_changed", {row["issue"] for row in preflight["errors"]})
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            run_entry(manifest, manifest["runs"][0])
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            score_manifest(manifest)

    def test_empty_success_rule_is_never_verified(self) -> None:
        task = self.task("execution__1__20250103", "1", "30")
        result = verify_success(task, str(self.xml), {task["task_id"]: {"source": "empty"}})
        self.assertFalse(result["success_verified"])
        manifest = prepare_experiment(self.config)
        write_json(self.rules, {"rules": {task["task_id"]: {"source": "empty"}}})
        preflight = preflight_manifest(manifest)
        self.assertEqual(preflight["status"], "failed")
        self.assertIn("invalid_success_rule", {row["issue"] for row in preflight["errors"]})

    def test_score_is_bound_to_raw_result_hash(self) -> None:
        manifest = prepare_experiment(self.config)
        for entry in manifest["runs"]:
            run_entry(manifest, entry)
        score_manifest(manifest)
        entry = manifest["runs"][0]
        raw_path = Path(entry["run_dir"]) / "raw_results.jsonl"
        rows = read_jsonl(raw_path)
        rows[0]["time"] = 999.0
        write_jsonl(raw_path, rows)
        audit = audit_manifest(manifest, require_paper_eligible=False)
        issues = {row["issue"] for row in audit["issues"] if row.get("run_id") == entry["id"]}
        self.assertIn("raw_result_hash_missing_or_changed", issues)
        self.assertIn("score_not_bound_to_raw_results", issues)

    def test_unknown_manual_annotation_is_rejected(self) -> None:
        manifest = prepare_experiment(self.config)
        for entry in manifest["runs"]:
            run_entry(manifest, entry)
        annotations = self.root / "unknown_annotations.csv"
        with annotations.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["run_id", "task_id", "success", "annotator", "evidence"])
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": "unknown-run",
                    "task_id": "unknown-task",
                    "success": "true",
                    "annotator": "tester",
                    "evidence": "reviewed screenshot",
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown run/task"):
            score_manifest(manifest, annotations)

    def test_official_similarity_matches_reference_formula(self) -> None:
        agent = ["click(coordinates=(1,2), content='done')"]
        golden = ["click(coordinates=(1,2), content='done')"]
        up_sim, down_sim, similarity = official_execution_similarity(agent, golden, [])
        self.assertTrue(official_action_text(agent, predicted=True).endswith(OFFICIAL_ACTION_SEPARATOR))
        self.assertEqual(up_sim, 0.99)
        self.assertEqual(down_sim, 0.4)
        self.assertEqual(similarity, up_sim / 0.4)

    def test_adb_scroll_direction_matches_official_reference(self) -> None:
        device = object.__new__(AdbDevice)
        device.action_delay = 0.0
        device.wait_delay = 0.0
        device.long_click_duration = 500
        device.scroll_distance = 500
        device.scroll_duration = 300
        calls = []
        device._run = lambda arguments, binary=False: calls.append(arguments) or ""
        device.act(parse_action("scroll(coordinates=(100,800), direction='down')"))
        self.assertEqual(calls[-1], ["shell", "input", "swipe", "100", "800", "100", "300", "300"])

    def test_official_wait_preserves_raw_model_output(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.outputs = ["not an action", "finished()"]
                self.histories = []

            def predict(self, task, *, screenshot, xml_path, previous_actions):
                del task, screenshot, xml_path
                self.histories.append(list(previous_actions))
                raw = self.outputs.pop(0)
                return Prediction(raw, parse_action(raw), 0.0, 0, 0)

        class Device:
            def __init__(self, image, xml) -> None:
                self.image = image
                self.xml = xml
                self.actions = []

            def reset(self, task, task_dir):
                del task, task_dir
                return Observation(str(self.image), str(self.xml), 0)

            def act(self, action):
                self.actions.append(action.raw)

            def observe(self, task_dir, index):
                del task_dir
                return Observation(str(self.image), str(self.xml), index)

            def close_task(self, task):
                del task

        model = Model()
        device = Device(self.image, self.xml)
        result = execute_task(
            self.task("execution__1__20250103", "1", "30"),
            model=model,
            device=device,
            rules={},
            task_dir=self.root / "official-wait",
            max_steps=10,
            max_invalid_actions=1,
            invalid_action_policy="official_wait",
            run_id="run",
            model_id="model",
            condition="correct_full_history",
        )
        self.assertEqual(result["official_agent_outputs"], ["not an action", "finished()"])
        self.assertEqual(result["agent_actions"], ["wait()", "finished()"])
        self.assertEqual(device.actions, ["wait()"])
        self.assertEqual(model.histories[1], ["not an action"])
        self.assertEqual(result["real_step"], 2)
        scored = score_result(result, None)
        self.assertEqual(scored["real_step"], 2)
        self.assertIn("not an action", scored["official_agent_outputs"])

    def test_preflight_rejects_empty_required_app_hook(self) -> None:
        errors = []
        warnings = []
        validate_device(
            {
                "backend": "adb",
                "adb_path": sys.executable,
                "require_app_hook": True,
                "app_hooks": {"example.app": {"before_task": [], "after_task": []}},
            },
            {"example.app"},
            errors,
            warnings,
            False,
        )
        self.assertIn("empty_required_before_task_hook", {row["issue"] for row in errors})

    def test_runtime_templates_cover_apps_and_require_rule_review(self) -> None:
        output = self.root / "templates"
        report = build_runtime_templates(self.tasks, output)
        self.assertEqual(report["status"], "passed")
        hooks = read_json(output / "app_hooks.template.json")["app_hooks"]
        rules = read_json(output / "success_rules.template.json")["rules"]
        self.assertIn("example.app", hooks)
        self.assertTrue(hooks["example.app"]["before_task"])
        self.assertEqual(set(rules), {"execution__1__20250103"})
        self.assertEqual(rules["execution__1__20250103"]["xml_contains_all"], [])

    def test_same_official_type_cross_reference_is_rejected(self) -> None:
        task = self.task("execution__1__20250103", "1", "9")
        with self.assertRaisesRegex(ValueError, "official different type"):
            apply_condition([task], "correct_full_history", seed=42)

    def test_comparison_components_are_aligned_to_identical_task_ids(self) -> None:
        first = self.task("execution__1__20250103", "1", "30")
        first["input"]["same_user_action_references"].append(
            {"user_id": "1", "time": "20241231", "actions": ["press_home()"]}
        )
        second = self.task("execution__2__20250104", "2", "30")
        second["input"]["same_user_action_references"] = []
        write_jsonl(self.tasks, [first, second])
        config = read_json(self.config)
        config["runs"] = [
            {"id": "recent", "model": "base", "condition": "correct_recent_history", "required": True},
            {"id": "none", "model": "base", "condition": "no_history", "required": True},
            {"id": "stale", "model": "base", "condition": "stale_history", "required": True},
        ]
        config["comparisons"] = [
            {"id": "recent_vs_none", "reference_run": "none", "candidate_run": "recent"},
            {"id": "recent_vs_stale", "reference_run": "stale", "candidate_run": "recent"},
        ]
        write_json(self.config, config)
        manifest = prepare_experiment(self.config)
        task_sets = [
            {row["task_id"] for row in read_jsonl(entry["tasks_path"])}
            for entry in manifest["runs"]
        ]
        self.assertEqual(task_sets, [{"execution__1__20250103"}] * 3)

    def test_history_ablation_conditions_are_distinct(self) -> None:
        task = self.task("execution__1__20250103", "1", "30")
        task["input"]["time"] = "20250105"
        task["input"]["same_user_action_references"] = [
            {"user_id": "1", "time": f"2025010{index}", "actions": [f"action-{index}"]}
            for index in range(1, 5)
        ]
        full, _ = apply_condition([task], "correct_full_history", seed=42)
        recent, _ = apply_condition([task], "correct_recent_history", seed=42)
        truncated, _ = apply_condition([task], "truncated_history", seed=42)
        stale, _ = apply_condition([task], "stale_history", seed=42)
        self.assertEqual(len(full[0]["input"]["same_user_action_references"]), 4)
        self.assertEqual(len(recent[0]["input"]["same_user_action_references"]), 1)
        self.assertEqual(len(truncated[0]["input"]["same_user_action_references"]), 2)
        self.assertEqual(stale[0]["input"]["same_user_action_references"][0]["time"], "20250101")

    def test_prepare_rejects_wrong_task_protocol(self) -> None:
        task = self.task("execution__1__20250103", "1", "30")
        task["metadata"]["protocol_id"] = "wrong"
        write_jsonl(self.tasks, [task])
        with self.assertRaisesRegex(ValueError, "bad_protocol=1"):
            prepare_experiment(self.config)

    def test_manual_annotation_requires_evidence(self) -> None:
        manifest = prepare_experiment(self.config)
        entry = manifest["runs"][0]
        run_entry(manifest, entry)
        raw_path = Path(entry["run_dir"]) / "raw_results.jsonl"
        raw = read_jsonl(raw_path)
        raw[0]["success_verified"] = False
        raw[0]["success"] = None
        write_jsonl(raw_path, raw)
        report_path = Path(entry["run_dir"]) / "run_report.json"
        report = read_json(report_path)
        report["raw_results_sha256"] = sha256_file(raw_path)
        write_json(report_path, report)
        annotations = self.root / "annotations.csv"
        with annotations.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["run_id", "task_id", "success", "annotator", "evidence"])
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": entry["id"],
                    "task_id": raw[0]["task_id"],
                    "success": "true",
                    "annotator": "",
                    "evidence": "",
                }
            )
        with self.assertRaisesRegex(ValueError, "requires annotator and evidence"):
            score_manifest(manifest, annotations)

    def task(self, identifier: str, user: str, cross_user: str) -> dict:
        return {
            "task_id": identifier,
            "task_type": "personalized_execution",
            "input": {
                "user_id": user,
                "time": "20250103",
                "scenario": "home",
                "app": "example.app",
                "instruction": "complete task",
                "user_profile": {"age": "20"},
                "initial_screenshot": str(self.image),
                "initial_xml": str(self.xml),
                "same_user_action_references": [
                    {"user_id": user, "time": "20250101", "actions": ["click(coordinates=(1,2), content='done')"]}
                ],
                "cross_user_action_references": [
                    {"user_id": cross_user, "time": "20250101", "actions": ["press_back()"]}
                ],
            },
            "target": {
                "actions": [
                    "click(coordinates=(1,2), content='done')",
                    "finished()",
                ]
            },
            "metadata": {
                "papo_episode_id": identifier.replace("execution__", ""),
                "protocol_id": "strict-test",
                "partition": "official_test",
                "target_actions_are_evaluation_only": True,
                "cross_user_reference_is_different_age_group_counterfactual": True,
            },
        }


if __name__ == "__main__":
    unittest.main()
