from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.papo.proactive_listwise_v4 import (
    PROTOCOL_ID,
    V4ValidationError,
    audit_source_tasks,
    build_groups,
    build_retrieval_candidate_pools,
    build_release,
    create_candidate_requests,
    dataset_info_v4,
    import_candidate_results,
    merge_candidate_shards,
    normalize_text,
    read_jsonl,
    retrieval_pool_map,
    sha256_file,
    verify_release,
    write_json,
    write_jsonl,
)
from src.papo.proactive_manual_review import apply_manual_review, export_manual_review
from src.papo.proactive_quality_gate_v4 import audit_v4_groups

try:
    import jsonschema
except ImportError:
    jsonschema = None


def _task(task_id: str, split: str, user: str, stamp: str, image: Path, target: str) -> dict:
    return {
        "task_id": task_id,
        "task_type": "proactive_suggestion",
        "input": {
            "user_id": user,
            "time": stamp,
            "scenario": "学校",
            "user_profile": {"occupation": "学生"},
            "previous_intents": [
                {
                    "episode_id": f"{user}__20260101_080000",
                    "user_id": user,
                    "time": "20260101_080000",
                    "scenario": "住所",
                    "intent": "打开音乐播放收藏",
                }
            ],
            "initial_screenshots": [str(image)],
        },
        "target": {"intent": target, "app": "synthetic.app", "intent_class": "合成任务"},
        "metadata": {
            "papo_episode_id": f"{user}__{stamp}",
            "history_policy": "same_user_strictly_before_target_time",
            "target_is_hidden_from_input": True,
            "history_episode_ids": [f"{user}__20260101_080000"],
            "partition": split,
            "protocol_id": PROTOCOL_ID,
            "target_split": f"proactive_{split}_targets.csv",
            "history_split": "proactive_history.csv",
        },
    }


def _model_candidates(tasks: list[dict]) -> dict[str, list[str]]:
    return {
        str(task["task_id"]): [f"打开设置，查看网络状态{index}"]
        for index, task in enumerate(tasks)
    }


class ProactiveListwiseV4PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "screen.jpg"
        self.image.write_bytes(b"synthetic-image")
        self.train_path = self.root / "train.jsonl"
        self.eval_path = self.root / "eval.jsonl"
        self.train_tasks = [
            _task("train-1", "train", "u1", "20260201_100000", self.image, "打开闹钟设置十点提醒"),
            _task("train-2", "train", "u2", "20260202_110000", self.image, "打开相机拍摄照片"),
        ]
        self.eval_tasks = [_task("eval-1", "eval", "u1", "20260301_100000", self.image, "打开地图搜索学校")]
        write_jsonl(self.train_path, self.train_tasks)
        write_jsonl(self.eval_path, self.eval_tasks)
        self.workspace = self.root / "workspace"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_source_audit_utf8_counts_causality_and_image_retention(self) -> None:
        manifest = audit_source_tasks(self.train_path, self.eval_path, self.workspace)
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["train"]["line_count"], 2)
        self.assertEqual(manifest["train"]["unique_user_count"], 2)
        self.assertEqual(manifest["train_eval_task_id_overlap_count"], 0)
        bad = self.root / "bad.jsonl"
        bad.write_bytes(b"{\"x\":\"\xff\"}\n")
        with self.assertRaisesRegex(V4ValidationError, "UTF-8"):
            list(read_jsonl(bad))

        missing_task = _task("missing-image", "train", "u3", "20260203_120000", self.root / "none.jpg", "打开日历")
        missing_path = self.root / "missing.jsonl"
        write_jsonl(missing_path, [missing_task])
        write_jsonl(self.eval_path, self.eval_tasks)
        unavailable = audit_source_tasks(missing_path, self.eval_path, self.root / "missing-workspace")
        self.assertEqual(unavailable["status"], "passed_with_unavailable_images")
        report = read_jsonl(self.root / "missing-workspace" / "reports" / "source_unavailable_images.jsonl")
        self.assertEqual(report[0]["original_path"], str(self.root / "none.jpg"))
        self.assertEqual(report[0]["action"], "retained_not_deleted")

    def test_candidate_requests_resume_merge_and_import(self) -> None:
        output = self.root / "requests.jsonl"
        request_manifest = self.root / "requests.manifest.json"
        decoding = {"num_candidates": 2, "temperature": 0.8}
        first = create_candidate_requests(
            self.train_path,
            output,
            request_manifest,
            split="train",
            base_model="base",
            adapter="adapter",
            decoding=decoding,
        )
        second = create_candidate_requests(
            self.train_path,
            output,
            request_manifest,
            split="train",
            base_model="base",
            adapter="adapter",
            decoding=decoding,
        )
        self.assertEqual(first["request_count"], second["request_count"])
        self.assertEqual(second["new_request_count"], 0)
        self.assertNotIn("打开闹钟设置十点提醒", json.dumps(read_jsonl(output), ensure_ascii=False))

        shards = []
        for index, task in enumerate(self.train_tasks):
            path = self.root / f"shard-{index}.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "task_id": task["task_id"],
                        "candidates": [f"模型候选{index}甲", f"模型候选{index}乙"],
                        "generation_error": None,
                        "provenance": {
                            "task_file_sha256": sha256_file(self.train_path),
                            "base_model": "base",
                            "adapter": "adapter",
                            "decoding": decoding,
                            "code_commit": "synthetic-test-commit",
                            "shard_index": index,
                            "shard_count": 2,
                        },
                    }
                ],
            )
            shards.append(path)
        merged, manifest_path = self.root / "merged.jsonl", self.root / "merged.manifest.json"
        merge = merge_candidate_shards(
            self.train_path,
            shards,
            merged,
            manifest_path,
            base_model="base",
            adapter="adapter",
            decoding=decoding,
            candidate_count=2,
        )
        imported = self.root / "imported.jsonl"
        report = import_candidate_results(
            self.train_path,
            merged,
            manifest_path,
            imported,
            expected_manifest_sha256=merge["manifest_sha256"],
            expected_base_model="base",
            expected_adapter="adapter",
        )
        self.assertEqual(report["task_count"], 2)

    def test_causal_retrieval_builds_all_three_candidate_types(self) -> None:
        target = _task("target", "train", "u1", "20260210_100000", self.image, "打开闹钟设置十点提醒")
        similar = _task("same-intent", "train", "u1", "20260201_090000", self.image, "打开闹钟设置九点提醒")
        context = _task("same-context", "train", "u1", "20260202_100500", self.image, "查看校园课程表")
        context["target"]["intent_class"] = "日程管理"
        context["target"]["app"] = "calendar.app"
        cross = _task("cross-intent", "train", "u2", "20260203_080000", self.image, "打开闹钟设置八点提醒")
        cross_oracle = _task("cross-oracle", "train", "u3", "20260203_070000", self.image, "打开闹钟设置十点提醒")
        cross_history = _task("cross-history", "train", "u4", "20260203_060000", self.image, "打开音乐")
        future = _task("future", "train", "u1", "20260220_100000", self.image, "打开闹钟设置十一点提醒")
        copied = _task("history-copy", "train", "u1", "20260201_070000", self.image, "打开音乐")
        rows = build_retrieval_candidate_pools(
            [target],
            [target, similar, context, cross, cross_oracle, cross_history, future, copied],
            split="train",
            max_per_type=2,
        )
        candidates = rows[0]["candidates"]
        self.assertEqual(candidates["same_user_similar_intent"][0]["source_task_id"], "same-intent")
        self.assertEqual(
            candidates["same_user_similar_context_different_intent"][0]["source_task_id"], "same-context"
        )
        self.assertEqual(candidates["cross_user_similar_intent"][0]["source_task_id"], "cross-intent")
        for values in candidates.values():
            self.assertTrue(all(item["source_time"] < "20260210_100000" for item in values))
            self.assertTrue(all(item["source_task_id"] != "history-copy" for item in values))
            self.assertTrue(all(item["source_task_id"] not in {"cross-oracle", "cross-history"} for item in values))
        with self.assertRaisesRegex(V4ValidationError, "strict train partition"):
            build_retrieval_candidate_pools([target], [self.eval_tasks[0]], split="train")

        groups = build_groups(
            [target],
            split="train",
            model_candidates={"target": ["模型生成的闹钟候选"]},
            retrieval_candidates=retrieval_pool_map(rows),
            synthetic_smoke=False,
        )
        sources = {candidate["source"] for candidate in groups[0]["candidates"]}
        self.assertIn("same_user_similar_intent", sources)
        self.assertIn("same_user_similar_context_different_intent", sources)
        self.assertNotIn("cross_user_similar_intent", sources)
        context_index = next(
            index
            for index, candidate in enumerate(groups[0]["candidates"])
            if candidate["source"] == "same_user_similar_context_different_intent"
        )
        self.assertEqual(groups[0]["target_distribution"][context_index], 0.0)
        self.assertEqual(groups[0]["candidates"][context_index]["metadata"]["eligibility"], "contrast_only_zero_mass")
        self.assertAlmostEqual(groups[0]["target_distribution"][groups[0]["oracle_index"]], 0.90)
        self.assertAlmostEqual(
            sum(
                probability
                for index, probability in enumerate(groups[0]["target_distribution"])
                if index != groups[0]["oracle_index"]
            ),
            0.10,
        )
        rejected = groups[0]["metadata"]["dpo_rejected_candidates"]
        analysis = groups[0]["metadata"]["cross_user_analysis_candidates"]
        self.assertEqual(len(rejected) + len(analysis), 1)

    def test_content_safety_normalization_weak_positive_and_false_negative_guard(self) -> None:
        self.assertEqual(normalize_text("打开微信，“蓝牙开锁”！"), normalize_text("打开微信蓝牙开锁"))

        target = _task("target-safe", "train", "u1", "20260210_100000", self.image, "打开华为阅读，查看上新推荐")
        target["target"]["intent_class"] = "阅读"
        weak = _task("weak", "train", "u1", "20260201_090000", self.image, "打开扇贝阅读英文短文")
        weak["target"]["intent_class"] = "阅读"
        duplicate = _task("duplicate", "train", "u1", "20260202_090000", self.image, "打开华为阅读查看上新推荐！")
        duplicate["target"]["intent_class"] = "其他"
        near_false_negative = _task(
            "near-false-negative",
            "train",
            "u1",
            "20260203_100000",
            self.image,
            "打开华为阅读，查看上新推荐。",
        )
        near_false_negative["target"]["intent_class"] = "应用管理"
        rows = build_retrieval_candidate_pools(
            [target],
            [target, weak, duplicate, near_false_negative],
            split="train",
            max_per_type=4,
        )
        pool = rows[0]
        similar = pool["candidates"]["same_user_similar_intent"]
        context = pool["candidates"]["same_user_similar_context_different_intent"]
        self.assertEqual([item["source_task_id"] for item in similar], ["weak"])
        self.assertEqual(similar[0]["eligibility"], "review_required_zero_mass")
        self.assertFalse(context)
        self.assertGreaterEqual(pool["exclusion_counts"]["oracle_text_duplicate"], 2)

        groups = build_groups(
            [target],
            split="train",
            model_candidates=None,
            retrieval_candidates=retrieval_pool_map(rows),
            synthetic_smoke=True,
        )
        by_source = {item["source"]: index for index, item in enumerate(groups[0]["candidates"])}
        weak_index = by_source["same_user_similar_intent"]
        self.assertEqual(groups[0]["target_distribution"][weak_index], 0.0)
        self.assertNotIn("synthetic_smoke", by_source)
        self.assertEqual(groups[0]["target_distribution"][groups[0]["oracle_index"]], 1.0)

    def test_smoke_keeps_unreviewed_cross_user_candidate_outside_group(self) -> None:
        target = _task("cross-target", "train", "u1", "20260210_100000", self.image, "打开百度搜索母亲节日期")
        target["target"]["intent_class"] = "信息查询"
        cross = _task("cross-source", "train", "u2", "20260201_090000", self.image, "打开夸克搜索甲状腺素作用")
        cross["target"]["intent_class"] = "信息查询"
        rows = build_retrieval_candidate_pools([target], [target, cross], split="train")
        self.assertEqual(len(rows[0]["candidates"]["cross_user_similar_intent"]), 1)
        with self.assertRaisesRegex(V4ValidationError, "no safe retrieved non-oracle candidate"):
            build_groups(
                [target],
                split="train",
                model_candidates=None,
                retrieval_candidates=retrieval_pool_map(rows),
                synthetic_smoke=True,
            )

    def test_quality_gate_blocks_workflow_language_as_candidate(self) -> None:
        task = self.train_tasks[0]
        group = build_groups(
            [task],
            split="train",
            model_candidates={task["task_id"]: ["先询问用户是否需要打开闹钟"]},
            synthetic_smoke=True,
        )[0]
        quality, issues = audit_v4_groups([group], [])
        self.assertEqual(quality["status"], "failed")
        self.assertIn("non_intent_workflow_phrase", {issue.category for issue in issues})

    def test_history_recurrence_metadata_and_normalized_duplicate_gate(self) -> None:
        repeated = _task("repeated", "train", "u1", "20260210_100000", self.image, "打开微信，蓝牙开锁")
        repeated["input"]["previous_intents"][0]["intent"] = "打开微信蓝牙开锁！"
        group = build_groups(
            [repeated],
            split="train",
            model_candidates=_model_candidates([repeated]),
            synthetic_smoke=True,
        )[0]
        recurrence = group["metadata"]["target_history_recurrence"]
        self.assertTrue(recurrence["normalized_exact"])
        self.assertEqual(recurrence["exact_match_count"], 1)

        duplicate = dict(group["candidates"][1])
        duplicate["text"] = "打开微信蓝牙开锁。"
        duplicate["candidate_id"] = "a" * 24
        group["candidates"].append(duplicate)
        group["target_distribution"].append(0.0)
        quality, issues = audit_v4_groups([group], [])
        self.assertEqual(quality["status"], "failed")
        self.assertIn("duplicate_or_empty_candidate", {issue.category for issue in issues})

    def test_manual_review_schema_gate_release_and_hashes(self) -> None:
        source_manifest = audit_source_tasks(self.train_path, self.eval_path, self.workspace)
        train = build_groups(
            self.train_tasks,
            split="train",
            model_candidates=_model_candidates(self.train_tasks),
            synthetic_smoke=True,
        )
        evaluation = build_groups(
            self.eval_tasks,
            split="eval",
            model_candidates=_model_candidates(self.eval_tasks),
            synthetic_smoke=True,
        )
        quality, issues = audit_v4_groups(train, evaluation, source_manifest=source_manifest)
        self.assertEqual(quality["status"], "passed")
        self.assertFalse(issues)

        candidate_csv = self.root / "manual_candidate_review.csv"
        group_csv = self.root / "manual_group_review.csv"
        exported = export_manual_review(train, candidate_csv, group_csv, sample_size=2)
        self.assertEqual(exported["sampled_groups"], 2)
        with candidate_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["decision"] = "keep"
        rows[0]["reviewer"] = "unit-test"
        with candidate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with group_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            group_rows = list(csv.DictReader(handle))
            group_fields = list(group_rows[0])
        group_rows[0]["decision"] = "keep"
        with group_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=group_fields)
            writer.writeheader()
            writer.writerows(group_rows)
        reviewed, review_report = apply_manual_review(
            train, candidate_csv, self.root / "review-audit.jsonl", group_csv
        )
        self.assertEqual(review_report["candidate_annotations_applied"], 1)
        self.assertEqual(review_report["group_annotations_applied"], 1)
        self.assertTrue(reviewed[0]["candidates"][0]["metadata"]["reviewed"])

        release = build_release(
            self.workspace,
            reviewed,
            evaluation,
            release_kind="smoke_v4",
            source_manifest=source_manifest,
            quality_report=quality,
            timestamp="20260621T000000Z",
        )
        verified = verify_release(release["release_dir"])
        self.assertEqual(verified["status"], "passed")
        self.assertFalse(release["manifest"]["formal_full_v4_complete"])
        registered = self.root / "registered"
        write_json(registered / "dataset_info.json", {"legacy_v3": {"file_name": "legacy.json"}})
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts" / "28_register_proactive_listwise_v4.py"),
                "--release-dir",
                release["release_dir"],
                "--dataset-dir",
                str(registered),
                "--allow-synthetic-smoke",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        registered_info = json.loads((registered / "dataset_info.json").read_text(encoding="utf-8"))
        self.assertIn("legacy_v3", registered_info)
        self.assertIn("papo_proactive_train_listwise_v4", registered_info)
        with self.assertRaisesRegex(V4ValidationError, "Synthetic"):
            build_release(
                self.workspace,
                reviewed,
                evaluation,
                release_kind="full_v4",
                source_manifest=source_manifest,
                quality_report=quality,
                candidate_provenance={"test": True},
                timestamp="20260621T000001Z",
            )

    def test_candidate_backed_draft_and_split_review_exports(self) -> None:
        source_manifest = audit_source_tasks(self.train_path, self.eval_path, self.workspace)
        imported_paths = {}
        import_report = {}
        for split, task_path, tasks in (
            ("train", self.train_path, self.train_tasks),
            ("eval", self.eval_path, self.eval_tasks),
        ):
            candidate_path = self.root / f"ui_tars_sft_{split}_candidates.jsonl"
            write_jsonl(
                candidate_path,
                [
                    {
                        "task_id": task["task_id"],
                        "candidates": [f"打开设置查看网络状态-{task['task_id']}"],
                        "candidate_manifest_sha256": "b" * 64,
                        "provenance": {
                            "base_model": "ui-tars-test",
                            "adapter": "sft-test",
                            "decoding": {"temperature": 0.8},
                            "task_file_sha256": sha256_file(task_path),
                            "shard_count": 1,
                        },
                    }
                    for task in tasks
                ],
            )
            imported_paths[split] = candidate_path
            import_report[split] = {
                "status": "passed",
                "task_count": len(tasks),
                "task_file_sha256": sha256_file(task_path),
                "output": str(candidate_path.resolve()),
                "output_sha256": sha256_file(candidate_path),
            }
        import_manifest = self.root / "candidate_import_manifest.json"
        write_json(import_manifest, import_report)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts" / "30_build_proactive_listwise_v4_draft.py"),
                "--train-tasks",
                str(self.train_path),
                "--eval-tasks",
                str(self.eval_path),
                "--workspace",
                str(self.workspace),
                "--train-candidates",
                str(imported_paths["train"]),
                "--eval-candidates",
                str(imported_paths["eval"]),
                "--candidate-import-manifest",
                str(import_manifest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        draft_dir = self.workspace / "intermediate" / "draft_v4"
        train_groups = draft_dir / "papo_proactive_train_listwise_v4.draft.groups.json"
        eval_groups = draft_dir / "papo_proactive_eval_listwise_v4.draft.groups.json"
        draft_manifest = json.loads(
            (self.workspace / "manifests" / "listwise_v4_draft_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(draft_manifest["status"], "passed_for_manual_review")
        self.assertFalse(draft_manifest["formal_release"])

        for split, groups_path in (("train", train_groups), ("eval", eval_groups)):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "24_export_manual_review_v4.py"),
                    "--groups",
                    str(groups_path),
                    "--workspace",
                    str(self.workspace),
                    "--split",
                    split,
                    "--sample-size",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            review_dir = self.workspace / "manual_review" / split
            candidate_review_path = review_dir / f"manual_candidate_review_{split}.csv"
            self.assertTrue(candidate_review_path.is_file())
            self.assertTrue((review_dir / f"manual_group_review_{split}.csv").is_file())
            self.assertTrue((review_dir / f"manual_review_export_{split}.manifest.json").is_file())
            with candidate_review_path.open("r", encoding="utf-8-sig", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            self.assertTrue(review_rows[0]["prompt_text"])
            self.assertTrue(review_rows[0]["image_paths_json"])
            self.assertIn("candidate_eligibility", review_rows[0])
            self.assertIn("retrieval_semantic_similarity", review_rows[0])
        self.assertEqual(source_manifest["hard_error_count"], 0)

    def test_quality_gate_detects_train_eval_target_leakage(self) -> None:
        train = build_groups(
            [self.train_tasks[0]],
            split="train",
            model_candidates=_model_candidates([self.train_tasks[0]]),
            synthetic_smoke=True,
        )
        leaked_task = _task("eval-leak", "eval", "u1", "20260201_100000", self.image, "打开闹钟设置十点提醒")
        evaluation = build_groups(
            [leaked_task],
            split="eval",
            model_candidates=_model_candidates([leaked_task]),
            synthetic_smoke=True,
        )
        quality, _ = audit_v4_groups(train, evaluation)
        self.assertEqual(quality["status"], "failed")
        self.assertEqual(quality["train_eval_target_overlap_count"], 1)

    def test_dataset_info_and_json_schema_registration(self) -> None:
        info = dataset_info_v4()
        self.assertEqual(info["papo_proactive_train_listwise_v4"]["formatting"], "papo_group")
        self.assertEqual(
            info["papo_proactive_eval_listwise_v4"]["columns"]["target_distribution"], "target_distribution"
        )
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "papo_listwise_v4.schema.json").read_text())
        self.assertEqual(schema["properties"]["messages"]["items"]["properties"]["role"]["enum"], ["system", "user", "observation"])
        if jsonschema is not None:
            group = build_groups(
                [self.train_tasks[0]],
                split="train",
                model_candidates=_model_candidates([self.train_tasks[0]]),
                synthetic_smoke=True,
            )[0]
            jsonschema.validate(group, schema)


if __name__ == "__main__":
    unittest.main()
