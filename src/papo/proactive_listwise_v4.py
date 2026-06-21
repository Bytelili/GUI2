from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4


PROTOCOL_ID = "fingertip20k_strict_temporal_v2"
SYSTEM_PROMPT = (
    "You are a personalized Android GUI agent. Infer the current intent from the visible context and relevant "
    "history. Output exactly one Chinese sentence. Never reveal hidden target fields."
)
FORBIDDEN_TARGET_SPLITS = {"test_suggestion.csv", "test_execution.csv", "total.csv"}
SPACE_RE = re.compile(r"\s+")


class V4ValidationError(ValueError):
    """An actionable, user-facing v4 validation failure."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _atomic_replace(temporary, destination)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    _atomic_replace(temporary, destination)


def _atomic_replace(temporary: Path, destination: Path) -> None:
    try:
        for attempt in range(8):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise V4ValidationError(f"{source}:{line_number}: blank JSONL line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise V4ValidationError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
                if not isinstance(value, dict):
                    raise V4ValidationError(f"{source}:{line_number}: JSONL row is not an object")
                yield line_number, value
    except UnicodeDecodeError as error:
        raise V4ValidationError(f"{source}: not strict UTF-8 at byte {error.start}") from error


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(path)]


def normalize_text(value: Any) -> str:
    return SPACE_RE.sub("", str(value or "").strip()).casefold()


def _image_resolution(original: str, image_roots: list[Path]) -> tuple[Path | None, list[str]]:
    attempts: list[Path] = [Path(original)]
    original_path = Path(original.replace("\\", "/"))
    parts = list(original_path.parts)
    suffix_parts: list[str] = []
    if "fingertip20k" in parts:
        suffix_parts = parts[parts.index("fingertip20k") + 1 :]
    for root in image_roots:
        attempts.append(root.joinpath(*suffix_parts) if suffix_parts else root / original_path.name)
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve(), [str(item) for item in attempts]
    return None, [str(item) for item in attempts]


def _audit_split(path: Path, split: str, image_roots: list[Path], missing_handle: Any) -> dict[str, Any]:
    task_ids: Counter[str] = Counter()
    user_ids: set[str] = set()
    episode_ids: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    image_count = 0
    available_image_count = 0
    target_signatures: set[str] = set()

    for line_number, task in iter_jsonl(path):
        item_id = str(task.get("task_id") or f"line:{line_number}")
        task_ids[item_id] += 1
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        user_id = str(inputs.get("user_id") or "")
        target_time = str(inputs.get("time") or "")
        episode_id = str(metadata.get("papo_episode_id") or "")
        if user_id:
            user_ids.add(user_id)
        if episode_id:
            episode_ids[episode_id] += 1

        if not task.get("task_id"):
            issue_counts["missing_task_id"] += 1
        if task.get("task_type") != "proactive_suggestion":
            issue_counts["invalid_task_type"] += 1
        if metadata.get("partition") != split:
            issue_counts["partition_mismatch"] += 1
        if metadata.get("protocol_id") != PROTOCOL_ID:
            issue_counts["protocol_mismatch"] += 1
        if str(metadata.get("target_split") or "").lower() in FORBIDDEN_TARGET_SPLITS:
            issue_counts["forbidden_target_split"] += 1
        if metadata.get("target_is_hidden_from_input") is not True:
            issue_counts["target_not_marked_hidden"] += 1
        if not user_id or not target_time:
            issue_counts["missing_user_or_time"] += 1
        target_intent = str(target.get("intent") or "").strip()
        if not target_intent or not str(target.get("intent_class") or "").strip():
            issue_counts["invalid_target"] += 1
        target_signatures.add(sha256_json([user_id, target_time, normalize_text(target_intent)]))

        previous = inputs.get("previous_intents") if isinstance(inputs.get("previous_intents"), list) else []
        history_ids: list[str] = []
        for history_index, history in enumerate(previous):
            if not isinstance(history, dict):
                issue_counts["invalid_history_item"] += 1
                continue
            history_user = str(history.get("user_id") or user_id)
            history_time = str(history.get("time") or "")
            history_id = str(history.get("episode_id") or "")
            history_ids.append(history_id)
            if history_user != user_id:
                issue_counts["cross_user_history"] += 1
            if not history_time or not target_time or history_time >= target_time:
                issue_counts["non_causal_history"] += 1
            if not history_id:
                issue_counts["missing_history_episode_id"] += 1
            if normalize_text(history.get("intent")) == normalize_text(target_intent):
                issue_counts["target_repeated_in_history"] += 1
        if list(metadata.get("history_episode_ids") or []) != history_ids:
            issue_counts["history_id_mismatch"] += 1

        screenshots = inputs.get("initial_screenshots") if isinstance(inputs.get("initial_screenshots"), list) else []
        if not screenshots:
            issue_counts["missing_image_reference"] += 1
        for original in screenshots:
            image_count += 1
            resolved, attempts = _image_resolution(str(original), image_roots)
            if resolved is not None:
                available_image_count += 1
            else:
                issue_counts["unavailable_image_path"] += 1
                missing_handle.write(
                    json.dumps(
                        {
                            "split": split,
                            "task_id": item_id,
                            "original_path": str(original),
                            "resolution_attempts": attempts,
                            "action": "retained_not_deleted",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    duplicate_ids = sorted(task_id for task_id, count in task_ids.items() if count > 1)
    duplicate_episodes = sorted(item for item, count in episode_ids.items() if count > 1)
    issue_counts["duplicate_task_id"] = len(duplicate_ids)
    issue_counts["duplicate_episode_id"] = len(duplicate_episodes)
    hard_categories = {
        "missing_task_id",
        "invalid_task_type",
        "partition_mismatch",
        "protocol_mismatch",
        "forbidden_target_split",
        "target_not_marked_hidden",
        "missing_user_or_time",
        "invalid_target",
        "invalid_history_item",
        "cross_user_history",
        "non_causal_history",
        "missing_history_episode_id",
        "history_id_mismatch",
        "duplicate_task_id",
        "duplicate_episode_id",
    }
    hard_count = sum(issue_counts[name] for name in hard_categories)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "line_count": sum(task_ids.values()),
        "unique_user_count": len(user_ids),
        "unique_task_id_count": len(task_ids),
        "unique_episode_id_count": len(episode_ids),
        "duplicate_task_ids": duplicate_ids,
        "duplicate_episode_ids": duplicate_episodes,
        "image_reference_count": image_count,
        "available_image_count": available_image_count,
        "unavailable_image_count": image_count - available_image_count,
        "issue_counts": dict(issue_counts),
        "hard_error_count": hard_count,
        "_task_ids": set(task_ids),
        "_target_signatures": target_signatures,
    }


def audit_source_tasks(
    train_tasks: str | Path,
    eval_tasks: str | Path,
    workspace: str | Path,
    image_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    train_path, eval_path, root = Path(train_tasks), Path(eval_tasks), Path(workspace)
    for path in (train_path, eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required strict task file does not exist: {path}")
    for relative in (
        "source",
        "candidates/train",
        "candidates/eval",
        "manual_review",
        "intermediate",
        "releases/smoke_v4",
        "releases/full_v4",
        "reports",
        "manifests",
        "logs",
    ):
        root.joinpath(*relative.split("/")).mkdir(parents=True, exist_ok=True)

    unavailable_path = root / "reports" / "source_unavailable_images.jsonl"
    with unavailable_path.open("w", encoding="utf-8", newline="\n") as missing_handle:
        roots = [Path(value) for value in image_roots]
        train = _audit_split(train_path, "train", roots, missing_handle)
        evaluation = _audit_split(eval_path, "eval", roots, missing_handle)

    overlap = sorted(train.pop("_task_ids") & evaluation.pop("_task_ids"))
    target_overlap_count = len(train.pop("_target_signatures") & evaluation.pop("_target_signatures"))
    hard_error_count = train["hard_error_count"] + evaluation["hard_error_count"] + len(overlap)
    manifest = {
        "schema_version": "papo_source_task_manifest_v4",
        "created_at": utc_timestamp(),
        "protocol_id": PROTOCOL_ID,
        "encoding": "utf-8-strict",
        "source_files_read_only": True,
        "status": "failed" if hard_error_count else "passed_with_unavailable_images" if (
            train["unavailable_image_count"] + evaluation["unavailable_image_count"]
        ) else "passed",
        "train": train,
        "eval": evaluation,
        "train_eval_task_id_overlap_count": len(overlap),
        "train_eval_task_id_overlap": overlap,
        "train_eval_target_signature_overlap_count": target_overlap_count,
        "hard_error_count": hard_error_count,
        "unavailable_images": {
            "path": str(unavailable_path.resolve()),
            "sha256": sha256_file(unavailable_path),
            "policy": "report_and_retain_original_path; never silently drop",
        },
    }
    output = root / "manifests" / "source_task_manifest.json"
    write_json(output, manifest)
    return manifest


def proactive_prompt(inputs: dict[str, Any]) -> str:
    history = [
        f"- {item.get('time', '')} | {item.get('scenario', '')} | {item.get('intent', '')}"
        for item in inputs.get("previous_intents", [])
        if isinstance(item, dict)
    ]
    return "\n".join(
        [
            "Infer the user's current intent. Output exactly one Chinese sentence.",
            f"Time: {inputs.get('time', '')}",
            f"Scenario: {inputs.get('scenario', '')}",
            f"User profile: {json.dumps(inputs.get('user_profile', {}), ensure_ascii=False)}",
            "Previous intents:",
            *(history or ["- none"]),
        ]
    )


def create_candidate_requests(
    tasks_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str,
    base_model: str,
    adapter: str,
    decoding: dict[str, Any],
    shard_index: int = 0,
    shard_count: int = 1,
    global_seed: int = 20260621,
    resume: bool = True,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    if split not in {"train", "eval"} or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise V4ValidationError("Invalid split or shard configuration for candidate requests.")
    if int(decoding.get("num_candidates", 0)) < 1:
        raise V4ValidationError("decoding.num_candidates must be positive.")
    source = Path(tasks_path)
    output = Path(output_path)
    existing: list[dict[str, Any]] = []
    completed: set[str] = set()
    if resume and output.exists():
        existing = read_jsonl(output)
        completed = {str(row.get("task_id") or "") for row in existing}

    task_sha = sha256_file(source)
    additions: list[dict[str, Any]] = []
    eligible = 0
    for _, task in iter_jsonl(source):
        task_id = str(task.get("task_id") or "")
        stable = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16], 16)
        if stable % shard_count != shard_index:
            continue
        eligible += 1
        if task_id in completed:
            continue
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        seed = int(hashlib.sha256(f"{global_seed}:{task_id}".encode("utf-8")).hexdigest()[:8], 16)
        images = [str(value) for value in inputs.get("initial_screenshots", [])]
        additions.append(
            {
                "request_id": sha256_json([task_sha, task_id, decoding, adapter])[:24],
                "task_id": task_id,
                "split": split,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>" * len(images) + proactive_prompt(inputs)},
                ],
                "images": images,
                "seed": seed,
                "decoding": decoding,
                "provenance": {
                    "task_file_sha256": task_sha,
                    "base_model": base_model,
                    "adapter": adapter,
                    "code_commit": code_commit,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                },
            }
        )
    rows = existing + additions
    write_jsonl(output, rows)
    manifest = {
        "schema_version": "papo_candidate_requests_v4",
        "created_at": utc_timestamp(),
        "split": split,
        "task_file": str(source.resolve()),
        "task_file_sha256": task_sha,
        "base_model": base_model,
        "adapter": adapter,
        "code_commit": code_commit,
        "decoding": decoding,
        "global_seed": global_seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "eligible_task_count": eligible,
        "request_count": len(rows),
        "new_request_count": len(additions),
        "request_file": str(output.resolve()),
        "request_file_sha256": sha256_file(output),
        "resume": resume,
        "oracle_target_in_prompt": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def import_candidate_results(
    tasks_path: str | Path,
    candidate_path: str | Path,
    candidate_manifest_path: str | Path,
    output_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_base_model: str | None = None,
    expected_adapter: str | None = None,
) -> dict[str, Any]:
    source, candidates_file, manifest_file = Path(tasks_path), Path(candidate_path), Path(candidate_manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8", errors="strict"))
    task_sha = sha256_file(source)
    manifest_sha = sha256_file(manifest_file)
    errors: list[str] = []
    if manifest_sha != expected_manifest_sha256:
        errors.append("candidate manifest SHA256 mismatch")
    if manifest.get("task_file_sha256") != task_sha:
        errors.append("task file SHA256 mismatch")
    if manifest.get("candidate_file_sha256") != sha256_file(candidates_file):
        errors.append("candidate file SHA256 mismatch")
    if expected_base_model and manifest.get("base_model") != expected_base_model:
        errors.append("base model provenance mismatch")
    if expected_adapter and manifest.get("adapter") != expected_adapter:
        errors.append("adapter provenance mismatch")
    if not str(manifest.get("base_model") or "") or not str(manifest.get("adapter") or ""):
        errors.append("missing base model or adapter provenance")
    if not isinstance(manifest.get("decoding"), dict):
        errors.append("missing decoding provenance")
    if not isinstance(manifest.get("code_commits"), list) or not manifest.get("code_commits"):
        errors.append("missing code commit provenance")
    if int(manifest.get("candidate_count", 0)) < 1:
        errors.append("invalid candidate_count provenance")
    if int(manifest.get("shard_count", 0)) < 1:
        errors.append("invalid shard_count provenance")
    shard_indices = manifest.get("shard_indices")
    if not isinstance(shard_indices, list) or set(shard_indices) != set(range(int(manifest.get("shard_count", 0)))):
        errors.append("candidate shards are incomplete")

    expected_ids = {str(row.get("task_id") or "") for _, row in iter_jsonl(source)}
    imported: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_candidate_count = 0
    generation_error_count = 0
    for line_number, row in iter_jsonl(candidates_file):
        task_id = str(row.get("task_id") or "")
        if task_id in seen:
            errors.append(f"duplicate task_id at candidate line {line_number}: {task_id}")
            continue
        seen.add(task_id)
        if row.get("generation_error"):
            generation_error_count += 1
            errors.append(f"generation error for task {task_id}: {row.get('generation_error')}")
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            errors.append(f"candidates is not a list for task {task_id}")
            continue
        texts: list[str] = []
        normalized: set[str] = set()
        for item in raw_candidates:
            text = str(item.get("text") if isinstance(item, dict) else item).strip()
            key = normalize_text(text)
            if not key:
                errors.append(f"empty candidate for task {task_id}")
            elif key in normalized:
                duplicate_candidate_count += 1
            else:
                normalized.add(key)
                texts.append(text)
        if len(texts) != int(manifest.get("candidate_count", len(texts))):
            errors.append(f"candidate count mismatch for task {task_id}")
        imported.append(
            {
                "task_id": task_id,
                "candidates": texts,
                "candidate_manifest_sha256": manifest_sha,
                "provenance": {
                    "base_model": manifest.get("base_model"),
                    "adapter": manifest.get("adapter"),
                    "decoding": manifest.get("decoding"),
                    "task_file_sha256": task_sha,
                    "shard_count": manifest.get("shard_count"),
                },
            }
        )
    missing = sorted(expected_ids - seen)
    extra = sorted(seen - expected_ids)
    if missing:
        errors.append(f"task coverage missing {len(missing)} task(s), first={missing[:3]}")
    if extra:
        errors.append(f"candidate results contain {len(extra)} unknown task(s), first={extra[:3]}")
    if duplicate_candidate_count:
        errors.append(f"duplicate candidates found: {duplicate_candidate_count}")
    if errors:
        raise V4ValidationError("Candidate import failed:\n- " + "\n- ".join(errors[:50]))
    write_jsonl(output_path, imported)
    return {
        "status": "passed",
        "task_count": len(imported),
        "task_file_sha256": task_sha,
        "candidate_file_sha256": sha256_file(candidates_file),
        "candidate_manifest_sha256": manifest_sha,
        "duplicate_candidate_count": duplicate_candidate_count,
        "generation_error_count": generation_error_count,
        "output": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(output_path),
    }


def merge_candidate_shards(
    tasks_path: str | Path,
    shard_paths: Iterable[str | Path],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    base_model: str,
    adapter: str,
    decoding: dict[str, Any],
    candidate_count: int,
) -> dict[str, Any]:
    paths = [Path(value) for value in shard_paths]
    if not paths:
        raise V4ValidationError("At least one candidate shard is required.")
    task_sha = sha256_file(tasks_path)
    by_task: dict[str, dict[str, Any]] = {}
    shard_indices: set[int] = set()
    code_commits: set[str] = set()
    shard_count: int | None = None
    for path in paths:
        current_index: int | None = None
        for _, row in iter_jsonl(path):
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
            if provenance.get("task_file_sha256") != task_sha:
                raise V4ValidationError(f"Shard task SHA256 mismatch: {path}")
            if provenance.get("base_model") != base_model or provenance.get("adapter") != adapter:
                raise V4ValidationError(f"Shard model/adapter provenance mismatch: {path}")
            if provenance.get("decoding") != decoding:
                raise V4ValidationError(f"Shard decoding provenance mismatch: {path}")
            code_commit = str(provenance.get("code_commit") or "")
            if not code_commit:
                raise V4ValidationError(f"Shard code commit provenance is missing: {path}")
            code_commits.add(code_commit)
            row_index, row_count = int(provenance.get("shard_index", -1)), int(provenance.get("shard_count", 0))
            if current_index is None:
                current_index = row_index
            if row_index != current_index or row_count < 1:
                raise V4ValidationError(f"Inconsistent shard provenance inside: {path}")
            if shard_count is None:
                shard_count = row_count
            if row_count != shard_count:
                raise V4ValidationError("Shard count differs across result files.")
            task_id = str(row.get("task_id") or "")
            if task_id in by_task:
                raise V4ValidationError(f"Duplicate task across candidate shards: {task_id}")
            by_task[task_id] = row
        if current_index is None:
            raise V4ValidationError(f"Empty candidate shard: {path}")
        if current_index in shard_indices:
            raise V4ValidationError(f"Duplicate shard index: {current_index}")
        shard_indices.add(current_index)
    if shard_count is None or shard_indices != set(range(shard_count)):
        raise V4ValidationError(f"Incomplete shard set: got={sorted(shard_indices)}, expected=0..{(shard_count or 1) - 1}")
    rows = [by_task[task_id] for task_id in sorted(by_task)]
    write_jsonl(output_path, rows)
    manifest = {
        "schema_version": "papo_ui_tars_candidate_results_v4",
        "created_at": utc_timestamp(),
        "task_file_sha256": task_sha,
        "candidate_file_sha256": sha256_file(output_path),
        "base_model": base_model,
        "adapter": adapter,
        "decoding": decoding,
        "candidate_count": candidate_count,
        "shard_count": shard_count,
        "shard_indices": sorted(shard_indices),
        "code_commits": sorted(code_commits),
        "task_count": len(rows),
        "source_shards": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def _specificity(text: str) -> float:
    length = len(normalize_text(text))
    if length < 4:
        return length / 4.0
    return 1.0 if length <= 48 else max(0.0, 1.0 - (length - 48) / 96.0)


def _candidate_reward(text: str, target: str, source: str) -> dict[str, float]:
    task = _similarity(text, target)
    user = {
        "oracle_target": 1.0,
        "same_user_similar_intent": 0.9,
        "same_user_similar_context_different_intent": 0.8,
        "ui_tars_sft": 0.5,
        "synthetic_smoke": 0.5,
    }.get(source, 0.0)
    context = {
        "oracle_target": 1.0,
        "same_user_similar_intent": 0.8,
        "same_user_similar_context_different_intent": 0.9,
        "ui_tars_sft": 1.0,
        "synthetic_smoke": 1.0,
    }.get(source, 0.0)
    specificity = _specificity(text)
    total = 0.55 * task + 0.20 * user + 0.15 * context + 0.10 * specificity
    return {
        "R_task": task,
        "R_user": user,
        "R_context": context,
        "R_specificity": specificity,
        "total": total,
    }


def stratified_tasks(tasks: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(tasks):
        return tasks
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        buckets[(str(inputs.get("user_id") or ""), str(target.get("intent_class") or ""))].append(task)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def _task_record(task: dict[str, Any]) -> dict[str, Any]:
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    intent = str(target.get("intent") or "").strip()
    normalized = normalize_text(intent)
    return {
        "task_id": str(task.get("task_id") or ""),
        "episode_id": str(metadata.get("papo_episode_id") or ""),
        "partition": str(metadata.get("partition") or ""),
        "protocol_id": str(metadata.get("protocol_id") or ""),
        "user_id": str(inputs.get("user_id") or ""),
        "time": str(inputs.get("time") or ""),
        "scenario": str(inputs.get("scenario") or ""),
        "intent": intent,
        "_normalized_intent": normalized,
        "_intent_bigrams": {
            normalized[index : index + 2]
            for index in range(max(1, len(normalized) - 1))
            if normalized[index : index + 2]
        },
        "intent_class": str(target.get("intent_class") or ""),
        "app": str(target.get("app") or ""),
    }


def _retrieval_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_grams = left["_intent_bigrams"]
    right_grams = right["_intent_bigrams"]
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _hour_similarity(left: str, right: str) -> float:
    try:
        left_hour = int(left.split("_", 1)[1][:2])
        right_hour = int(right.split("_", 1)[1][:2])
    except (IndexError, TypeError, ValueError):
        return 0.0
    distance = min(abs(left_hour - right_hour), 24 - abs(left_hour - right_hour))
    return 1.0 - distance / 12.0


def _retrieval_candidate(
    target: dict[str, Any],
    source: dict[str, Any],
    relation: str,
    score: float,
    *,
    eligibility: str,
) -> dict[str, Any]:
    similarity = _retrieval_similarity(target, source)
    return {
        "candidate_id": sha256_json([target["task_id"], relation, source["episode_id"], normalize_text(source["intent"])])[:24],
        "text": source["intent"],
        "source": relation,
        "source_task_id": source["task_id"],
        "source_episode_id": source["episode_id"],
        "source_user_id": source["user_id"],
        "source_time": source["time"],
        "source_scenario": source["scenario"],
        "source_intent_class": source["intent_class"],
        "source_app": source["app"],
        "retrieval": {
            "semantic_similarity": similarity,
            "scenario_match": target["scenario"] == source["scenario"],
            "hour_similarity": _hour_similarity(target["time"], source["time"]),
            "score": score,
            "strictly_before_target": source["time"] < target["time"],
        },
        "eligibility": eligibility,
    }


def build_retrieval_candidate_pools(
    target_tasks: list[dict[str, Any]],
    reference_tasks: list[dict[str, Any]],
    *,
    split: str,
    max_per_type: int = 2,
    pseudo_negative_similarity: float = 0.55,
) -> list[dict[str, Any]]:
    r"""Build causal retrieval pools without using eval targets as references."""
    if split not in {"train", "eval"} or max_per_type < 1:
        raise V4ValidationError("Invalid retrieval pool split or max_per_type.")
    records = [_task_record(task) for task in reference_tasks]
    if any(record["partition"] != "train" or record["protocol_id"] != PROTOCOL_ID for record in records):
        raise V4ValidationError("Retrieval reference tasks must come exclusively from the strict train partition.")
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_user[record["user_id"]].append(record)
        if record["intent_class"]:
            by_class[record["intent_class"]].append(record)
        if record["app"]:
            by_app[record["app"]].append(record)

    pools: list[dict[str, Any]] = []
    for task in target_tasks:
        target = _task_record(task)
        if target["partition"] != split or target["protocol_id"] != PROTOCOL_ID:
            raise V4ValidationError(f"Retrieval target partition/protocol mismatch: {target['task_id']}")
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        history_texts = {
            normalize_text(item.get("intent"))
            for item in inputs.get("previous_intents", [])
            if isinstance(item, dict) and normalize_text(item.get("intent"))
        }
        exclusions: Counter[str] = Counter()

        same_similar: list[tuple[float, dict[str, Any]]] = []
        same_context: list[tuple[float, dict[str, Any]]] = []
        for source in by_user.get(target["user_id"], []):
            if not source["time"] or source["time"] >= target["time"] or source["task_id"] == target["task_id"]:
                exclusions["not_strictly_earlier"] += 1
                continue
            key = normalize_text(source["intent"])
            if not key:
                exclusions["empty_intent"] += 1
                continue
            if any(key in history_key or history_key in key for history_key in history_texts):
                exclusions["verbatim_prompt_history_copy"] += 1
                continue
            if key == normalize_text(target["intent"]):
                exclusions["oracle_text_duplicate"] += 1
                continue
            similarity = _retrieval_similarity(target, source)
            scenario = float(bool(target["scenario"]) and target["scenario"] == source["scenario"])
            hour = _hour_similarity(target["time"], source["time"])
            same_intent_family = bool(
                (target["intent_class"] and target["intent_class"] == source["intent_class"])
                or (target["app"] and target["app"] == source["app"])
            )
            if same_intent_family:
                same_similar.append((0.65 * similarity + 0.20 * scenario + 0.15 * hour, source))
            elif scenario > 0.0 or hour >= 0.75:
                same_context.append((0.15 * similarity + 0.55 * scenario + 0.30 * hour, source))

        cross_sources: dict[str, dict[str, Any]] = {}
        for source in by_class.get(target["intent_class"], []):
            cross_sources[source["task_id"]] = source
        for source in by_app.get(target["app"], []):
            cross_sources[source["task_id"]] = source
        cross_similar: list[tuple[float, dict[str, Any], str]] = []
        for source in cross_sources.values():
            if source["user_id"] == target["user_id"]:
                continue
            if not source["time"] or source["time"] >= target["time"]:
                exclusions["cross_user_not_strictly_earlier"] += 1
                continue
            key = normalize_text(source["intent"])
            if not key:
                continue
            similarity = _retrieval_similarity(target, source)
            scenario = float(bool(target["scenario"]) and target["scenario"] == source["scenario"])
            hour = _hour_similarity(target["time"], source["time"])
            score = 0.75 * similarity + 0.15 * scenario + 0.10 * hour
            eligibility = (
                "analysis_only_pseudo_negative_risk"
                if similarity >= pseudo_negative_similarity
                else "dpo_rejected_review_required"
            )
            cross_similar.append((score, source, eligibility))

        same_similar.sort(key=lambda item: (item[0], item[1]["time"], item[1]["task_id"]), reverse=True)
        same_context.sort(key=lambda item: (item[0], item[1]["time"], item[1]["task_id"]), reverse=True)
        cross_similar.sort(key=lambda item: (item[0], item[1]["time"], item[1]["task_id"]), reverse=True)
        candidates = {
            "same_user_similar_intent": [
                _retrieval_candidate(target, source, "same_user_similar_intent", score, eligibility="listwise")
                for score, source in same_similar[:max_per_type]
            ],
            "same_user_similar_context_different_intent": [
                _retrieval_candidate(
                    target,
                    source,
                    "same_user_similar_context_different_intent",
                    score,
                    eligibility="listwise",
                )
                for score, source in same_context[:max_per_type]
            ],
            "cross_user_similar_intent": [
                _retrieval_candidate(target, source, "cross_user_similar_intent", score, eligibility=eligibility)
                for score, source, eligibility in cross_similar[:max_per_type]
            ],
        }
        pools.append(
            {
                "task_id": target["task_id"],
                "split": split,
                "target_time": target["time"],
                "target_user_id": target["user_id"],
                "target_intent_class": target["intent_class"],
                "reference_partition": "train",
                "candidates": candidates,
                "exclusion_counts": dict(exclusions),
            }
        )
    return pools


def retrieval_pool_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        str(row.get("task_id") or ""): dict(row.get("candidates") or {})
        for row in rows
        if str(row.get("task_id") or "")
    }


def _synthetic_negative(target: str) -> str:
    return f"先询问用户是否需要{target.rstrip('。')}"


def build_groups(
    tasks: list[dict[str, Any]],
    *,
    split: str,
    model_candidates: dict[str, list[str]] | None,
    retrieval_candidates: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    synthetic_smoke: bool,
    oracle_probability: float = 0.80,
) -> list[dict[str, Any]]:
    if not 0.5 < oracle_probability < 1.0:
        raise V4ValidationError("oracle_probability must be between 0.5 and 1.0")
    if not synthetic_smoke and model_candidates is None:
        raise V4ValidationError("Formal v4 requires imported UI-TARS SFT candidates; only smoke may be synthetic.")
    groups: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        target_text = str(target.get("intent") or "").strip()
        history_text = {normalize_text(item.get("intent")) for item in inputs.get("previous_intents", []) if isinstance(item, dict)}
        candidate_specs: list[dict[str, Any]] = [{"text": target_text, "source": "oracle_target"}]
        retrieval = (retrieval_candidates or {}).get(task_id, {})
        for source_name in ("same_user_similar_intent", "same_user_similar_context_different_intent"):
            values = retrieval.get(source_name) or []
            if values:
                candidate_specs.append(dict(values[0]))
        if model_candidates is not None:
            for text in model_candidates.get(task_id, []):
                key = normalize_text(text)
                if key and key != normalize_text(target_text) and key not in history_text:
                    candidate_specs.append({"text": str(text).strip(), "source": "ui_tars_sft"})
                if len(candidate_specs) >= 4:
                    break
        if synthetic_smoke and len(candidate_specs) < 2:
            candidate_specs.append({"text": _synthetic_negative(target_text), "source": "synthetic_smoke"})
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for spec in candidate_specs:
            text, source = str(spec.get("text") or ""), str(spec.get("source") or "")
            key = normalize_text(text)
            if not key or key in seen:
                continue
            seen.add(key)
            reward = _candidate_reward(text, target_text, source)
            candidates.append(
                {
                    "candidate_id": sha256_json([task_id, source, key])[:24],
                    "text": text,
                    "source": source,
                    "reward": reward,
                    "metadata": {
                        "rank": 0,
                        "target_probability": 0.0,
                        "source_task_id": spec.get("source_task_id", ""),
                        "source_episode_id": spec.get("source_episode_id", ""),
                        "source_user_id": spec.get("source_user_id", ""),
                        "source_time": spec.get("source_time", ""),
                        "retrieval": spec.get("retrieval", {}),
                    },
                }
            )
        if len(candidates) < 2:
            raise V4ValidationError(f"Task {task_id} has no usable non-oracle candidate.")
        candidates.sort(key=lambda item: (item["source"] != "oracle_target", -item["reward"]["total"]))
        oracle_index = next(index for index, item in enumerate(candidates) if item["source"] == "oracle_target")
        non_oracle_total = sum(max(float(item["reward"]["total"]), 1e-8) for item in candidates if item["source"] != "oracle_target")
        distribution: list[float] = []
        for index, item in enumerate(candidates):
            probability = oracle_probability if index == oracle_index else (
                (1.0 - oracle_probability) * max(float(item["reward"]["total"]), 1e-8) / non_oracle_total
            )
            item["metadata"]["target_probability"] = probability
            distribution.append(probability)
        ranked = sorted(range(len(candidates)), key=lambda index: distribution[index], reverse=True)
        for rank, index in enumerate(ranked, start=1):
            candidates[index]["metadata"]["rank"] = rank
        images = [str(value) for value in inputs.get("initial_screenshots", [])]
        groups.append(
            {
                "task_id": task_id,
                "group_id": f"papo_v4::{split}::{task_id}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "<image>" * len(images) + proactive_prompt(inputs)},
                ],
                "images": images,
                "candidates": candidates,
                "target_distribution": distribution,
                "oracle_index": oracle_index,
                "metadata": {
                    "partition": split,
                    "protocol_id": metadata.get("protocol_id"),
                    "papo_episode_id": metadata.get("papo_episode_id"),
                    "history_episode_ids": list(metadata.get("history_episode_ids") or []),
                    "target_split": metadata.get("target_split"),
                    "history_policy": metadata.get("history_policy"),
                    "target_time": inputs.get("time"),
                    "user_id": inputs.get("user_id"),
                    "intent_class": target.get("intent_class"),
                    "oracle_text_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
                    "target_identity_sha256": sha256_json(
                        [inputs.get("user_id"), inputs.get("time"), normalize_text(target_text)]
                    ),
                    "reward_definition": {
                        "R_user": "source-evidence prior; history-copy candidates are excluded, not rewarded by text overlap",
                        "weights": {"R_task": 0.55, "R_user": 0.20, "R_context": 0.15, "R_specificity": 0.10},
                    },
                    "release_eligibility": "synthetic_smoke_only" if synthetic_smoke else "formal_candidate_import",
                    "dpo_rejected_candidates": [
                        item
                        for item in retrieval.get("cross_user_similar_intent", [])
                        if item.get("eligibility") == "dpo_rejected"
                    ],
                    "cross_user_analysis_candidates": [
                        item
                        for item in retrieval.get("cross_user_similar_intent", [])
                        if item.get("eligibility") != "dpo_rejected"
                    ],
                },
            }
        )
    return groups


def dataset_info_v4() -> dict[str, Any]:
    common = {
        "formatting": "papo_group",
        "columns": {
            "messages": "messages",
            "images": "images",
            "candidates": "candidates",
            "target_distribution": "target_distribution",
            "oracle_index": "oracle_index",
            "group_id": "group_id",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    return {
        "papo_proactive_train_listwise_v4": {
            "file_name": "papo_proactive_train_listwise_v4.json",
            **common,
        },
        "papo_proactive_eval_listwise_v4": {
            "file_name": "papo_proactive_eval_listwise_v4.json",
            **common,
        },
    }


def load_candidate_map(path: str | Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    mapping: dict[str, list[str]] = {}
    provenance: dict[str, Any] = {}
    for _, row in iter_jsonl(path):
        task_id = str(row.get("task_id") or "")
        mapping[task_id] = [str(value) for value in row.get("candidates", [])]
        provenance[task_id] = row.get("provenance")
    return mapping, provenance


def build_release(
    workspace: str | Path,
    train_groups: list[dict[str, Any]],
    eval_groups: list[dict[str, Any]],
    *,
    release_kind: str,
    source_manifest: dict[str, Any],
    quality_report: dict[str, Any],
    candidate_provenance: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if release_kind not in {"smoke_v4", "full_v4"}:
        raise V4ValidationError("release_kind must be smoke_v4 or full_v4")
    synthetic = any(
        group.get("metadata", {}).get("release_eligibility") == "synthetic_smoke_only"
        for group in train_groups + eval_groups
    )
    if release_kind == "full_v4" and synthetic:
        raise V4ValidationError("Synthetic candidates can never be exported as a formal full_v4 release.")
    if release_kind == "full_v4" and not candidate_provenance:
        raise V4ValidationError("Formal full_v4 release requires imported candidate provenance.")
    if quality_report.get("status") == "failed":
        raise V4ValidationError("Quality gate failed; release export is blocked.")

    stamp = timestamp or utc_timestamp()
    release_dir = Path(workspace) / "releases" / release_kind / stamp
    if release_dir.exists():
        raise FileExistsError(f"Refusing to overwrite release directory: {release_dir}")
    release_dir.mkdir(parents=True)
    train_path = release_dir / "papo_proactive_train_listwise_v4.json"
    eval_path = release_dir / "papo_proactive_eval_listwise_v4.json"
    dataset_path = release_dir / "dataset_info_v4.json"
    quality_path = release_dir / "listwise_v4_quality_report.json"
    write_json(train_path, train_groups)
    write_json(eval_path, eval_groups)
    write_json(dataset_path, dataset_info_v4())
    write_json(quality_path, quality_report)
    artifacts = [train_path, eval_path, dataset_path, quality_path]
    manifest = {
        "schema_version": "papo_listwise_v4_manifest",
        "created_at": stamp,
        "release_kind": release_kind,
        "release_status": "synthetic_smoke_not_for_formal_training" if synthetic else "formal_candidate_release",
        "protocol_id": PROTOCOL_ID,
        "source_task_manifest_sha256": sha256_json(source_manifest),
        "source_tasks": {
            "train_sha256": source_manifest.get("train", {}).get("sha256"),
            "eval_sha256": source_manifest.get("eval", {}).get("sha256"),
        },
        "candidate_provenance": candidate_provenance,
        "group_counts": {"train": len(train_groups), "eval": len(eval_groups)},
        "dataset_hashes": {path.name: sha256_file(path) for path in artifacts[:3]},
        "quality_report_sha256": sha256_file(quality_path),
        "quality_status": quality_report.get("status"),
        "formal_full_v4_complete": release_kind == "full_v4" and not synthetic,
    }
    manifest_path = release_dir / "listwise_v4_manifest.json"
    write_json(manifest_path, manifest)
    artifacts.append(manifest_path)
    sums_path = release_dir / "SHA256SUMS.txt"
    sums_path.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in artifacts), encoding="utf-8")

    archive = release_dir.parent / f"PAPO_Listwise_v4_release_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in [*artifacts, sums_path]:
            handle.add(path, arcname=path.name)
    archive_sha_path = archive.with_suffix(archive.suffix + ".sha256")
    archive_sha_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return {
        "release_dir": str(release_dir.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256_file": str(archive_sha_path.resolve()),
        "manifest": manifest,
    }


def verify_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir)
    sums = root / "SHA256SUMS.txt"
    errors: list[str] = []
    checked = 0
    for line in sums.read_text(encoding="utf-8", errors="strict").splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.strip()
        checked += 1
        if not path.is_file():
            errors.append(f"missing artifact: {path.name}")
        elif sha256_file(path) != expected:
            errors.append(f"SHA256 mismatch: {path.name}")
    manifest = json.loads((root / "listwise_v4_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("dataset_hashes", {}).items():
        if sha256_file(root / name) != expected:
            errors.append(f"manifest dataset hash mismatch: {name}")
    return {"status": "failed" if errors else "passed", "checked": checked, "errors": errors}
