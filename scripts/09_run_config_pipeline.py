from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
from papo.data_protocol import PROTOCOL_FILES  # noqa: E402
from papo.dpo import build_pairs  # noqa: E402
from papo.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from papo.llamafactory_export import (  # noqa: E402
    attach_prior_actions,
    dataset_info,
    export_execution_dpo,
    export_execution_listwise,
    export_execution_sft,
    export_proactive_sft,
    write_json as write_json_array,
)
from papo.papo_objective import propagate_residual_values, score_tree_leaves  # noqa: E402
from papo.raw_builder import build_from_raw  # noqa: E402
from papo.tasks import build_personalized_execution_tasks, build_proactive_suggestion_tasks  # noqa: E402
from papo.tree_builder import build_offline_counterfactual_tree, build_tree_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict config-driven PAPO offline pipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--stages",
        default="tasks,raw,trees,objective,export",
        help="Comma-separated: proactive_tasks, proactive_export, tasks, raw, trees, objective, export",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit per partition.")
    args = parser.parse_args()

    config = load_config(args.config)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    paths = _paths(config)
    _require_protocol(paths)
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    paths["task_dir"].mkdir(parents=True, exist_ok=True)

    if "proactive_tasks" in stages:
        _build_proactive_tasks(config, paths, args.limit)
    if "proactive_export" in stages:
        _export_proactive(config, paths)
    if "tasks" in stages:
        _build_tasks(config, paths, args.limit)
    if "raw" in stages:
        _build_raw(config, paths, args.limit)
    if "trees" in stages:
        _build_trees(config, paths, args.limit)
    if "objective" in stages:
        _build_objective(config, paths)
    if "export" in stages:
        _export_llamafactory(config, paths)


def _build_proactive_tasks(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    data = config["data"]
    protocol_id = str(data["protocol"]["protocol_id"])
    for partition in ["train", "eval"]:
        tasks = build_proactive_suggestion_tasks(
            paths[f"proactive_{partition}_targets"],
            paths["proactive_history"],
            paths["official_root"] / "user_profile.csv",
            paths["raw_root"],
            screenshot_level=int(data["suggestion_screenshot_level"]),
            history_limit=int(data["suggestion_history_limit"]),
            limit=limit,
            require_complete=bool(data["require_complete"]),
            provenance={
                "partition": partition,
                "protocol_id": protocol_id,
                "target_split": paths[f"proactive_{partition}_targets"].name,
                "history_split": paths["proactive_history"].name,
            },
        )
        write_jsonl(paths[f"proactive_{partition}_tasks"], tasks)
        print(f"proactive {partition} tasks: {len(tasks)}")


def _export_proactive(config: dict[str, Any], paths: dict[str, Path]) -> None:
    output = paths["llamafactory_data_dir"]
    output.mkdir(parents=True, exist_ok=True)
    for partition in ["train", "eval"]:
        tasks = read_jsonl(paths[f"proactive_{partition}_tasks"])
        rows = export_proactive_sft(tasks, paths["raw_root"], str(config["paths"]["asset_prefix"]))
        path = output / f"papo_proactive_{partition}_sft.json"
        write_json_array(path, rows)
        print(f"proactive {partition} SFT: {len(rows)} -> {path}")
    _write_dataset_info(output)


def _build_tasks(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    _build_proactive_tasks(config, paths, limit)
    references = config["papo"]["references"]
    protocol_id = str(config["data"]["protocol"]["protocol_id"])
    for partition in ["train", "eval"]:
        tasks = build_personalized_execution_tasks(
            paths[f"execution_{partition}_targets"],
            paths["execution_references"],
            paths["official_root"] / "user_profile.csv",
            paths["raw_root"],
            limit=limit,
            require_complete=bool(config["data"]["require_complete"]),
            same_user_top_k=int(references["same_user_top_k"]),
            cross_user_top_k=int(references["cross_user_top_k"]),
            intent_similarity_threshold=float(references["intent_similarity_threshold"]),
            exclude_same_intent=bool(references["exclude_same_intent"]),
            provenance={
                "partition": partition,
                "protocol_id": protocol_id,
                "target_split": paths[f"execution_{partition}_targets"].name,
                "reference_split": paths["execution_references"].name,
            },
        )
        write_jsonl(paths[f"execution_{partition}_tasks"], tasks)
        print(f"execution {partition} tasks: {len(tasks)}")


def _build_raw(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    tasks = read_jsonl(paths["execution_train_tasks"]) + read_jsonl(paths["execution_eval_tasks"])
    if limit > 0:
        tasks = read_jsonl(paths["execution_train_tasks"])[:limit] + read_jsonl(paths["execution_eval_tasks"])[:limit]
    selected = _task_episode_pool(tasks)
    episodes, steps, audit = build_from_raw(
        paths["raw_root"],
        max_episodes=0,
        require_complete=bool(config["data"]["require_complete"]),
        selected_episodes=selected,
        max_episodes_per_user=0,
        progress_every=int(config["data"].get("progress_every", 25)),
    )
    write_jsonl(paths["episodes"], episodes)
    write_jsonl(paths["steps"], steps)
    write_json(paths["raw_audit"], audit)
    print(f"raw: episodes={len(episodes)}, steps={len(steps)}")


def _build_trees(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    steps = read_jsonl(paths["steps"])
    train_tasks = read_jsonl(paths["execution_train_tasks"])
    eval_tasks = read_jsonl(paths["execution_eval_tasks"])
    train_ids = {_task_episode_id(task) for task in train_tasks}
    eval_ids = {_task_episode_id(task) for task in eval_tasks}
    train_steps = [step for step in steps if str(step.get("episode_id") or "") in train_ids]
    eval_steps_by_episode: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        episode_id = str(step.get("episode_id") or "")
        if episode_id in eval_ids:
            eval_steps_by_episode.setdefault(episode_id, []).append(step)

    tree_config = config["papo"]["tree"]
    train_roots = _root_steps(train_steps, train_ids, bool(config["data"]["root_only"]), limit)
    train_context = build_tree_context(train_steps)
    train_trees = _make_trees(train_roots, train_steps, train_context, tree_config, config)
    write_jsonl(paths["trees"], train_trees)

    eval_roots = _root_steps(steps, eval_ids, bool(config["data"]["root_only"]), limit)
    eval_trees: list[dict[str, Any]] = []
    progress_every = int(config["data"].get("progress_every", 25))
    print(f"eval trees: building {len(eval_roots)} leakage-isolated trees...", flush=True)
    for index, root in enumerate(eval_roots, start=1):
        episode_id = str(root.get("episode_id") or "")
        isolated_steps = train_steps + eval_steps_by_episode.get(episode_id, [])
        eval_trees.append(
            build_offline_counterfactual_tree(
                root,
                isolated_steps,
                max_depth=int(tree_config["max_depth"]),
                same_user_k=int(tree_config["same_user_k"]),
                cross_user_k=int(tree_config["cross_user_k"]),
                max_candidates=int(tree_config["max_candidates"]),
                context=build_tree_context(isolated_steps),
            )
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(eval_roots)):
            print(f"eval tree progress: {index}/{len(eval_roots)}", flush=True)
    write_jsonl(paths["eval_trees"], eval_trees)
    print(f"trees: train={len(train_trees)}, eval={len(eval_trees)}")


def _build_objective(config: dict[str, Any], paths: dict[str, Path]) -> None:
    for partition in ["train", "eval"]:
        tasks = read_jsonl(paths[f"execution_{partition}_tasks"])
        _attach_canonical_reference_actions(tasks, read_jsonl(paths["steps"]))
        task_by_episode = {
            str(task.get("metadata", {}).get("papo_episode_id") or _task_episode_id(task)): task
            for task in tasks
        }
        trees_path = paths["trees"] if partition == "train" else paths["eval_trees"]
        scored_trees = [
            score_tree_leaves(tree, task_by_episode.get(str(tree.get("episode_id") or "")), config["papo"]["reward"])
            for tree in read_jsonl(trees_path)
        ]
        value_config = config["papo"]["value"]
        action_values, listwise = propagate_residual_values(
            scored_trees,
            alpha=float(value_config["conservative_alpha"]),
            beta=float(value_config["beta"]),
            coverage_kappa=float(config["papo"]["coverage"]["kappa"]),
        )
        pair_config = config["papo"]["pairwise"]
        pairs = build_pairs(
            action_values,
            margin=float(pair_config["margin"]),
            tau_m=float(pair_config["tau_m"]),
            w_max=float(pair_config["max_weight"]),
            beta=float(value_config["beta"]),
        )
        prefix = "" if partition == "train" else "eval_"
        write_jsonl(paths[f"{prefix}scored_trees"], scored_trees)
        write_jsonl(paths[f"{prefix}action_values"], action_values)
        write_jsonl(paths[f"{prefix}listwise"], listwise)
        write_jsonl(paths[f"{prefix}pairs"], pairs)
        print(
            f"objective {partition}: action_values={len(action_values)}, "
            f"listwise={len(listwise)}, pairs={len(pairs)}"
        )


def _export_llamafactory(config: dict[str, Any], paths: dict[str, Path]) -> None:
    steps = read_jsonl(paths["steps"])
    output = paths["llamafactory_data_dir"]
    asset_prefix = str(config["paths"]["asset_prefix"])
    output.mkdir(parents=True, exist_ok=True)

    for partition in ["train", "eval"]:
        proactive_tasks = read_jsonl(paths[f"proactive_{partition}_tasks"])
        execution_tasks = read_jsonl(paths[f"execution_{partition}_tasks"])
        attach_prior_actions(execution_tasks, steps)
        prefix = "" if partition == "train" else "eval_"
        exports = {
            f"papo_proactive_{partition}_sft.json": export_proactive_sft(
                proactive_tasks, paths["raw_root"], asset_prefix
            ),
            f"papo_execution_{partition}_sft.json": export_execution_sft(
                execution_tasks, steps, paths["raw_root"], asset_prefix
            ),
            f"papo_execution_{partition}_dpo.json": export_execution_dpo(
                execution_tasks, steps, read_jsonl(paths[f"{prefix}pairs"]), paths["raw_root"], asset_prefix
            ),
            f"papo_execution_{partition}_listwise.json": export_execution_listwise(
                execution_tasks, steps, read_jsonl(paths[f"{prefix}listwise"]), paths["raw_root"], asset_prefix
            ),
        }
        for filename, rows in exports.items():
            write_json_array(output / filename, rows)
            print(f"{filename}: {len(rows)}")
    _write_dataset_info(output)
    print(f"exported strict LLaMA-Factory data: {output}")


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    work = config_path(config, "paths.work_dir")
    tasks = config_path(config, "paths.task_dir")
    protocol = config_path(config, "paths.protocol_dir")
    paths = {
        "raw_root": config_path(config, "paths.raw_root"),
        "official_root": config_path(config, "paths.official_root"),
        "protocol_dir": protocol,
        "protocol_manifest": protocol / "protocol_manifest.json",
        "work_dir": work,
        "task_dir": tasks,
        "llamafactory_data_dir": config_path(config, "paths.llamafactory_data_dir"),
        "proactive_train_tasks": tasks / "proactive_train_config.jsonl",
        "proactive_eval_tasks": tasks / "proactive_eval_config.jsonl",
        "execution_train_tasks": tasks / "execution_train_config.jsonl",
        "execution_eval_tasks": tasks / "execution_eval_config.jsonl",
        "episodes": work / "papo_episodes.jsonl",
        "steps": work / "papo_steps.jsonl",
        "raw_audit": work / "papo_raw_audit.json",
        "trees": work / "papo_trees.jsonl",
        "eval_trees": work / "papo_eval_trees.jsonl",
        "scored_trees": work / "papo_scored_trees.jsonl",
        "eval_scored_trees": work / "papo_eval_scored_trees.jsonl",
        "action_values": work / "papo_action_values.jsonl",
        "eval_action_values": work / "papo_eval_action_values.jsonl",
        "listwise": work / "papo_listwise_targets.jsonl",
        "eval_listwise": work / "papo_eval_listwise_targets.jsonl",
        "pairs": work / "papo_dpo_pairs.jsonl",
        "eval_pairs": work / "papo_eval_dpo_pairs.jsonl",
    }
    for name, filename in PROTOCOL_FILES.items():
        paths[name] = protocol / filename
    return paths


def _require_protocol(paths: dict[str, Path]) -> None:
    missing = [str(paths["protocol_manifest"])]
    if paths["protocol_manifest"].exists():
        missing = [str(path) for name, path in paths.items() if name in PROTOCOL_FILES and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Strict data protocol is missing. Run `python scripts/14_build_data_protocol.py --config config.yaml` first. "
            f"Missing: {missing}"
        )


def _root_steps(
    steps: list[dict[str, Any]],
    episode_ids: set[str],
    root_only: bool,
    limit: int,
) -> list[dict[str, Any]]:
    roots = [step for step in steps if str(step.get("episode_id") or "") in episode_ids]
    if root_only:
        roots = [step for step in roots if int(step.get("step_index", 0) or 0) == 0]
    return roots[:limit] if limit > 0 else roots


def _make_trees(
    roots: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    context: Any,
    tree_config: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    trees: list[dict[str, Any]] = []
    progress_every = int(config["data"].get("progress_every", 25))
    print(f"train trees: building {len(roots)} offline counterfactual trees...", flush=True)
    for index, root in enumerate(roots, start=1):
        trees.append(
            build_offline_counterfactual_tree(
                root,
                steps,
                max_depth=int(tree_config["max_depth"]),
                same_user_k=int(tree_config["same_user_k"]),
                cross_user_k=int(tree_config["cross_user_k"]),
                max_candidates=int(tree_config["max_candidates"]),
                context=context,
            )
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(roots)):
            print(f"train tree progress: {index}/{len(roots)}", flush=True)
    return trees


def _write_dataset_info(output: Path) -> None:
    (output / "dataset_info.json").write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _attach_canonical_reference_actions(tasks: list[dict[str, Any]], steps: list[dict[str, Any]]) -> None:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        by_episode.setdefault(str(step.get("episode_id") or ""), []).append(step)
    canonical: dict[str, list[str]] = {}
    for episode_id, episode_steps in by_episode.items():
        episode_steps.sort(key=lambda row: int(row.get("step_index", 0) or 0))
        canonical[episode_id] = [
            str(row.get("action") or "") for row in episode_steps if str(row.get("action") or "")
        ]

    updated = 0
    missing = 0
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        for key in ["same_user_action_references", "cross_user_action_references"]:
            for reference in inputs.get(key, []):
                actions = canonical.get(str(reference.get("episode_id") or ""))
                if actions:
                    reference["actions"] = actions
                    updated += 1
                else:
                    missing += 1
    print(f"canonicalized reference trajectories: {updated}, missing: {missing}")


def _task_episode_id(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if metadata.get("papo_episode_id"):
        return str(metadata["papo_episode_id"])
    inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
    return f"{inputs.get('user_id', '')}__{inputs.get('time', '')}"


def _task_episode_pool(tasks: list[dict[str, Any]]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for task in tasks:
        inputs = task.get("input") if isinstance(task.get("input"), dict) else {}
        selected.add((str(inputs.get("user_id") or ""), str(inputs.get("time") or "")))
        for key in ["same_user_action_references", "cross_user_action_references"]:
            for reference in inputs.get(key, []):
                selected.add((str(reference.get("user_id") or ""), str(reference.get("time") or "")))
    return selected


if __name__ == "__main__":
    main()
