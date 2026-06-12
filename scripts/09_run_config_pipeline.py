from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import config_path, load_config  # noqa: E402
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
from papo.official_data import read_csv_rows  # noqa: E402
from papo.papo_objective import propagate_residual_values, score_tree_leaves  # noqa: E402
from papo.raw_builder import build_from_raw  # noqa: E402
from papo.tasks import build_personalized_execution_tasks, build_proactive_suggestion_tasks  # noqa: E402
from papo.tree_builder import build_offline_counterfactual_tree, build_tree_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the config-driven PAPO offline pipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--stages",
        default="tasks,raw,trees,objective,export",
        help="Comma-separated: tasks,raw,trees,objective,export",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit per stage.")
    args = parser.parse_args()

    config = load_config(args.config)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    paths = _paths(config)
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


def _build_proactive_tasks(
    config: dict[str, Any],
    paths: dict[str, Path],
    limit: int,
) -> None:
    data = config["data"]

    proactive = build_proactive_suggestion_tasks(
        paths["official_root"] / data["suggestion_split"],
        paths["official_root"] / "total.csv",
        paths["official_root"] / "user_profile.csv",
        paths["raw_root"],
        screenshot_level=int(data["suggestion_screenshot_level"]),
        history_limit=int(data["suggestion_history_limit"]),
        limit=limit,
        require_complete=bool(data["require_complete"]),
    )

    write_jsonl(paths["suggestion_tasks"], proactive)
    print(f"proactive tasks: {len(proactive)}")


def _export_proactive(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    suggestion_tasks = read_jsonl(paths["suggestion_tasks"])
    output = paths["llamafactory_data_dir"]
    output.mkdir(parents=True, exist_ok=True)

    proactive = export_proactive_sft(
        suggestion_tasks,
        paths["raw_root"],
        str(config["paths"]["asset_prefix"]),
    )

    write_json_array(
        output / "papo_proactive_sft.json",
        proactive,
    )

    print(
        "exported proactive LLaMA-Factory data:",
        output / "papo_proactive_sft.json",
    )
    print("proactive SFT:", len(proactive))


def _build_tasks(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    data = config["data"]
    proactive = build_proactive_suggestion_tasks(
        paths["official_root"] / data["suggestion_split"],
        paths["official_root"] / "total.csv",
        paths["official_root"] / "user_profile.csv",
        paths["raw_root"],
        screenshot_level=int(data["suggestion_screenshot_level"]),
        history_limit=int(data["suggestion_history_limit"]),
        limit=limit,
        require_complete=bool(data["require_complete"]),
    )
    execution = build_personalized_execution_tasks(
        paths["official_root"] / data["execution_split"],
        paths["official_root"] / "total.csv",
        paths["official_root"] / "user_profile.csv",
        paths["raw_root"],
        limit=limit,
        require_complete=bool(data["require_complete"]),
        same_user_top_k=int(config["papo"]["references"]["same_user_top_k"]),
        cross_user_top_k=int(config["papo"]["references"]["cross_user_top_k"]),
        intent_similarity_threshold=float(config["papo"]["references"]["intent_similarity_threshold"]),
        exclude_same_intent=bool(config["papo"]["references"]["exclude_same_intent"]),
    )
    write_jsonl(paths["suggestion_tasks"], proactive)
    write_jsonl(paths["execution_tasks"], execution)
    print(f"tasks: proactive={len(proactive)}, execution={len(execution)}")


def _build_raw(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    if limit > 0 and paths["execution_tasks"].exists():
        selected = _task_episode_pool(read_jsonl(paths["execution_tasks"])[:limit])
        max_episodes = 0
    else:
        selected: set[tuple[str, str]] = set()
        for split in config["data"]["retrieval_splits"]:
            selected.update(
                (str(row.get("user_id") or ""), str(row.get("time") or ""))
                for row in read_csv_rows(paths["official_root"] / split)
            )
        max_episodes = int(config["data"]["max_episodes"])
    episodes, steps, audit = build_from_raw(
        paths["raw_root"],
        max_episodes=max_episodes,
        require_complete=bool(config["data"]["require_complete"]),
        selected_episodes=selected,
        max_episodes_per_user=int(config["data"]["max_episodes_per_user"]),
        progress_every=int(config["data"].get("progress_every", 25)),
    )
    write_jsonl(paths["episodes"], episodes)
    write_jsonl(paths["steps"], steps)
    write_json(paths["raw_audit"], audit)
    print(f"raw: episodes={len(episodes)}, steps={len(steps)}")


def _build_trees(config: dict[str, Any], paths: dict[str, Path], limit: int) -> None:
    steps = read_jsonl(paths["steps"])
    execution_ids = {
        f"{row.get('user_id', '')}__{row.get('time', '')}"
        for row in read_csv_rows(paths["official_root"] / config["data"]["execution_split"])
    }
    roots = [step for step in steps if str(step.get("episode_id") or "") in execution_ids]
    if bool(config["data"]["root_only"]):
        roots = [step for step in roots if int(step.get("step_index", 0) or 0) == 0]
    if limit > 0:
        roots = roots[:limit]
    context = build_tree_context(steps)
    tree_config = config["papo"]["tree"]
    trees: list[dict[str, Any]] = []
    total_roots = len(roots)
    progress_every = int(config["data"].get("progress_every", 25))
    print(f"trees: building {total_roots} offline counterfactual trees...", flush=True)
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
        if progress_every > 0 and (index % progress_every == 0 or index == total_roots):
            print(f"tree progress: {index}/{total_roots}", flush=True)
    write_jsonl(paths["trees"], trees)
    print(f"trees: {len(trees)}")


def _build_objective(config: dict[str, Any], paths: dict[str, Path]) -> None:
    tasks = read_jsonl(paths["execution_tasks"])
    _attach_canonical_reference_actions(tasks, read_jsonl(paths["steps"]))
    task_by_episode = {
        str(task.get("metadata", {}).get("papo_episode_id") or _task_episode_id(task)): task
        for task in tasks
    }
    scored_trees = [
        score_tree_leaves(tree, task_by_episode.get(str(tree.get("episode_id") or "")), config["papo"]["reward"])
        for tree in read_jsonl(paths["trees"])
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
    write_jsonl(paths["scored_trees"], scored_trees)
    write_jsonl(paths["action_values"], action_values)
    write_jsonl(paths["listwise"], listwise)
    write_jsonl(paths["pairs"], pairs)
    print(f"objective: action_values={len(action_values)}, listwise={len(listwise)}, pairs={len(pairs)}")


def _export_llamafactory(config: dict[str, Any], paths: dict[str, Path]) -> None:
    suggestion_tasks = read_jsonl(paths["suggestion_tasks"])
    execution_tasks = read_jsonl(paths["execution_tasks"])
    steps = read_jsonl(paths["steps"])
    attach_prior_actions(execution_tasks, steps)
    output = paths["llamafactory_data_dir"]
    asset_prefix = str(config["paths"]["asset_prefix"])
    output.mkdir(parents=True, exist_ok=True)
    write_json_array(
        output / "papo_proactive_sft.json",
        export_proactive_sft(suggestion_tasks, paths["raw_root"], asset_prefix),
    )
    write_json_array(
        output / "papo_execution_sft.json",
        export_execution_sft(execution_tasks, steps, paths["raw_root"], asset_prefix),
    )
    write_json_array(
        output / "papo_execution_dpo.json",
        export_execution_dpo(execution_tasks, steps, read_jsonl(paths["pairs"]), paths["raw_root"], asset_prefix),
    )
    write_json_array(
        output / "papo_execution_listwise.json",
        export_execution_listwise(
            execution_tasks, steps, read_jsonl(paths["listwise"]), paths["raw_root"], asset_prefix
        ),
    )
    (output / "dataset_info.json").write_text(json.dumps(dataset_info(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported LLaMA-Factory data: {output}")


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    work = config_path(config, "paths.work_dir")
    tasks = config_path(config, "paths.task_dir")
    return {
        "raw_root": config_path(config, "paths.raw_root"),
        "official_root": config_path(config, "paths.official_root"),
        "work_dir": work,
        "task_dir": tasks,
        "llamafactory_data_dir": config_path(config, "paths.llamafactory_data_dir"),
        "suggestion_tasks": tasks / "proactive_config.jsonl",
        "execution_tasks": tasks / "execution_config.jsonl",
        "episodes": work / "papo_episodes.jsonl",
        "steps": work / "papo_steps.jsonl",
        "raw_audit": work / "papo_raw_audit.json",
        "trees": work / "papo_trees.jsonl",
        "scored_trees": work / "papo_scored_trees.jsonl",
        "action_values": work / "papo_action_values.jsonl",
        "listwise": work / "papo_listwise_targets.jsonl",
        "pairs": work / "papo_dpo_pairs.jsonl",
    }


def _attach_canonical_reference_actions(
    tasks: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> None:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        episode_id = str(step.get("episode_id") or "")
        by_episode.setdefault(episode_id, []).append(step)

    canonical: dict[str, list[str]] = {}
    for episode_id, episode_steps in by_episode.items():
        episode_steps.sort(key=lambda row: int(row.get("step_index", 0) or 0))
        canonical[episode_id] = [
            str(row.get("action") or "")
            for row in episode_steps
            if str(row.get("action") or "")
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
