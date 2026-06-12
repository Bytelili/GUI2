from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import (  # noqa: E402
    official_prompt,
    official_screen_description,
    profile_text,
    write_json,
    write_jsonl,
)
from baselines.metrics import retrieval_report  # noqa: E402
from baselines.retrieval import retrieve_all_modes  # noqa: E402
from src.papo.official_data import (  # noqa: E402
    complete_raw_index,
    episode_assets,
    episode_ref,
    load_profiles,
    read_csv_rows,
)
from src.papo.raw_builder import build_from_raw  # noqa: E402


VARIANTS = {
    "no_history": {"profile": False, "reference": None},
    "profile_only": {"profile": True, "reference": None},
    "cross_user_icl": {"profile": True, "reference": "cross_user_top1"},
    "official_icl": {"profile": True, "reference": "same_user_top1"},
    "official_icl_no_same_intent": {"profile": True, "reference": "same_user_no_same_intent"},
}
RETRIEVAL_MODES = [
    "same_user_top1",
    "same_user_no_same_intent",
    "cross_user_top1",
    "cross_user_strict_past",
    "random_same_user",
    "random_cross_user",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline FingerTip personalized-execution baselines.")
    parser.add_argument("--raw_root", default=r"D:\0608DataSet\Raw")
    parser.add_argument("--official_root", default=str(PROJECT_ROOT / "data/official/fingertip20k"))
    parser.add_argument("--test_split", default="test_execution.csv")
    parser.add_argument("--catalog", default="total.csv")
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "data/baselines/execution"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=0, help="Stratified random episode sample.")
    parser.add_argument("--variants", default="", help="Comma-separated prompt variants; empty means all.")
    parser.add_argument("--retrieval_modes", default="", help="Comma-separated retrieval modes; empty means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    official_root = Path(args.official_root)
    out_dir = Path(args.out_dir)
    raw_index = complete_raw_index(raw_root)
    profiles = load_profiles(official_root / "user_profile.csv")
    catalog = [episode_ref(row) for row in read_csv_rows(official_root / args.catalog)]
    targets = [episode_ref(row) for row in read_csv_rows(official_root / args.test_split)]
    targets = [target for target in targets if (target.user_id, target.time) in raw_index]
    if args.sample_size > 0:
        targets = _stratified_sample(targets, args.sample_size, args.seed)
    elif args.limit > 0:
        targets = targets[: args.limit]
    variants = _selected_names(args.variants, VARIANTS)
    retrieval_modes = _selected_names(args.retrieval_modes, {mode: {} for mode in RETRIEVAL_MODES})

    tasks = []
    for index, target in enumerate(targets, 1):
        target_assets = episode_assets(raw_index[(target.user_id, target.time)])
        references = {
            mode: [item.to_dict() for item in items]
            for mode, items in retrieve_all_modes(
                target,
                catalog,
                raw_index,
                retrieval_modes,
                seed=args.seed,
            ).items()
        }
        tasks.append(
            {
                "episode_id": target.episode_id,
                "user_id": target.user_id,
                "time": target.time,
                "instruction": target.intent,
                "scenario": target.scenario,
                "app": target.app,
                "profile": profiles.get(target.user_id, {}),
                "episode_path": target_assets["episode_path"],
                "target_actions": target_assets["actions"],
                "references": references,
            }
        )
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(targets)):
            print(f"retrieval progress: {index}/{len(targets)}", flush=True)

    selected = {(target.user_id, target.time) for target in targets}
    _episodes, steps, audit = build_from_raw(
        raw_root,
        selected_episodes=selected,
        require_complete=True,
        progress_every=args.progress_every,
        discovered_episode_dirs=[
            (target.user_id, target.time, raw_index[(target.user_id, target.time)])
            for target in targets
        ],
    )
    task_by_episode = {task["episode_id"]: task for task in tasks}
    prompt_rows = _build_prompt_rows(steps, task_by_episode, variants)

    write_jsonl(out_dir / "execution_tasks.jsonl", tasks)
    write_jsonl(out_dir / "execution_steps.jsonl", steps)
    write_jsonl(out_dir / "teacher_forced_prompts.jsonl", prompt_rows)
    write_json(out_dir / "retrieval_report.json", retrieval_report(tasks))
    write_json(out_dir / "raw_test_audit.json", audit)
    _write_summary_csv(out_dir / "retrieval_summary.csv", retrieval_report(tasks))
    print(f"tasks: {len(tasks)}")
    print(f"steps: {len(steps)}")
    print(f"prompt rows: {len(prompt_rows)}")
    print(f"wrote: {out_dir}")


def _build_prompt_rows(
    steps: list[dict[str, Any]],
    task_by_episode: dict[str, dict[str, Any]],
    variants: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior_by_episode: dict[str, list[str]] = {}
    for step in steps:
        episode_id = str(step.get("episode_id") or "")
        task = task_by_episode.get(episode_id)
        if task is None or not step.get("valid_observation"):
            continue
        prior = prior_by_episode.setdefault(episode_id, [])
        for variant in variants:
            config = VARIANTS[variant]
            reference_mode = config["reference"]
            references = task["references"].get(reference_mode, []) if reference_mode else []
            reference_actions = list(references[0].get("actions") or []) if references else []
            cross_refs = task["references"].get("cross_user_top1", [])
            cross_actions = list(cross_refs[0].get("actions") or []) if cross_refs else []
            rows.append(
                {
                    "row_id": f"{variant}__{step['papo_step_id']}",
                    "variant": variant,
                    "episode_id": episode_id,
                    "step_id": step["papo_step_id"],
                    "step_index": step["step_index"],
                    "image": step.get("screenshot_path", ""),
                    "xml": step.get("xml_path", ""),
                    "prompt": official_prompt(
                        instruction=task["instruction"],
                        profile=profile_text(task["profile"]) if config["profile"] else "",
                        size=_image_size(step.get("screenshot_path", "")),
                        screen_description=official_screen_description(step.get("xml_path", "")),
                        actions_reference=reference_actions,
                        previous_actions=prior,
                    ),
                    "target_action": step.get("raw_action") or "",
                    "target_semantic_action": step.get("action") or "",
                    "reference_mode": reference_mode or "none",
                    "reference_actions": reference_actions,
                    "cross_user_actions": cross_actions,
                    "prediction": None,
                }
            )
        prior.append(str(step.get("raw_action") or step.get("action") or ""))
    return rows


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = report.get("modes", {})
    rows = [{"mode": mode, **metrics} for mode, metrics in modes.items() if mode != "same_vs_cross"]
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _image_size(path: str) -> str:
    if not path:
        return "unknown"


def _selected_names(raw: str, available: dict[str, Any]) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()] if raw else list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown names: {unknown}. Available: {sorted(available)}")
    return names


def _stratified_sample(targets: list[Any], sample_size: int, seed: int) -> list[Any]:
    if sample_size >= len(targets):
        return targets
    by_user: dict[str, list[Any]] = {}
    for target in targets:
        by_user.setdefault(target.user_id, []).append(target)
    rng = random.Random(seed)
    for rows in by_user.values():
        rng.shuffle(rows)
    exact = {user: sample_size * len(rows) / len(targets) for user, rows in by_user.items()}
    counts = {user: min(len(by_user[user]), int(value)) for user, value in exact.items()}
    remaining = sample_size - sum(counts.values())
    order = sorted(by_user, key=lambda user: (exact[user] - counts[user], rng.random()), reverse=True)
    for user in order:
        if remaining <= 0:
            break
        if counts[user] < len(by_user[user]):
            counts[user] += 1
            remaining -= 1
    selected = [row for user, rows in by_user.items() for row in rows[: counts[user]]]
    selected.sort(key=lambda item: (item.user_id, item.time))
    return selected
    try:
        from PIL import Image

        with Image.open(path) as image:
            return f"{image.size[0]}x{image.size[1]}"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
