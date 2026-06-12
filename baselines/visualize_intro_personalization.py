from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import sequence_similarity, write_json  # noqa: E402


BLUE = "#2878B5"
LIGHT_BLUE = "#65A9D7"
RED = "#D95F59"
GREEN = "#82B366"
GOLD = "#E5A84B"
GRAY = "#707070"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Introduction figures motivating personalized GUI agents.")
    parser.add_argument(
        "--episode_cache",
        default=str(PROJECT_ROOT / "outputs/visualizations/total_dataset/total_episode_statistics.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "outputs/visualizations/introduction"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    episodes = _read_jsonl(Path(args.episode_cache))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = _build_repeated_intent_comparisons(episodes)
    exact_groups = _exact_intent_groups(episodes)

    _apply_style()
    manifest = {"source": str(args.episode_cache), "figures": []}
    manifest["figures"].append(_plot_personalization_opportunity(comparisons, out_dir, args.dpi))
    manifest["figures"].append(_plot_same_task_multiple_paths(exact_groups, out_dir, args.dpi))
    manifest["figures"].append(_plot_cross_category_signal(comparisons, out_dir, args.dpi))
    manifest["figures"].append(_plot_history_depth(comparisons, out_dir, args.dpi))
    manifest["figures"].append(_plot_intro_composite(comparisons, exact_groups, out_dir, args.dpi))
    write_json(out_dir / "introduction_visualization_manifest.json", manifest)
    _write_summary(out_dir / "introduction_visualization_summary.md", comparisons, exact_groups)
    print(f"repeated-intent comparisons: {len(comparisons)}")
    print(f"multi-user exact-intent groups: {len(exact_groups)}")
    print(f"figures: {len(manifest['figures'])}")
    print(f"wrote: {out_dir}")


def _build_repeated_intent_comparisons(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_intent[_intent_key(row["intent"])].append(row)

    output = []
    for intent_key, intent_rows in by_intent.items():
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in intent_rows:
            by_user[row["user_id"]].append(row)
        if len(by_user) < 2:
            continue
        for rows in by_user.values():
            rows.sort(key=lambda item: item["time"])
        for user_id, user_rows in by_user.items():
            if len(user_rows) < 2:
                continue
            cross_pool = [row for other_user, rows in by_user.items() if other_user != user_id for row in rows]
            for target_index in range(1, len(user_rows)):
                target = user_rows[target_index]
                same_candidates = user_rows[:target_index]
                same_ref = max(same_candidates, key=lambda row: sequence_similarity(target["actions"], row["actions"]))
                cross_ref = max(cross_pool, key=lambda row: sequence_similarity(target["actions"], row["actions"]))
                same_similarity = sequence_similarity(target["actions"], same_ref["actions"])
                cross_similarity = sequence_similarity(target["actions"], cross_ref["actions"])
                output.append(
                    {
                        "intent": target["intent"],
                        "intent_class": target["intent_class"],
                        "user_id": user_id,
                        "target_episode_id": target["episode_id"],
                        "same_episode_id": same_ref["episode_id"],
                        "cross_episode_id": cross_ref["episode_id"],
                        "prior_same_intent_count": target_index,
                        "same_user_similarity": same_similarity,
                        "best_cross_user_similarity": cross_similarity,
                        "personalization_gain": same_similarity - cross_similarity,
                    }
                )
    return output


def _exact_intent_groups(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[_intent_key(row["intent"])].append(row)
    output = []
    for intent_key, rows in grouped.items():
        users = {row["user_id"] for row in rows}
        if len(users) < 2:
            continue
        type_sequences = {
            " > ".join(_action_type(action) for action in row["actions"])
            for row in rows
        }
        output.append(
            {
                "intent": rows[0]["intent"],
                "intent_class": rows[0]["intent_class"],
                "num_episodes": len(rows),
                "num_users": len(users),
                "num_distinct_action_type_paths": len(type_sequences),
                "distinct_path_ratio": len(type_sequences) / len(rows),
            }
        )
    return output


def _plot_personalization_opportunity(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    same = np.array([row["same_user_similarity"] for row in rows])
    cross = np.array([row["best_cross_user_similarity"] for row in rows])
    gains = same - cross
    meaningful = gains > 0.1
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].scatter(cross, same, color=BLUE, alpha=0.55, s=22, edgecolor="white", linewidth=0.3)
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1.2)
    axes[0].fill_between([0, 1], [0.1, 1.1], [0, 1], color=BLUE, alpha=0.06)
    axes[0].set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Best cross-user path similarity", ylabel="Same-user history similarity")
    axes[0].set_title("A generic path often misses the user's preferred path")
    axes[0].grid(alpha=0.2)

    axes[1].hist(gains, bins=np.linspace(-0.8, 0.8, 30), color=BLUE, alpha=0.85)
    axes[1].axvline(0, color=GRAY, linestyle="--")
    axes[1].axvline(0.1, color=RED, linewidth=1.8, label="Meaningful gain threshold = 0.1")
    axes[1].set(xlabel="Same-user advantage over best cross-user path", ylabel="Repeated-task episodes")
    axes[1].set_title(f"{meaningful.mean() * 100:.1f}% retain >0.1 personalization opportunity")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("One-size-fits-all execution leaves personalized utility on the table", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "intro_fig01_one_size_fits_all_gap", dpi)
    _write_csv(out_dir / "intro_fig01_one_size_fits_all_gap.csv", rows)
    return {"name": "one_size_fits_all_gap", "files": paths}


def _plot_same_task_multiple_paths(groups: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    counts = Counter(min(int(row["num_distinct_action_type_paths"]), 8) for row in groups)
    x = sorted(counts)
    labels = [str(value) if value < 8 else "8+" for value in x]
    values = [counts[value] for value in x]
    diverse = np.mean([row["num_distinct_action_type_paths"] > 1 for row in groups])
    ratios = np.array([row["distinct_path_ratio"] for row in groups])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    bars = axes[0].bar(labels, values, color=GREEN)
    axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set(xlabel="Distinct action-type paths for the exact same intent", ylabel="Intent groups")
    axes[0].set_title(f"{diverse * 100:.1f}% of exact intents have multiple execution paths")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].hist(ratios, bins=np.linspace(0, 1, 16), color=GOLD, alpha=0.85)
    axes[1].set(xlabel="Distinct path ratio within an exact-intent group", ylabel="Intent groups")
    axes[1].set_title("Identical instructions do not imply a single canonical trajectory")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Personalized execution matters even when users ask for exactly the same task", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "intro_fig02_same_task_multiple_paths", dpi)
    _write_csv(out_dir / "intro_fig02_same_task_multiple_paths.csv", groups)
    return {"name": "same_task_multiple_paths", "files": paths}


def _plot_cross_category_signal(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_class[row["intent_class"]].append(float(row["personalization_gain"]))
    selected = [
        (name, values)
        for name, values in by_class.items()
        if len(values) >= 8
    ]
    selected.sort(key=lambda item: len(item[1]), reverse=True)
    selected = selected[:12]
    labels = [item[0] for item in selected]
    means = [mean(item[1]) for item in selected]
    win_rates = [np.mean(np.array(item[1]) > 0) for item in selected]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    colors = [BLUE if value >= 0 else RED for value in means]
    axes[0].barh(labels[::-1], means[::-1], color=colors[::-1])
    axes[0].axvline(0, color=GRAY, linestyle="--")
    axes[0].set(xlabel="Mean same-user personalization gain", title="Personalization gain appears across task categories")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(labels[::-1], win_rates[::-1], color=LIGHT_BLUE)
    axes[1].axvline(0.5, color=GRAY, linestyle="--")
    axes[1].set(xlim=(0, 1), xlabel="Fraction where same-user history wins", title="The effect is not confined to one app or task type")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("User-specific execution preferences are a broad phenomenon", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "intro_fig03_cross_category_personalization", dpi)
    csv_rows = [{"intent_class": label, "num_comparisons": len(values), "mean_gain": gain, "same_user_win_rate": win} for (label, values), gain, win in zip(selected, means, win_rates)]
    _write_csv(out_dir / "intro_fig03_cross_category_personalization.csv", csv_rows)
    return {"name": "cross_category_personalization", "files": paths}


def _plot_history_depth(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count = int(row["prior_same_intent_count"])
        label = "1 prior" if count == 1 else "2 priors" if count == 2 else "3+ priors"
        buckets[label].append(row)
    labels = ["1 prior", "2 priors", "3+ priors"]
    mean_same = [mean(float(row["same_user_similarity"]) for row in buckets[label]) for label in labels]
    mean_cross = [mean(float(row["best_cross_user_similarity"]) for row in buckets[label]) for label in labels]
    mean_gain = [mean(float(row["personalization_gain"]) for row in buckets[label]) for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(x, mean_same, marker="o", linewidth=2.2, color=BLUE, label="Same-user history")
    axes[0].plot(x, mean_cross, marker="o", linewidth=2.2, color=RED, label="Best cross-user path")
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="Trajectory similarity", title="Longitudinal histories provide increasingly useful references")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    bars = axes[1].bar(labels, mean_gain, color=GREEN)
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].axhline(0, color=GRAY, linestyle="--")
    axes[1].set(ylabel="Mean personalization gain", title="More repeated history strengthens personalized advantage")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Personalization improves as the agent observes more of the same user", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "intro_fig04_history_depth_value", dpi)
    _write_csv(out_dir / "intro_fig04_history_depth_value.csv", [{"history_depth": label, "num_comparisons": len(buckets[label]), "same_similarity": same, "cross_similarity": cross, "gain": gain} for label, same, cross, gain in zip(labels, mean_same, mean_cross, mean_gain)])
    return {"name": "history_depth_value", "files": paths}


def _plot_intro_composite(rows: list[dict[str, Any]], groups: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    gains = np.array([row["personalization_gain"] for row in rows])
    distinct = np.array([row["num_distinct_action_type_paths"] for row in groups])
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1), constrained_layout=True)
    axes[0].hist(distinct, bins=np.arange(0.5, min(distinct.max(), 10) + 1.5), color=GREEN, alpha=0.85)
    axes[0].set(xlabel="Distinct paths for the exact same intent", ylabel="Intent groups", title="Same task, multiple paths")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].hist(gains, bins=np.linspace(-0.8, 0.8, 28), color=BLUE, alpha=0.85)
    axes[1].axvline(0, color=GRAY, linestyle="--")
    axes[1].axvline(float(gains.mean()), color=RED, linewidth=2)
    axes[1].set(xlabel="Same-user personalization gain", ylabel="Repeated-task episodes", title=f"Same-user history wins on {(gains > 0).mean() * 100:.1f}%")
    axes[1].grid(axis="y", alpha=0.2)
    buckets = defaultdict(list)
    for row in rows:
        depth = int(row["prior_same_intent_count"])
        buckets["1" if depth == 1 else "2" if depth == 2 else "3+"].append(row["personalization_gain"])
    labels = ["1", "2", "3+"]
    values = [mean(buckets[label]) for label in labels]
    bars = axes[2].bar(labels, values, color=GOLD)
    axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].set(xlabel="Prior same-user demonstrations", ylabel="Mean personalization gain", title="History makes personalization actionable")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Why personalized GUI agents need user history", fontsize=14, weight="bold")
    paths = _save(fig, out_dir, "intro_fig05_personalization_motivation_composite", dpi)
    return {"name": "personalization_motivation_composite", "files": paths}


def _intent_key(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _action_type(action: str) -> str:
    match = re.match(r"\s*([A-Za-z_]+)", action)
    return match.group(1).lower() if match else "unknown"


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _save(fig: Any, out_dir: Path, stem: str, dpi: int) -> list[str]:
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png), str(pdf)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    gains = np.array([row["personalization_gain"] for row in rows])
    diverse = np.mean([row["num_distinct_action_type_paths"] > 1 for row in groups])
    meaningful = np.mean(gains > 0.1)
    text = f"""# Introduction Personalization Visualization Summary

These figures use real complete episodes from `total.csv` and motivate why
personalized GUI execution matters. They are not model-performance results.

- Multi-user exact-intent groups: `{len(groups):,}`
- Exact intents with multiple action-type paths: `{diverse * 100:.1f}%`
- Repeated-task same-user/cross-user comparisons: `{len(rows):,}`
- Same-user history wins: `{(gains > 0).mean() * 100:.1f}%`
- Mean same-user personalization gain: `{gains.mean():.3f}`
- Comparisons with meaningful gain above 0.1: `{meaningful * 100:.1f}%`

Recommended Introduction figure: `intro_fig05_personalization_motivation_composite`.
It compactly communicates that identical tasks admit multiple paths, same-user
history is more predictive, and longitudinal history makes personalization
actionable.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
