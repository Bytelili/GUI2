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
from src.papo.official_data import complete_raw_index, read_csv_rows  # noqa: E402


BLUE = "#2878B5"
LIGHT_BLUE = "#65A9D7"
RED = "#D95F59"
GREEN = "#82B366"
GOLD = "#E5A84B"
PURPLE = "#7B61A8"
GRAY = "#707070"
ACTION_TYPES = ["click", "scroll", "type", "wait", "finished", "press_back", "press_home", "long_click", "press_recent"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the complete FingerTip dataset for paper introduction.")
    parser.add_argument("--raw_root", default=r"D:\0608DataSet\Raw")
    parser.add_argument("--official_root", default=str(PROJECT_ROOT / "data/official/fingertip20k"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs/visualizations/total_dataset"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "total_episode_statistics.jsonl"
    if cache_path.exists() and not args.force_rebuild:
        episodes = _read_jsonl(cache_path)
    else:
        episodes = _build_episode_statistics(Path(args.raw_root), Path(args.official_root))
        _write_jsonl(cache_path, episodes)

    _apply_style()
    manifest = {"source": "total.csv complete episodes", "num_episodes": len(episodes), "figures": []}
    manifest["figures"].append(_plot_scale_dashboard(episodes, out_dir, args.dpi))
    manifest["figures"].append(_plot_user_longitudinal_coverage(episodes, out_dir, args.dpi))
    manifest["figures"].append(_plot_task_context_diversity(episodes, out_dir, args.dpi))
    manifest["figures"].append(_plot_total_action_lengths(episodes, out_dir, args.dpi))
    manifest["figures"].append(_plot_user_behavior_fingerprints(episodes, out_dir, args.dpi))
    pair_rows = _exact_intent_pairs(episodes)
    manifest["figures"].append(_plot_exact_intent_personalization(pair_rows, out_dir, args.dpi))
    write_json(out_dir / "total_visualization_manifest.json", manifest)
    _write_total_summary(out_dir / "total_visualization_summary.md", episodes, pair_rows)
    print(f"complete episodes: {len(episodes)}")
    print(f"same/cross exact-intent pairs: {len(pair_rows)}")
    print(f"figures: {len(manifest['figures'])}")
    print(f"wrote: {out_dir}")


def _build_episode_statistics(raw_root: Path, official_root: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(official_root / "total.csv")
    raw_index = complete_raw_index(raw_root)
    output = []
    for index, row in enumerate(rows, 1):
        key = (str(row.get("user_id") or ""), str(row.get("time") or ""))
        episode_dir = raw_index.get(key)
        if episode_dir is None:
            continue
        actions = _read_actions(episode_dir / "action.jsonl")
        action_counts = Counter(_action_type(action) for action in actions)
        output.append(
            {
                "episode_id": f"{key[0]}__{key[1]}",
                "user_id": key[0],
                "time": key[1],
                "date": key[1][:8],
                "scenario": str(row.get("scenario") or ""),
                "app": str(row.get("app") or ""),
                "intent": str(row.get("intentDescription") or ""),
                "intent_class": str(row.get("intentClass") or ""),
                "num_actions": len(actions),
                "actions": actions,
                "action_counts": dict(action_counts),
            }
        )
        if index % 1000 == 0:
            print(f"total-data progress: {index}/{len(rows)}", flush=True)
    return output


def _plot_scale_dashboard(episodes: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    values = [
        len(episodes),
        len({row["user_id"] for row in episodes}),
        len({row["app"] for row in episodes}),
        len({row["intent_class"] for row in episodes}),
        sum(row["num_actions"] for row in episodes),
    ]
    labels = ["Complete\nepisodes", "Users", "Apps", "Intent\ncategories", "Action steps"]
    colors = [BLUE, LIGHT_BLUE, RED, GREEN, PURPLE]
    fig, axis = plt.subplots(figsize=(10.2, 4.1), constrained_layout=True)
    bars = axis.bar(range(len(values)), values, color=colors, width=0.68)
    axis.set_yscale("log")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Count (log scale)")
    axis.set_title("FingerTip-20K provides large-scale longitudinal mobile interaction data")
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4, fontsize=10)
    axis.grid(axis="y", alpha=0.2)
    paths = _save(fig, out_dir, "total_fig01_dataset_scale", dpi)
    _write_csv(out_dir / "total_fig01_dataset_scale.csv", [{"metric": label.replace("\n", " "), "count": value} for label, value in zip(labels, values)])
    return {"name": "dataset_scale", "files": paths}


def _plot_user_longitudinal_coverage(episodes: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_user[row["user_id"]].append(row)
    counts = sorted([len(rows) for rows in by_user.values()], reverse=True)
    spans = []
    for user, rows in by_user.items():
        dates = sorted(row["date"] for row in rows)
        spans.append({"user_id": user, "episodes": len(rows), "active_days": len(set(dates))})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    axes[0].bar(range(1, len(counts) + 1), counts, color=BLUE)
    axes[0].axhline(mean(counts), color=RED, linewidth=1.8, label=f"Mean = {mean(counts):.1f}")
    axes[0].set(xlabel="Users ranked by episode count", ylabel="Complete episodes", title="Longitudinal coverage across users")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    x = [row["active_days"] for row in spans]
    y = [row["episodes"] for row in spans]
    axes[1].scatter(x, y, color=GREEN, alpha=0.75, s=30, edgecolor="white", linewidth=0.4)
    axes[1].set(xlabel="Active collection days", ylabel="Complete episodes", title="Repeated observations enable preference modeling")
    axes[1].grid(alpha=0.2)
    paths = _save(fig, out_dir, "total_fig02_longitudinal_user_coverage", dpi)
    _write_csv(out_dir / "total_fig02_longitudinal_user_coverage.csv", spans)
    return {"name": "longitudinal_user_coverage", "files": paths}


def _plot_task_context_diversity(episodes: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    classes = Counter(row["intent_class"] for row in episodes)
    scenarios = Counter(_normalize_scenario(row["scenario"]) for row in episodes)
    top_classes = classes.most_common(12)
    top_scenarios = scenarios.most_common(10)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), constrained_layout=True)
    axes[0].barh([item[0] for item in reversed(top_classes)], [item[1] for item in reversed(top_classes)], color=PURPLE)
    axes[0].set(xlabel="Episodes", title="Top intent categories")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh([item[0] for item in reversed(top_scenarios)], [item[1] for item in reversed(top_scenarios)], color=GOLD)
    axes[1].set(xlabel="Episodes", title="Real-life collection scenarios")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("The complete dataset spans diverse tasks and everyday contexts", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "total_fig03_task_context_diversity", dpi)
    _write_csv(out_dir / "total_fig03_intent_categories.csv", [{"intent_class": key, "count": value} for key, value in classes.most_common()])
    _write_csv(out_dir / "total_fig03_scenarios.csv", [{"scenario": key, "count": value} for key, value in scenarios.most_common()])
    return {"name": "task_context_diversity", "files": paths}


def _plot_total_action_lengths(episodes: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    lengths = np.array([row["num_actions"] for row in episodes])
    clipped = np.minimum(lengths, 40)
    fig, axis = plt.subplots(figsize=(9.2, 4.3), constrained_layout=True)
    axis.hist(clipped, bins=np.arange(0.5, 41.5, 1), color=BLUE, alpha=0.85)
    axis.axvline(float(np.median(lengths)), color=RED, linewidth=2, label=f"Median = {np.median(lengths):.0f}")
    axis.axvline(float(np.mean(lengths)), color=GRAY, linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(lengths):.1f}")
    axis.set(xlabel="Trajectory length (40 includes longer episodes)", ylabel="Episodes", title="Human demonstrations contain multi-step mobile workflows")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    paths = _save(fig, out_dir, "total_fig04_action_length_distribution", dpi)
    _write_csv(out_dir / "total_fig04_action_length_distribution.csv", [{"episode_id": row["episode_id"], "num_actions": row["num_actions"]} for row in episodes])
    return {"name": "action_length_distribution", "files": paths}


def _plot_user_behavior_fingerprints(episodes: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    by_user: dict[str, Counter[str]] = defaultdict(Counter)
    for row in episodes:
        by_user[row["user_id"]].update(row["action_counts"])
    selected_users = sorted(by_user, key=lambda user: sum(by_user[user].values()), reverse=True)[:30]
    matrix = []
    for user in selected_users:
        total = sum(by_user[user].values())
        matrix.append([by_user[user][action] / max(total, 1) for action in ACTION_TYPES[:6]])
    fig, axis = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=max(max(row) for row in matrix))
    axis.set_xticks(range(6), ACTION_TYPES[:6], rotation=25, ha="right")
    axis.set_yticks(range(len(selected_users)), selected_users)
    axis.set(xlabel="Fraction of user's actions", ylabel="User ID", title="Users exhibit distinct interaction-style fingerprints")
    fig.colorbar(image, ax=axis, label="Action fraction", shrink=0.8)
    paths = _save(fig, out_dir, "total_fig05_user_behavior_fingerprints", dpi)
    rows = [{"user_id": user, **{action: value for action, value in zip(ACTION_TYPES[:6], values)}} for user, values in zip(selected_users, matrix)]
    _write_csv(out_dir / "total_fig05_user_behavior_fingerprints.csv", rows)
    return {"name": "user_behavior_fingerprints", "files": paths}


def _plot_exact_intent_personalization(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    same = np.array([row["same_user_similarity"] for row in rows])
    cross = np.array([row["cross_user_similarity"] for row in rows])
    gains = same - cross
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    data = [same, cross]
    boxes = axes[0].boxplot(data, labels=["Same user", "Cross user"], patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], [BLUE, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set(ylabel="Trajectory similarity", title="Exactly matched intents still admit user-specific paths")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].hist(gains, bins=np.linspace(-1, 1, 30), color=BLUE, alpha=0.85)
    axes[1].axvline(0, color=GRAY, linestyle="--")
    axes[1].axvline(float(gains.mean()), color=RED, linewidth=2, label=f"Mean gain = {gains.mean():.3f}")
    axes[1].set(xlabel="Same-user minus cross-user similarity", ylabel="Exact-intent groups", title=f"Same-user wins on {(gains > 0).mean() * 100:.1f}% of comparisons")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Full-dataset evidence of personalized execution behavior", fontsize=13, weight="bold")
    paths = _save(fig, out_dir, "total_fig06_exact_intent_personalization", dpi)
    _write_csv(out_dir / "total_fig06_exact_intent_personalization.csv", rows)
    return {"name": "exact_intent_personalization", "files": paths}


def _exact_intent_pairs(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        key = re.sub(r"\s+", "", row["intent"]).lower()
        if key:
            by_intent[key].append(row)
    output = []
    for intent_key, rows in by_intent.items():
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_user[row["user_id"]].append(row)
        users = sorted(by_user)
        if len(users) < 2:
            continue
        for user in users:
            user_rows = sorted(by_user[user], key=lambda row: row["time"])
            if len(user_rows) < 2:
                continue
            target, same_ref = user_rows[-1], user_rows[-2]
            cross_candidates = [item for other in users if other != user for item in by_user[other]]
            cross_ref = min(cross_candidates, key=lambda item: abs(len(item["actions"]) - len(target["actions"])))
            output.append(
                {
                    "intent": target["intent"],
                    "target_episode_id": target["episode_id"],
                    "same_episode_id": same_ref["episode_id"],
                    "cross_episode_id": cross_ref["episode_id"],
                    "same_user_similarity": sequence_similarity(target["actions"], same_ref["actions"]),
                    "cross_user_similarity": sequence_similarity(target["actions"], cross_ref["actions"]),
                }
            )
    return output


def _read_actions(path: Path) -> list[str]:
    actions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        actions.extend(str(value) for value in item.values()) if isinstance(item, dict) else actions.append(str(item))
    return actions


def _action_type(action: str) -> str:
    match = re.match(r"\s*([A-Za-z_]+)", action)
    return match.group(1).lower() if match else "unknown"


def _normalize_scenario(value: str) -> str:
    mappings = {
        "快递点": "快递站", "快递驿站": "快递站", "驿站": "快递站", "快递柜": "快递站",
        "快递柜附近": "快递站", "小区楼底": "其他", "停车场": "其他", "地下车库": "其他",
        "他人住所": "住所", "电梯": "其他", "高铁站": "交通工具", "旅店": "其他", "银行": "其他",
        "小卖部": "商场",
    }
    return mappings.get(value, value or "其他")


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_total_summary(path: Path, episodes: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    gains = np.array([row["same_user_similarity"] - row["cross_user_similarity"] for row in pairs])
    text = f"""# Total-Dataset Visualization Summary

These figures use all `{len(episodes):,}` complete episodes from `total.csv`.
They describe dataset properties and personalized-behavior evidence; they are
not model-performance results.

- Users: `{len({row['user_id'] for row in episodes})}`
- Apps: `{len({row['app'] for row in episodes})}`
- Intent categories: `{len({row['intent_class'] for row in episodes})}`
- Action steps: `{sum(row['num_actions'] for row in episodes):,}`
- Exact-intent personalized comparisons: `{len(pairs):,}`
- Same-user wins in exact-intent comparisons: `{(gains > 0).mean() * 100:.1f}%`
- Mean exact-intent personalization gain: `{gains.mean():.3f}`

Use these total-dataset figures in the Introduction or dataset analysis.
Continue using `test_execution.csv` figures for baseline and PAPO performance.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
