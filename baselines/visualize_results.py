from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import sequence_similarity, write_json  # noqa: E402


COLORS = {
    "same": "#2878B5",
    "same_no_intent": "#65A9D7",
    "cross": "#D95F59",
    "random_same": "#82B366",
    "random_cross": "#E5A84B",
    "neutral": "#707070",
    "accent": "#7B61A8",
}

LABELS = {
    "same_user_top1": "Same-user Top-1",
    "same_user_no_same_intent": "Same-user\nno same intent",
    "cross_user_top1": "Cross-user Top-1",
    "cross_user_strict_past": "Cross-user\nstrict past",
    "random_same_user": "Random same-user",
    "random_cross_user": "Random cross-user",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready FingerTip baseline visualizations.")
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "data/baselines/execution"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs/visualizations/baseline"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = _read_jsonl(input_dir / "execution_tasks.jsonl")
    steps = _read_jsonl(input_dir / "execution_steps.jsonl")
    retrieval = _read_json(input_dir / "retrieval_report.json")
    copy_report = _read_json(input_dir / "reference_copy_report.json")

    _apply_style()
    manifest: dict[str, Any] = {"figures": [], "source": str(input_dir)}
    manifest["figures"].append(_plot_retrieval_overview(retrieval, out_dir, args.dpi))
    episode_rows = _episode_retrieval_rows(tasks)
    manifest["figures"].append(_plot_paired_personalization(episode_rows, out_dir, args.dpi))
    manifest["figures"].append(_plot_intent_action_relationship(episode_rows, out_dir, args.dpi))
    manifest["figures"].append(_plot_action_distribution(steps, out_dir, args.dpi))
    manifest["figures"].append(_plot_task_lengths(tasks, out_dir, args.dpi))
    manifest["figures"].append(_plot_copy_tradeoff(copy_report, out_dir, args.dpi))
    manifest["figures"].append(_plot_case_study(tasks, out_dir, args.dpi))
    write_json(out_dir / "visualization_manifest.json", manifest)
    _write_summary(out_dir / "visualization_summary.md", tasks, steps, episode_rows, retrieval, copy_report)
    print(f"figures: {len(manifest['figures'])}")
    print(f"wrote: {out_dir}")


def _plot_retrieval_overview(report: dict[str, Any], out_dir: Path, dpi: int) -> dict[str, Any]:
    modes = [
        "same_user_top1",
        "same_user_no_same_intent",
        "cross_user_top1",
        "random_same_user",
        "random_cross_user",
    ]
    metric_specs = [
        ("intent_similarity", "Intent similarity"),
        ("sequence_similarity", "Trajectory similarity"),
        ("action_levenshtein_similarity", "Action Levenshtein similarity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    palette = [COLORS["same"], COLORS["same_no_intent"], COLORS["cross"], COLORS["random_same"], COLORS["random_cross"]]
    rows = []
    for axis, (metric, title) in zip(axes, metric_specs):
        values = [float(report["modes"][mode][metric]) for mode in modes]
        bars = axis.bar(range(len(modes)), values, color=palette, width=0.72)
        axis.set_title(title)
        axis.set_xticks(range(len(modes)), [LABELS[mode] for mode in modes], rotation=24, ha="right")
        axis.set_ylim(0, max(values) * 1.22)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        rows.extend({"mode": mode, "metric": metric, "value": value} for mode, value in zip(modes, values))
    fig.suptitle("Same-user history contains substantially stronger personalized execution signal", fontsize=13, weight="bold")
    paths = _save_figure(fig, out_dir, "fig01_retrieval_signal_overview", dpi)
    _write_csv(out_dir / "fig01_retrieval_signal_overview.csv", rows)
    return {"name": "retrieval_signal_overview", "files": paths, "paper_use": "Introduction motivation / retrieval analysis"}


def _plot_paired_personalization(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    same = np.array([row["same_sequence_similarity"] for row in rows])
    cross = np.array([row["cross_sequence_similarity"] for row in rows])
    gains = same - cross
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    axes[0].scatter(cross, same, s=24, alpha=0.65, color=COLORS["same"], edgecolor="white", linewidth=0.3)
    limit = max(float(same.max()), float(cross.max()), 1.0)
    axes[0].plot([0, limit], [0, limit], "--", color=COLORS["neutral"], linewidth=1)
    axes[0].set(xlabel="Cross-user trajectory similarity", ylabel="Same-user trajectory similarity")
    axes[0].set_title(f"Same-user wins on {(gains > 0).mean() * 100:.1f}% of episodes")
    axes[0].grid(alpha=0.2)

    axes[1].hist(gains, bins=np.linspace(-1, 1, 25), color=COLORS["same"], alpha=0.85)
    axes[1].axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    axes[1].axvline(float(gains.mean()), color=COLORS["cross"], linewidth=2, label=f"Mean gain = {gains.mean():.3f}")
    axes[1].set(xlabel="Same-user minus cross-user similarity", ylabel="Episodes")
    axes[1].set_title("Per-episode personalization gain")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Personalized signal is visible at the individual episode level", fontsize=13, weight="bold")
    paths = _save_figure(fig, out_dir, "fig02_paired_personalization_gain", dpi)
    _write_csv(out_dir / "fig02_paired_personalization_gain.csv", rows)
    return {"name": "paired_personalization_gain", "files": paths, "paper_use": "Introduction evidence / statistical analysis"}


def _plot_intent_action_relationship(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, prefix, color, title in [
        (axes[0], "same", COLORS["same"], "Same-user retrieval"),
        (axes[1], "cross", COLORS["cross"], "Cross-user retrieval"),
    ]:
        x = np.array([row[f"{prefix}_intent_similarity"] for row in rows])
        y = np.array([row[f"{prefix}_sequence_similarity"] for row in rows])
        axis.scatter(x, y, s=22, alpha=0.55, color=color, edgecolor="white", linewidth=0.25)
        if np.std(x) > 0:
            coefficients = np.polyfit(x, y, 1)
            line_x = np.linspace(0, 1, 100)
            axis.plot(line_x, coefficients[0] * line_x + coefficients[1], color=COLORS["neutral"], linewidth=1.5)
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        axis.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Intent similarity", ylabel="Trajectory similarity")
        axis.set_title(f"{title}\nPearson r = {correlation:.2f}")
        axis.grid(alpha=0.2)
    fig.suptitle("Similar intent helps retrieval, but does not fully determine the user's action path", fontsize=13, weight="bold")
    paths = _save_figure(fig, out_dir, "fig03_intent_vs_trajectory_similarity", dpi)
    _write_csv(out_dir / "fig03_intent_vs_trajectory_similarity.csv", rows)
    return {"name": "intent_vs_trajectory_similarity", "files": paths, "paper_use": "Motivation for preference modeling beyond intent retrieval"}


def _plot_action_distribution(steps: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    counts = Counter(str(step.get("action_type") or "unknown") for step in steps)
    order = [name for name, _count in counts.most_common()]
    values = [counts[name] for name in order]
    fig, axis = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    bars = axis.bar(order, values, color=COLORS["accent"])
    axis.bar_label(bars, labels=[f"{value}\n({value / sum(values):.0%})" for value in values], padding=3, fontsize=8)
    axis.set(ylabel="Action steps", title="Execution test action-space distribution")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(axis="y", alpha=0.2)
    paths = _save_figure(fig, out_dir, "fig04_action_space_distribution", dpi)
    rows = [{"action_type": name, "count": counts[name], "fraction": counts[name] / sum(values)} for name in order]
    _write_csv(out_dir / "fig04_action_space_distribution.csv", rows)
    return {"name": "action_space_distribution", "files": paths, "paper_use": "Dataset / experimental setup"}


def _plot_task_lengths(tasks: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    lengths = np.array([len(task.get("target_actions") or []) for task in tasks])
    bins = np.arange(0.5, min(int(lengths.max()), 30) + 1.5, 1)
    fig, axis = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    axis.hist(np.minimum(lengths, 30), bins=bins, color=COLORS["same"], alpha=0.85)
    axis.axvline(float(np.median(lengths)), color=COLORS["cross"], linewidth=2, label=f"Median = {np.median(lengths):.0f}")
    axis.axvline(float(np.mean(lengths)), color=COLORS["neutral"], linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(lengths):.1f}")
    axis.set(xlabel="Golden trajectory length (steps; 30 includes longer tasks)", ylabel="Episodes", title="Execution tasks span short routines and long multi-step workflows")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    paths = _save_figure(fig, out_dir, "fig05_task_length_distribution", dpi)
    rows = [{"episode_id": task["episode_id"], "length": len(task.get("target_actions") or [])} for task in tasks]
    _write_csv(out_dir / "fig05_task_length_distribution.csv", rows)
    return {"name": "task_length_distribution", "files": paths, "paper_use": "Dataset / difficulty analysis"}


def _plot_copy_tradeoff(report: dict[str, Any], out_dir: Path, dpi: int) -> dict[str, Any]:
    variants = ["cross_user_icl", "official_icl", "official_icl_no_same_intent"]
    colors = [COLORS["cross"], COLORS["same"], COLORS["same_no_intent"]]
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    rows = []
    for variant, color in zip(variants, colors):
        metrics = report["variants"][variant]
        x = float(metrics["action_type_accuracy"])
        y = float(metrics["sequence_up_sim"])
        size = 220 + 80 * min(float(metrics["sequence_sim2"]), 5.0)
        axis.scatter(x, y, s=size, color=color, alpha=0.8, edgecolor="white", linewidth=1.0)
        display_labels = {
            "cross_user_icl": "Cross-user Top-1",
            "official_icl": "Same-user Top-1",
            "official_icl_no_same_intent": "Same-user, no same intent",
        }
        axis.annotate(display_labels[variant], (x, y), xytext=(7, 6), textcoords="offset points", fontsize=9)
        rows.append({"variant": variant, **metrics})
    axis.set(xlabel="Next-action type accuracy", ylabel="Similarity to target-user trajectory", title="Retrieval-only policy reveals utility and personalization trade-offs")
    axis.grid(alpha=0.2)
    paths = _save_figure(fig, out_dir, "fig06_reference_copy_tradeoff", dpi)
    _write_csv(out_dir / "fig06_reference_copy_tradeoff.csv", rows)
    return {"name": "reference_copy_tradeoff", "files": paths, "paper_use": "Baseline diagnostic; do not present as a trained agent"}


def _plot_case_study(tasks: list[dict[str, Any]], out_dir: Path, dpi: int) -> dict[str, Any]:
    ranked = sorted(
        tasks,
        key=lambda task: _case_gain(task),
        reverse=True,
    )
    task = next(
        (
            item for item in ranked
            if len(item.get("target_actions") or []) <= 10
            and item.get("references", {}).get("same_user_top1")
            and item.get("references", {}).get("cross_user_top1")
        ),
        ranked[0],
    )
    target = list(task["target_actions"])
    same = list(task["references"]["same_user_top1"][0]["actions"])
    cross = list(task["references"]["cross_user_top1"][0]["actions"])
    columns = [
        ("Target user", target, COLORS["neutral"]),
        ("Same-user history", same, COLORS["same"]),
        ("Cross-user history", cross, COLORS["cross"]),
    ]
    max_steps = min(max(len(actions) for _label, actions, _color in columns), 12)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, max(4.5, 0.36 * max_steps + 2)), constrained_layout=True)
    for axis, (label, actions, color) in zip(axes, columns):
        axis.set_xlim(0, 1)
        axis.set_ylim(max_steps + 0.7, -0.7)
        axis.axis("off")
        axis.set_title(f"{label}\n{len(actions)} steps", color=color, weight="bold")
        for index, action in enumerate(actions[:max_steps]):
            text = _short_action(action)
            axis.text(
                0.02,
                index,
                f"{index + 1}. {text}",
                va="center",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "alpha": 0.11, "edgecolor": color},
            )
        if len(actions) > max_steps:
            axis.text(0.02, max_steps, f"... +{len(actions) - max_steps} steps", fontsize=8, color=COLORS["neutral"])
    same_sim = sequence_similarity(target, same)
    cross_sim = sequence_similarity(target, cross)
    fig.suptitle(
        f"Case study: {task['instruction']}\nSame-user similarity {same_sim:.3f} vs cross-user {cross_sim:.3f}",
        fontsize=12,
        weight="bold",
    )
    paths = _save_figure(fig, out_dir, "fig07_personalized_trajectory_case_study", dpi)
    write_json(
        out_dir / "fig07_personalized_trajectory_case_study.json",
        {"episode_id": task["episode_id"], "instruction": task["instruction"], "target": target, "same_user": same, "cross_user": cross},
    )
    return {"name": "personalized_trajectory_case_study", "files": paths, "paper_use": "Introduction teaser / qualitative analysis"}


def _episode_retrieval_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        same = task["references"]["same_user_top1"][0]
        cross = task["references"]["cross_user_top1"][0]
        target_actions = task["target_actions"]
        rows.append(
            {
                "episode_id": task["episode_id"],
                "user_id": task["user_id"],
                "instruction": task["instruction"],
                "same_intent_similarity": same["intent_similarity"],
                "same_sequence_similarity": sequence_similarity(target_actions, same["actions"]),
                "cross_intent_similarity": cross["intent_similarity"],
                "cross_sequence_similarity": sequence_similarity(target_actions, cross["actions"]),
                "personalization_gain": sequence_similarity(target_actions, same["actions"]) - sequence_similarity(target_actions, cross["actions"]),
            }
        )
    return rows


def _case_gain(task: dict[str, Any]) -> float:
    same = task.get("references", {}).get("same_user_top1") or []
    cross = task.get("references", {}).get("cross_user_top1") or []
    if not same or not cross:
        return -math.inf
    target = list(task.get("target_actions") or [])
    return sequence_similarity(target, same[0]["actions"]) - sequence_similarity(target, cross[0]["actions"])


def _short_action(action: str, max_length: int = 48) -> str:
    text = str(action).replace("coordinates=", "xy=").replace("content=", "")
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str, dpi: int) -> list[str]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png), str(pdf)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_summary(
    path: Path,
    tasks: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    retrieval: dict[str, Any],
    copy_report: dict[str, Any],
) -> None:
    gains = np.array([row["personalization_gain"] for row in episode_rows])
    same_intent = np.array([row["same_intent_similarity"] for row in episode_rows])
    same_sequence = np.array([row["same_sequence_similarity"] for row in episode_rows])
    cross_intent = np.array([row["cross_intent_similarity"] for row in episode_rows])
    cross_sequence = np.array([row["cross_sequence_similarity"] for row in episode_rows])
    lengths = np.array([len(task.get("target_actions") or []) for task in tasks])
    action_counts = Counter(str(step.get("action_type") or "unknown") for step in steps)
    text = f"""# Baseline Visualization Summary

## Introduction-ready findings

- Same-user Top-1 history reaches trajectory similarity
  `{retrieval['modes']['same_user_top1']['sequence_similarity']:.3f}`, compared
  with `{retrieval['modes']['cross_user_top1']['sequence_similarity']:.3f}` for
  cross-user Top-1 history.
- Same-user history is closer than cross-user history on
  `{(gains > 0).mean() * 100:.1f}%` of the `{len(tasks)}` available execution-test episodes.
- Mean per-episode personalization gain is `{gains.mean():.3f}`.
- Excluding exactly matching intents retains trajectory similarity
  `{retrieval['modes']['same_user_no_same_intent']['sequence_similarity']:.3f}`,
  showing that the signal is not explained only by repeated identical tasks.
- Intent-to-trajectory Pearson correlation is `{np.corrcoef(same_intent, same_sequence)[0, 1]:.2f}`
  for same-user retrieval and `{np.corrcoef(cross_intent, cross_sequence)[0, 1]:.2f}`
  for cross-user retrieval. Intent similarity alone does not determine the preferred path.

## Dataset and diagnostic findings

- The execution-test subset contains `{len(tasks)}` complete episodes and
  `{len(steps)}` parsed actions.
- Median trajectory length is `{np.median(lengths):.0f}` steps; mean length is
  `{np.mean(lengths):.1f}` steps.
- Clicks account for `{action_counts['click'] / len(steps):.1%}` of actions,
  scrolls for `{action_counts['scroll'] / len(steps):.1%}`, and waits for
  `{action_counts['wait'] / len(steps):.1%}`.
- The retrieval-copy diagnostic obtains action-type accuracy
  `{copy_report['variants']['official_icl']['action_type_accuracy']:.3f}` with
  same-user history versus
  `{copy_report['variants']['cross_user_icl']['action_type_accuracy']:.3f}` with
  cross-user history. It is a lower-bound diagnostic, not a trained agent result.

## Suggested Introduction Text

For similar mobile tasks, users often follow distinct action paths. On the
FingerTip-20K personalized execution split, the most similar same-user history
is closer to the target trajectory than a different-type cross-user history on
{(gains > 0).mean() * 100:.1f}% of episodes, yielding an average similarity
gain of {gains.mean():.3f}. This advantage remains after excluding exactly
matching historical intents. Meanwhile, intent similarity is only moderately
correlated with action-path similarity, suggesting that retrieving a similar
task is insufficient: user history should instead induce a personalized action
policy.

## Figure Cautions

- `fig06_reference_copy_tradeoff` must be described as a retrieval-only
  diagnostic. It does not represent online task success or a trained policy.
- Retrieval and trajectory similarities are motivation evidence, not proof that
  a model can exploit the history. Model inference and SFT results are still
  required.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
