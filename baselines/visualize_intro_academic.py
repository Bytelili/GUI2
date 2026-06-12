from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLUE = "#3B73B9"
ORANGE = "#D9822B"
GREEN = "#4C956C"
PURPLE = "#8064A2"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate restrained academic Introduction figures.")
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "outputs/visualizations/introduction"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs/visualizations/introduction_academic"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = _read_csv(input_dir / "intro_fig01_one_size_fits_all_gap.csv")
    groups = _read_csv(input_dir / "intro_fig02_same_task_multiple_paths.csv")

    _style()
    _plot_similarity_density(comparisons, out_dir, args.dpi)
    _plot_similarity_ecdf(comparisons, out_dir, args.dpi)
    _plot_category_violins(comparisons, out_dir, args.dpi)
    _plot_history_raincloud(comparisons, out_dir, args.dpi)
    _plot_path_diversity_scatter(groups, out_dir, args.dpi)
    _plot_academic_composite(comparisons, groups, out_dir, args.dpi)
    print(f"wrote 6 academic figures to: {out_dir}")


def _plot_similarity_density(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    same, cross, gains = _comparison_arrays(rows)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    density = ax.hexbin(cross, same, gridsize=24, mincnt=1, cmap="Blues", linewidths=0.2)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1.2, label="Equal similarity")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Best cross-user trajectory similarity", ylabel="Same-user history similarity")
    ax.text(0.04, 0.96, f"Same-user wins: {(gains > 0).mean() * 100:.1f}%", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    colorbar = fig.colorbar(density, ax=ax, pad=0.02)
    colorbar.set_label("Number of comparisons")
    _finish(fig, out_dir / "academic_fig01_similarity_hexbin", dpi)


def _plot_similarity_ecdf(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    same, cross, gains = _comparison_arrays(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for values, color, label in [(same, BLUE, "Same-user history"), (cross, ORANGE, "Best cross-user path")]:
        x, y = _ecdf(values)
        axes[0].plot(x, y, color=color, linewidth=2, label=label)
    axes[0].set(xlabel="Trajectory similarity", ylabel="Empirical cumulative probability")
    axes[0].legend(frameon=False)

    x, y = _ecdf(gains)
    axes[1].plot(x, y, color=PURPLE, linewidth=2)
    axes[1].axvline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[1].axvline(0.1, color=GREEN, linestyle=":", linewidth=1.5)
    axes[1].set(xlabel="Same-user personalization gain", ylabel="Empirical cumulative probability")
    axes[1].text(0.03, 0.96, f"P(gain > 0.1) = {(gains > 0.1).mean():.3f}", transform=axes[1].transAxes, va="top")
    _finish(fig, out_dir / "academic_fig02_similarity_ecdf", dpi)


def _plot_category_violins(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    selected = _category_groups(rows)
    labels = [item[0] for item in selected]
    values = [item[1] for item in selected]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    parts = ax.violinplot(values, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor(BLUE)
        body.set_alpha(0.28)
    ax.boxplot(values, vert=False, widths=0.16, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": BLUE},
               medianprops={"color": ORANGE, "linewidth": 1.8},
               whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    rng = np.random.default_rng(7)
    for index, group in enumerate(values, start=1):
        y = index + rng.uniform(-0.08, 0.08, len(group))
        ax.scatter(group, y, s=9, color=BLUE, alpha=0.28, linewidths=0)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(1, len(labels) + 1), labels)
    ax.set(xlabel="Same-user personalization gain", ylabel="Task category")
    _finish(fig, out_dir / "academic_fig03_category_violin", dpi)


def _plot_history_raincloud(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    labels, groups = _history_groups(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    parts = ax.violinplot(groups, positions=np.arange(1, 4), showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(GREEN)
        body.set_edgecolor(GREEN)
        body.set_alpha(0.25)
    ax.boxplot(groups, positions=np.arange(1, 4), widths=0.16, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": GREEN},
               medianprops={"color": ORANGE, "linewidth": 1.8},
               whiskerprops={"color": GREEN}, capprops={"color": GREEN})
    rng = np.random.default_rng(11)
    for index, group in enumerate(groups, start=1):
        sample = rng.choice(group, size=min(120, len(group)), replace=False)
        ax.scatter(index + rng.uniform(0.10, 0.24, len(sample)), sample, s=9, color=GREEN, alpha=0.25, linewidths=0)
    means = np.array([np.mean(group) for group in groups])
    lower, upper = zip(*[_bootstrap_ci(group) for group in groups])
    ax.errorbar(np.arange(1, 4), means, yerr=[means - lower, np.array(upper) - means], color=PURPLE,
                marker="o", markersize=5, linewidth=1.8, capsize=3, label="Mean and 95% bootstrap CI")
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(1, 4), labels)
    ax.set(xlabel="Number of prior same-user demonstrations", ylabel="Same-user personalization gain")
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, out_dir / "academic_fig04_history_raincloud", dpi)


def _plot_path_diversity_scatter(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    episodes = np.array([float(row["num_episodes"]) for row in rows])
    paths = np.array([float(row["num_distinct_action_type_paths"]) for row in rows])
    ratios = np.array([float(row["distinct_path_ratio"]) for row in rows])
    users = np.array([float(row["num_users"]) for row in rows])
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    scatter = ax.scatter(episodes, paths, s=20 + users * 12, c=ratios, cmap="viridis", alpha=0.72, edgecolor="white", linewidth=0.4)
    max_value = max(episodes.max(), paths.max())
    ax.plot([0, max_value], [0, max_value], "--", color=GRAY, linewidth=1, label="One distinct path per episode")
    ax.set(xlabel="Episodes sharing the exact same intent", ylabel="Distinct action-type paths")
    ax.legend(frameon=False, loc="lower right")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Distinct-path ratio")
    _finish(fig, out_dir / "academic_fig05_path_diversity_scatter", dpi)


def _plot_academic_composite(rows: list[dict[str, str]], groups: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    same, cross, gains = _comparison_arrays(rows)
    labels, history = _history_groups(rows)
    episodes = np.array([float(row["num_episodes"]) for row in groups])
    paths = np.array([float(row["num_distinct_action_type_paths"]) for row in groups])
    categories = _category_groups(rows)[:6]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    axes[0, 0].hexbin(cross, same, gridsize=20, mincnt=1, cmap="Blues", linewidths=0.15)
    axes[0, 0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
    axes[0, 0].set(xlabel="Best cross-user similarity", ylabel="Same-user similarity", title="(a) Repeated-task similarity")

    for values, color, label in [(same, BLUE, "Same-user"), (cross, ORANGE, "Cross-user")]:
        x, y = _ecdf(values)
        axes[0, 1].plot(x, y, color=color, linewidth=2, label=label)
    axes[0, 1].set(xlabel="Trajectory similarity", ylabel="Empirical cumulative probability", title="(b) Similarity distributions")
    axes[0, 1].legend(frameon=False)

    parts = axes[1, 0].violinplot(history, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(GREEN)
        body.set_edgecolor(GREEN)
        body.set_alpha(0.3)
    axes[1, 0].axhline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(np.arange(1, 4), labels)
    axes[1, 0].set(xlabel="Prior same-user demonstrations", ylabel="Personalization gain", title="(c) Gain by history depth")

    values = [item[1] for item in categories]
    category_labels = [item[0] for item in categories]
    parts = axes[1, 1].violinplot(values, vert=False, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PURPLE)
        body.set_edgecolor(PURPLE)
        body.set_alpha(0.3)
    axes[1, 1].axvline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[1, 1].set_yticks(np.arange(1, len(category_labels) + 1), category_labels)
    axes[1, 1].set(xlabel="Personalization gain", title="(d) Gain across task categories")

    fig.suptitle("Empirical motivation for personalized GUI execution", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, out_dir / "academic_fig06_introduction_composite", dpi, tight=False)


def _comparison_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    same = np.array([float(row["same_user_similarity"]) for row in rows])
    cross = np.array([float(row["best_cross_user_similarity"]) for row in rows])
    return same, cross, same - cross


def _category_groups(rows: list[dict[str, str]]) -> list[tuple[str, np.ndarray]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["intent_class"]].append(float(row["personalization_gain"]))
    selected = [(name, np.array(values)) for name, values in grouped.items() if len(values) >= 8]
    return sorted(selected, key=lambda item: np.mean(item[1]))


def _history_groups(rows: list[dict[str, str]]) -> tuple[list[str], list[np.ndarray]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        depth = int(row["prior_same_intent_count"])
        label = "1" if depth == 1 else "2" if depth == 2 else "3+"
        grouped[label].append(float(row["personalization_gain"]))
    labels = ["1", "2", "3+"]
    return labels, [np.array(grouped[label]) for label in labels]


def _bootstrap_ci(values: np.ndarray, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(23)
    means = np.mean(rng.choice(values, size=(repeats, len(values)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _finish(fig: Any, path: Path, dpi: int, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    main()
