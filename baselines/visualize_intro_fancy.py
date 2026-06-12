from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVY = "#07111F"
PANEL = "#0C1B2E"
CYAN = "#46D9FF"
BLUE = "#3B82F6"
VIOLET = "#9B7BFF"
PINK = "#FF5FA2"
GOLD = "#FFC857"
MINT = "#63E6BE"
WHITE = "#F5F8FF"
MUTED = "#A8B5C7"
GRID = "#29415E"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fancy Introduction figures from real personalization statistics.")
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "outputs/visualizations/introduction"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "outputs/visualizations/introduction_fancy"))
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = _read_csv(input_dir / "intro_fig01_one_size_fits_all_gap.csv")
    groups = _read_csv(input_dir / "intro_fig02_same_task_multiple_paths.csv")
    categories = _read_csv(input_dir / "intro_fig03_cross_category_personalization.csv")
    history = _read_csv(input_dir / "intro_fig04_history_depth_value.csv")

    _style()
    _hero_teaser(comparisons, groups, history, out_dir, args.dpi)
    _personalization_landscape(comparisons, out_dir, args.dpi)
    _category_orbit(categories, out_dir, args.dpi)
    _history_ribbon(history, out_dir, args.dpi)
    print(f"wrote 4 fancy figures to: {out_dir}")


def _hero_teaser(rows: list[dict[str, str]], groups: list[dict[str, str]], history: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    gains = np.array([float(row["personalization_gain"]) for row in rows])
    win_rate = np.mean(gains > 0)
    meaningful = np.mean(gains > 0.1)
    multi_path = np.mean([float(row["num_distinct_action_type_paths"]) > 1 for row in groups])
    history_gain = [float(row["gain"]) for row in history]

    fig = plt.figure(figsize=(16, 7.2), facecolor=NAVY)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.82, 1.18], left=0.035, right=0.975, top=0.82, bottom=0.12, wspace=0.08)
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    for ax in axes:
        _panel(ax)

    ax = axes[0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    _glow_node(ax, (0.17, 0.5), 0.075, CYAN, "SAME\nINTENT", 10)
    user_points = [(0.82, 0.78), (0.82, 0.58), (0.82, 0.38), (0.82, 0.18)]
    colors = [CYAN, VIOLET, PINK, GOLD]
    bends = [0.26, 0.10, -0.10, -0.26]
    for point, color, bend in zip(user_points, colors, bends):
        _glow_curve(ax, (0.25, 0.5), (point[0] - 0.07, point[1]), color, bend)
        _glow_node(ax, point, 0.047, color, "", 8)
    ax.text(0.5, 0.94, "ONE REQUEST, MANY VALID PATHS", color=WHITE, ha="center", va="top", fontsize=14, weight="bold")
    ax.text(0.5, 0.06, f"{multi_path * 100:.0f}% of shared exact intents show multiple paths", color=MUTED, ha="center", fontsize=10)

    ax = axes[1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    for radius, alpha in [(0.34, 0.05), (0.27, 0.08), (0.205, 0.14)]:
        ax.add_patch(Circle((0.5, 0.53), radius, color=CYAN, alpha=alpha, lw=0))
    ax.add_patch(Circle((0.5, 0.53), 0.16, facecolor="#102C46", edgecolor=CYAN, lw=2.2))
    ax.text(0.5, 0.56, f"{win_rate * 100:.1f}%", color=WHITE, fontsize=28, weight="bold", ha="center", va="center")
    ax.text(0.5, 0.45, "same-user\nhistory wins", color=CYAN, fontsize=11, weight="bold", ha="center", va="center")
    ax.text(0.5, 0.94, "PERSONAL HISTORY IS SIGNAL", color=WHITE, ha="center", va="top", fontsize=14, weight="bold")
    ax.text(0.5, 0.10, f"{meaningful * 100:.1f}% retain >0.1 advantage", color=MUTED, ha="center", fontsize=10)

    ax = axes[2]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    x = np.array([0.18, 0.50, 0.82])
    y = np.array(history_gain)
    y_display = 0.22 + (y / max(y)) * 0.50
    ax.fill_between(x, 0.18, y_display, color=VIOLET, alpha=0.12)
    for width, alpha in [(9, 0.05), (5, 0.10), (2.5, 1.0)]:
        ax.plot(x, y_display, color=VIOLET, lw=width, alpha=alpha)
    for xi, yi, value, label in zip(x, y_display, y, ["1", "2", "3+"]):
        _glow_node(ax, (xi, yi), 0.045, VIOLET, "", 8)
        ax.text(xi, yi + 0.10, f"{value:.3f}", color=WHITE, ha="center", fontsize=12, weight="bold")
        ax.text(xi, 0.11, label, color=MUTED, ha="center", fontsize=11)
    ax.text(0.5, 0.94, "HISTORY UNLOCKS PERSONALIZATION", color=WHITE, ha="center", va="top", fontsize=14, weight="bold")
    ax.text(0.5, 0.04, "prior same-user demonstrations", color=MUTED, ha="center", fontsize=10)

    fig.text(0.5, 0.93, "WHY GUI AGENTS MUST REMEMBER THE USER", color=WHITE, ha="center", fontsize=25, weight="bold")
    fig.text(0.5, 0.875, "Real repeated-task evidence from longitudinal GUI executions", color=CYAN, ha="center", fontsize=12)
    fig.text(0.5, 0.035, "Curves are a schematic visual encoding; all displayed statistics are computed from real complete episodes.", color=MUTED, ha="center", fontsize=8)
    _save(fig, out_dir / "fancy_fig01_personalization_teaser", dpi)


def _personalization_landscape(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    cross = np.array([float(row["best_cross_user_similarity"]) for row in rows])
    same = np.array([float(row["same_user_similarity"]) for row in rows])
    gains = same - cross
    cmap = LinearSegmentedColormap.from_list("neon", [PINK, VIOLET, BLUE, CYAN, MINT])
    fig, ax = plt.subplots(figsize=(9.2, 8.2), facecolor=NAVY)
    _panel(ax)
    hb = ax.hexbin(cross, same, C=gains, reduce_C_function=np.mean, gridsize=22, mincnt=1, cmap=cmap, linewidths=0.25, edgecolors="#B9E9FF")
    for width, alpha in [(8, 0.04), (4, 0.08), (1.4, 0.9)]:
        ax.plot([0, 1], [0, 1], color=WHITE, lw=width, alpha=alpha, linestyle="--")
    ax.fill_between([0, 1], [0, 1], [1, 1], color=CYAN, alpha=0.035)
    ax.text(0.08, 0.89, "PERSONAL\nHISTORY WINS", color=CYAN, fontsize=15, weight="bold", transform=ax.transAxes)
    ax.text(0.69, 0.08, "CROSS-USER\nPATH WINS", color=PINK, fontsize=13, weight="bold", transform=ax.transAxes)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel="Best cross-user trajectory similarity", ylabel="Same-user history similarity")
    ax.set_title("THE PERSONALIZATION LANDSCAPE", color=WHITE, fontsize=20, weight="bold", pad=18)
    ax.grid(color=GRID, alpha=0.35, linewidth=0.7)
    cbar = fig.colorbar(hb, ax=ax, pad=0.025, shrink=0.78)
    cbar.set_label("Same-user personalization gain", color=WHITE)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(GRID)
    fig.text(0.5, 0.015, "Each hexagon aggregates real repeated-task comparisons; color encodes mean same-user advantage.", color=MUTED, ha="center", fontsize=9)
    _save(fig, out_dir / "fancy_fig02_personalization_landscape", dpi)


def _category_orbit(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    rows = sorted(rows, key=lambda row: float(row["mean_gain"]), reverse=True)
    labels = [row["intent_class"] for row in rows]
    gains = np.array([float(row["mean_gain"]) for row in rows])
    wins = np.array([float(row["same_user_win_rate"]) for row in rows])
    theta = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
    widths = np.full(len(rows), 2 * np.pi / len(rows) * 0.72)
    colors = plt.cm.cool(np.linspace(0.08, 0.88, len(rows)))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"}, facecolor=NAVY)
    ax.set_facecolor(NAVY)
    base = 0.28
    bars = ax.bar(theta, gains, width=widths, bottom=base, color=colors, alpha=0.9, edgecolor=WHITE, linewidth=0.6)
    ax.scatter(theta, base + gains + 0.055, s=90 + wins * 220, c=colors, edgecolors=WHITE, linewidths=0.7, zorder=5)
    for angle, label, gain, win in zip(theta, labels, gains, wins):
        rotation = np.degrees(angle)
        align = "left"
        if np.pi / 2 < angle < 3 * np.pi / 2:
            rotation += 180
            align = "right"
        ax.text(angle, base + gain + 0.145, label, color=WHITE, fontsize=9, ha=align, va="center", rotation=rotation, rotation_mode="anchor")
        ax.text(angle, base + gain / 2, f"{gain:.2f}", color=NAVY, fontsize=8, weight="bold", ha="center", va="center")
    ax.set_ylim(0, base + gains.max() + 0.28)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.text(0.5, 0.5, "PERSONALIZATION\nIS BROAD,\nNOT NICHE", transform=ax.transAxes, color=WHITE, fontsize=18, weight="bold", ha="center", va="center")
    fig.suptitle("A CROSS-CATEGORY ORBIT OF USER-SPECIFIC BEHAVIOR", color=WHITE, fontsize=19, weight="bold", y=0.97)
    fig.text(0.5, 0.04, "Bar length = mean gain   •   orbit marker size = same-user win rate", color=MUTED, ha="center", fontsize=10)
    _save(fig, out_dir / "fancy_fig03_cross_category_orbit", dpi)


def _history_ribbon(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    labels = [row["history_depth"].replace(" priors", "").replace(" prior", "") for row in rows]
    x = np.arange(len(rows))
    same = np.array([float(row["same_similarity"]) for row in rows])
    cross = np.array([float(row["cross_similarity"]) for row in rows])
    gain = same - cross
    fig, ax = plt.subplots(figsize=(11.8, 6.8), facecolor=NAVY)
    _panel(ax)
    ax.fill_between(x, cross, same, color=CYAN, alpha=0.13)
    for index in range(len(x) - 1):
        ax.fill_between(x[index:index + 2], cross[index:index + 2], same[index:index + 2], color=[BLUE, VIOLET][index], alpha=0.22)
    for values, color, name in [(same, CYAN, "Same-user history"), (cross, PINK, "Best cross-user path")]:
        for width, alpha in [(9, 0.04), (5, 0.08), (2.7, 1)]:
            ax.plot(x, values, color=color, lw=width, alpha=alpha, marker="o" if width == 2.7 else None, ms=9, label=name if width == 2.7 else None)
    for xi, low, high, delta in zip(x, cross, same, gain):
        ax.annotate("", xy=(xi, high), xytext=(xi, low), arrowprops={"arrowstyle": "<->", "color": GOLD, "lw": 1.5})
        ax.text(xi + 0.06, (low + high) / 2, f"+{delta:.3f}", color=GOLD, fontsize=11, weight="bold", va="center")
    ax.set_xticks(x, labels)
    ax.set(xlim=(-0.25, 2.3), ylim=(0.42, 0.81), xlabel="Prior same-user demonstrations", ylabel="Trajectory similarity")
    ax.set_title("MEMORY TURNS USER HISTORY INTO AN EXECUTION ADVANTAGE", color=WHITE, fontsize=19, weight="bold", pad=18)
    ax.grid(color=GRID, alpha=0.35)
    ax.legend(frameon=False, labelcolor=WHITE, loc="upper left")
    fig.text(0.5, 0.025, "The highlighted ribbon is the observed same-user advantage, not a causal model-performance claim.", color=MUTED, ha="center", fontsize=9)
    _save(fig, out_dir / "fancy_fig04_history_advantage_ribbon", dpi)


def _glow_curve(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str, bend: float) -> None:
    for width, alpha in [(10, 0.035), (6, 0.08), (2.2, 0.85)]:
        patch = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={bend}", arrowstyle="-|>", mutation_scale=12, color=color, lw=width, alpha=alpha)
        ax.add_patch(patch)


def _glow_node(ax: Any, point: tuple[float, float], radius: float, color: str, text: str, fontsize: int) -> None:
    for scale, alpha in [(2.0, 0.035), (1.55, 0.07), (1.2, 0.12)]:
        ax.add_patch(Circle(point, radius * scale, color=color, alpha=alpha, lw=0))
    ax.add_patch(Circle(point, radius, facecolor=PANEL, edgecolor=color, lw=2.0))
    if text:
        ax.text(*point, text, color=WHITE, fontsize=fontsize, weight="bold", ha="center", va="center")


def _panel(ax: Any) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if not hasattr(ax, "get_subplotspec"):
        return
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, boxstyle="round,pad=0.012,rounding_size=0.03", facecolor=PANEL, edgecolor=GRID, lw=1.0, zorder=-10, clip_on=False))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial"],
        "figure.facecolor": NAVY,
        "savefig.facecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "savefig.bbox": "tight",
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _save(fig: Any, path: Path, dpi: int) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor=NAVY)
    fig.savefig(path.with_suffix(".pdf"), facecolor=NAVY)
    plt.close(fig)


if __name__ == "__main__":
    main()
