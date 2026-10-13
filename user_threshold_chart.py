"""Chart rendering for the Facebook persona user-eligibility threshold analysis."""

from pathlib import Path
from typing import Any

BLUE = "#2a78d6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

METRIC_LABELS = {
    "post_count": "Số bài viết",
    "text_chars": "Tổng số ký tự",
    "history_days": "Độ dài lịch sử (ngày)",
}


def render_user_threshold_charts(
    curves: dict[str, list[dict[str, Any]]],
    scenarios: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Render retention curves (one per metric) plus a scenario comparison bar chart.

    Each retention curve answers "if the threshold is set to X, what % of the
    candidate population still passes" -- the exact question behind choosing
    min_posts/min_text_chars/min_history_days from data instead of guesswork.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    grid_spec = figure.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.5, wspace=0.3)

    for index, (metric, label) in enumerate(METRIC_LABELS.items()):
        axis = figure.add_subplot(grid_spec[0, index])
        axis.set_facecolor(SURFACE)
        points = curves[metric]
        # Grid thresholds are plotted at evenly spaced positions so every tick label is readable.
        positions = list(range(len(points)))
        ys = [p["retained_pct"] for p in points]
        axis.plot(positions, ys, color=BLUE, linewidth=2, marker="o", markersize=3)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{p['threshold']:,.0f}" for p in points], rotation=45, ha="right"
        )
        axis.set_ylim(0, 100)
        axis.set_title(label, color=INK, fontsize=11, loc="left")
        axis.set_ylabel("% user còn lại", color=SECONDARY_INK, fontsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(GRID)

    axis_bar = figure.add_subplot(grid_spec[1, :])
    axis_bar.set_facecolor(SURFACE)
    labels = [
        f"≥{row['min_posts']} bài\n≥{row['min_text_chars']:,} ký tự\n≥{row['min_history_days']:g} ngày"
        for row in scenarios
    ]
    values = [row["pct_kept"] for row in scenarios]
    positions = list(range(len(labels)))
    bars = axis_bar.bar(positions, values, color=BLUE, width=0.55)
    axis_bar.set_xticks(positions)
    axis_bar.set_xticklabels(labels, color=INK, fontsize=9)
    axis_bar.set_ylabel("% user còn lại (thỏa cả 3 ngưỡng)", color=SECONDARY_INK, fontsize=9)
    axis_bar.set_ylim(0, max(100, max(values) * 1.15))
    axis_bar.set_axisbelow(True)
    axis_bar.grid(axis="y", color=GRID, linewidth=0.8)
    axis_bar.tick_params(colors=MUTED, labelsize=8)
    for spine in axis_bar.spines.values():
        spine.set_color(GRID)
    for bar, row in zip(bars, scenarios):
        axis_bar.annotate(
            f"{row['pct_kept']:g}%\n({row['users_kept']:,})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 4),
            ha="center", va="bottom", fontsize=8, color=INK,
        )

    figure.suptitle(title, color=INK, fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
