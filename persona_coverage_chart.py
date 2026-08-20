"""Shared category-coverage chart rendering for persona pipelines."""

from pathlib import Path
from typing import Any


def render_category_coverage_chart(
    category_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = [str(row["category"]) for row in category_rows]
    coverage = [float(row["coverage_percentage"]) for row in category_rows]

    figure_width = max(16.0, len(categories) * 0.48)
    figure, coverage_axis = plt.subplots(figsize=(figure_width, 7.5))
    positions = list(range(len(categories)))
    coverage_axis.bar(
        positions,
        coverage,
        color="#1f4e79",
        width=0.72,
        label="Coverage (%)",
    )
    coverage_axis.set_ylabel("Coverage (%)", color="#1f4e79")
    coverage_axis.set_ylim(0, 105)
    coverage_axis.set_xticks(positions)
    coverage_axis.set_xticklabels(categories, rotation=48, ha="right", fontsize=8)
    coverage_axis.grid(axis="y", linestyle="--", alpha=0.25)

    coverage_axis.legend(loc="upper right")
    coverage_axis.set_title(title, fontweight="bold")
    coverage_axis.set_xlabel("Persona schema categories")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
