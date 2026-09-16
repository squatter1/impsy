"""Bar chart of prediction accuracy per heuristic, with reference lines for the unguided MDRNN and the combined heuristic."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from results import ACCURACY, HEURISTIC_LABELS, METRIC_LABELS, BAR_COLORS, FIGURES_DIR


def plot_column(corpus: str, metric: str, mode: str, output: Path):
    values = ACCURACY[(corpus, metric, mode)]
    unguided_value, combined_value = values[0], values[-1]
    bar_labels, bar_values = HEURISTIC_LABELS[1:-1], values[1:-1]
    bar_positions = np.arange(len(bar_labels))

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(bar_positions, bar_values, color=BAR_COLORS[corpus], width=0.6)

    # Reference lines across the whole plot
    unguided_line_color, combined_line_color = "#C44E52", "#55A868"
    ax.axhline(y=unguided_value, color=unguided_line_color, linestyle="-", linewidth=4)
    ax.axhline(y=combined_value, color=combined_line_color, linestyle="-", linewidth=4)

    ax.set_ylabel(METRIC_LABELS[metric], fontsize=16)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=16)
    ax.set_ylim(0, max(values) * 1.23)
    ax.legend(
        handles=[
            Line2D([0], [0], color=unguided_line_color, lw=4, label=f"Unguided MDRNN ({unguided_value:.2f}%)"),
            Line2D([0], [0], color=combined_line_color, lw=4, label=f"Combined Heuristic ({combined_value:.2f}%)"),
        ],
        loc="upper left",
    )

    # Value labels on top of each bar
    for bar, value in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1, f"{value:.2f}%", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["improv", "nottingham"], default="improv")
    parser.add_argument("--metric", choices=["total", "pitch", "time"], default="total")
    parser.add_argument("--mode", choices=["individual", "combined"], default="individual")
    parser.add_argument("--output", type=Path, help="Output image, default is evaluation/figures/column-<corpus>-<metric>-<mode>.png")
    args = parser.parse_args()
    if (args.corpus, args.metric, args.mode) not in ACCURACY:
        parser.error(f"No results for corpus={args.corpus} metric={args.metric} mode={args.mode}")
    output = args.output or FIGURES_DIR / f"column-{args.corpus}-{args.metric}-{args.mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_column(args.corpus, args.metric, args.mode, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
