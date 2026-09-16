"""Search time against MCTS iterations on a log-log plot with a linear fit through the origin."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from results import ITERATIONS, SEARCH_TIME_MS, FIGURES_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "search-time-vs-iterations.png")
    args = parser.parse_args()

    nodes, times = np.array(ITERATIONS), np.array(SEARCH_TIME_MS)
    # Line of best fit through the origin
    slope = np.sum(nodes * times) / np.sum(nodes ** 2)
    x_line = np.linspace(0, max(nodes), 100)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(nodes, times, color="blue", s=150)
    plt.plot(x_line, slope * x_line, "r-", linewidth=4, alpha=0.85, label=f"Linear fit (slope = {slope:.2f} ms/iteration)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Monte Carlo Tree Search Iterations", fontsize=16)
    plt.ylabel("Search Time (ms)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend(fontsize=16, loc="upper left", frameon=False)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Saved {args.output} (slope {slope:.2f} ms per iteration)")


if __name__ == "__main__":
    main()
