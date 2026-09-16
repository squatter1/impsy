"""Prediction accuracy against MCTS iterations with an asymptotic fit y = a - b / x^c.
--bootstrap adds a resampled confidence band and a confidence interval for the asymptote a."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from pathlib import Path
from results import ITERATIONS, ACCURACY_VS_ITERATIONS, ACCURACY, FIGURES_DIR


def asymptotic_func(x, a, b, c):
    return a - b / np.power(x, c)


def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_residual / ss_total


def fit(x, y, min_r_squared=0.0):
    """Fit from a grid of initial guesses to avoid local minima, keeping the best fit with R² above the threshold."""
    best_params, best_pcov, best_residual = None, None, np.inf
    for a in range(6, 25):
        for b in range(10, 100, 10):
            for c in np.arange(0.1, 1.1, 0.1):
                try:
                    params, pcov = curve_fit(asymptotic_func, x, y, p0=[a, b, c], maxfev=10000)
                except RuntimeError:
                    continue
                y_pred = asymptotic_func(x, *params)
                residual = np.sum((y - y_pred) ** 2)
                if r_squared(y, y_pred) >= min_r_squared and residual < best_residual:
                    best_params, best_pcov, best_residual = params, pcov, residual
    return best_params, best_pcov


def bootstrap(x, y, params, x_smooth, n_samples, min_r_squared, rng):
    """Refit on resampled data, returning the curves and asymptotes of fits with R² above the threshold."""
    curves, asymptotes = [], []
    for _ in range(n_samples):
        indices = rng.choice(len(x), size=len(x), replace=True)
        try:
            resampled_params, _ = curve_fit(asymptotic_func, x[indices], y[indices], p0=params, maxfev=10000)
        except RuntimeError:
            continue
        if r_squared(y[indices], asymptotic_func(x[indices], *resampled_params)) >= min_r_squared:
            curves.append(asymptotic_func(x_smooth, *resampled_params))
            asymptotes.append(resampled_params[0])
    return np.array(curves), np.array(asymptotes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=list(ACCURACY_VS_ITERATIONS), default="improv")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N", help="Number of bootstrap resamples for confidence intervals")
    parser.add_argument("--min-r-squared", type=float, default=0.95, help="Minimum R² for a fit to be kept")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Output image, default is evaluation/figures/accuracy-vs-iterations-<corpus>.png")
    args = parser.parse_args()

    x = np.array(ITERATIONS, dtype=float)
    y = np.array(ACCURACY_VS_ITERATIONS[args.corpus])
    unguided = ACCURACY[(args.corpus, "total", "individual")][0]

    params, pcov = fit(x, y, args.min_r_squared)
    if params is None:
        parser.exit(1, f"No fit reached R² >= {args.min_r_squared}\n")
    a_fit, b_fit, c_fit = params
    a_err = np.sqrt(np.diag(pcov))[0]
    print(f"Fit: a = {a_fit:.4f} ± {a_err:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}, R² = {r_squared(y, asymptotic_func(x, *params)):.4f}")
    print(f"Asymptotic maximum accuracy: {a_fit:.2f}%")

    x_smooth = np.logspace(np.log10(min(x)), np.log10(max(x) * 2), 300)
    y_fit = asymptotic_func(x_smooth, *params)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(x, y, color="#25cdd8", s=150)
    plt.plot(x_smooth, y_fit, "r-", linewidth=4, alpha=0.85, label=f"Asymptotic Fit (Max ≈ {a_fit:.1f}%)")
    plt.axhline(y=unguided, color="green", linestyle="--", linewidth=2, label=f"Unguided MDRNN ({unguided:.2f}%)")

    if args.bootstrap:
        rng = np.random.default_rng(args.seed)
        curves, asymptotes = bootstrap(x, y, params, x_smooth, args.bootstrap, args.min_r_squared, rng)
        print(f"Kept {len(curves)} of {args.bootstrap} bootstrap fits with R² >= {args.min_r_squared}")
        if len(curves) >= 10:
            lower, upper = np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0)
            plt.fill_between(x_smooth, lower, upper, color="red", alpha=0.15, label="Bootstrap 90% band")
            print(f"Bootstrap 90% interval for the asymptote: {np.percentile(asymptotes, 5):.2f}% to {np.percentile(asymptotes, 95):.2f}%")
        else:
            print("Too few bootstrap fits for a confidence band")

    plt.ylim(bottom=0)
    plt.xlabel("Monte Carlo Tree Search Iterations", fontsize=16)
    plt.ylabel("Prediction Accuracy (%)", fontsize=16)
    plt.xscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend(fontsize=16, loc="lower right")
    plt.tight_layout()

    output = args.output or FIGURES_DIR / f"accuracy-vs-iterations-{args.corpus}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
