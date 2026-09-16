"""Accuracy and timing results from the evaluation, used by the plotting scripts."""

from pathlib import Path

HEURISTIC_LABELS = [
    "unguided MDRNN",
    "key + mode",
    "tempo + swing",
    "pitch interval",
    "time multiple",
    "repetition",
    "combined heuristic",
]

# Percentage of next notes predicted correctly, one value per label above
# metric: "total" (pitch and duration), "pitch" or "time"
# mode: "individual" (each heuristic alone) or "combined" (all heuristics except the named one)
ACCURACY = {
    ("improv", "total", "individual"): [3.02, 2.58, 3.14, 3.90, 3.07, 3.49, 5.06],
    ("improv", "pitch", "individual"): [9.84, 11.44, 9.22, 16.12, 8.97, 11.60, 18.82],
    ("improv", "time", "individual"): [26.27, 17.50, 26.50, 19.10, 30.72, 26.19, 22.54],
    ("improv", "total", "combined"): [3.02, 4.40, 4.29, 3.99, 4.64, 5.00, 5.06],
    ("nottingham", "total", "individual"): [6.02, 7.46, 8.39, 6.77, 8.02, 11.24, 14.76],
    ("nottingham", "pitch", "individual"): [12.16, 20.21, 12.33, 20.74, 12.18, 18.83, 22.13],
    ("nottingham", "time", "individual"): [51.54, 37.62, 70.40, 34.21, 65.56, 51.85, 67.55],
    ("nottingham", "total", "combined"): [6.02, 13.23, 14.01, 13.97, 14.61, 13.06, 14.76],
}

METRIC_LABELS = {
    "total": "Prediction Accuracy (%)",
    "pitch": "Pitch Prediction Accuracy (%)",
    "time": "Time Prediction Accuracy (%)",
}

BAR_COLORS = {"improv": "#64B5CD", "nottingham": "#8172B2"}

# Accuracy and search time against the number of MCTS iterations per prediction
ITERATIONS = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]
ACCURACY_VS_ITERATIONS = {
    "improv": [2.92, 2.95, 3.13, 3.84, 3.97, 4.19, 4.61, 4.72, 4.92, 5.26, 5.34, 5.56, 5.34],
    "nottingham": [5.95, 6.14, 8.14, 9.94, 10.84, 12.18, 14.08, 14.55, 15.39, 16.14, 16.34, 16.94, 16.73],
}
SEARCH_TIME_MS = [1.18, 3.325, 3.695, 11.97, 24.485, 28.91, 48.185, 91.365, 158.79, 286.805, 516.38, 924.28, 1637.185]

# Where the plotting scripts save figures
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
