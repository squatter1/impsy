"""impsy.evaluation: measures how often MCTS-guided prediction and plain MDRNN sampling
predict the next note of a logged 2D (time, pitch) performance."""

import click
import numpy as np
from pathlib import Path
import datetime
import time
from typing import Callable, List, Optional, Tuple

from . import heuristics
from .mcts_prediction_tree import MCTSPredictionTree


MODEL_NAME_KEYS = ("dim", "layers", "units", "mixtures")


def model_params_from_filename(model_file: Path) -> dict:
    """Parses dimension, layers, units and mixtures from a model filename."""
    params = {}
    for part in Path(model_file).stem.split("-"):
        for key in MODEL_NAME_KEYS:
            if part.startswith(key) and part[len(key):].isdigit():
                params[key] = int(part[len(key):])
    missing = [key for key in MODEL_NAME_KEYS if key not in params]
    if missing:
        raise ValueError(f"Model filename {model_file} is missing {', '.join(missing)}")
    return params


def parse_log_file(log_file: Path) -> np.ndarray:
    """Parses a 2D IMPSY log into an array of (duration, pitch) rows."""
    sequence = []
    timestamps = []
    with open(log_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                timestamps.append(datetime.datetime.fromisoformat(parts[0]))
                sequence.append(float(parts[2]))
    if not sequence:
        return np.empty((0, 2))
    durations = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
    durations.append(durations[-1] if durations else 0.0)
    return np.array(list(zip(durations, sequence)))


def prediction_matches(prediction: np.ndarray, actual: np.ndarray, use_duration_match: bool, use_pitch_match: bool) -> bool:
    """A duration matches if the two values are within 10% of each other; a pitch matches on the exact MIDI note."""
    if use_duration_match:
        ratio = max(prediction[0], actual[0]) / max(min(prediction[0], actual[0]), 0.001)
        if ratio >= 1.1:
            return False
    if use_pitch_match:
        if round(prediction[1] * 127) != round(actual[1] * 127):
            return False
    return True


def pair_models_with_logs(models_dir: Path, logs_dir: Path, dimension: int) -> List[Tuple[Path, Path, str]]:
    """With one model, every log in logs_dir is evaluated against it. With several, each model pairs
    with the log sharing its leading number."""
    model_files = sorted(models_dir.glob("*.tflite"))
    if not model_files:
        return []
    if len(model_files) == 1:
        return [(model_files[0], log_file, f"log {log_file.stem.split('-')[0]}") for log_file in sorted(logs_dir.glob("*.log"))]
    pairs = []
    for model_file in model_files:
        model_num = model_file.stem.split("-")[0]
        log_file = logs_dir / f"{model_num}-{dimension}d-mdrnn.log"
        if log_file.exists():
            pairs.append((model_file, log_file, f"model {model_num}"))
        else:
            click.secho(f"Log file not found: {log_file}", fg="red")
    return pairs


class MCTSEvaluator:
    def __init__(self, dimension=2, units=64, mixtures=5, layers=2, pi_temp=1.5, sigma_temp=0.01):
        self.dimension = dimension
        self.units = units
        self.mixtures = mixtures
        self.layers = layers
        self.pi_temp = pi_temp
        self.sigma_temp = sigma_temp

    def load_model(self, model_file: Path):
        """Load an inference model from file."""
        from . import mdrnn  # imported here so the CLI starts without loading TensorFlow

        model_file = Path(model_file)
        args = (model_file, self.dimension, self.units, self.mixtures, self.layers)
        if model_file.suffix in (".keras", ".h5"):
            click.secho(f"MDRNN Loading from .keras or .h5 file: {model_file}", fg="green")
            return mdrnn.KerasMDRNN(*args, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)
        if model_file.suffix == ".tflite":
            click.secho(f"MDRNN Loading from .tflite file: {model_file}", fg="green")
            return mdrnn.TfliteMDRNN(*args, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)
        click.secho(f"MDRNN Loading dummy model: {model_file}", fg="yellow")
        return mdrnn.DummyMDRNN(*args, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)

    def evaluate_model(
        self,
        model_file: Path,
        log_file: Path,
        heuristic_functions: Optional[List[Tuple[Callable, Callable, float]]] = None,
        use_duration_match: bool = True,
        use_pitch_match: bool = True,
        init_memory_length: int = 45,
        simulation_depth: int = 2,
        greedy_weight: float = 0.4,
        exploration_weight: float = 0.1,
        time_limit_ms: float = 100,
        max_iterations: Optional[int] = None,
    ) -> dict:
        """Steps through a log one note at a time, predicting the next note with both the MCTS tree and the
        plain MDRNN, and counts how often each is correct. Returns the counts and total search time."""
        neural_net = self.load_model(model_file)
        sequence_data = parse_log_file(log_file)
        result = {"total": 0, "correct_mcts": 0, "correct_mdrnn": 0, "search_time": 0.0}

        if len(sequence_data) < init_memory_length + 2:
            click.secho(f"Not enough data in log file: {log_file}", fg="red")
            return result

        result["total"] = len(sequence_data) - init_memory_length - 1
        memory = sequence_data[:init_memory_length].tolist()

        for i in range(init_memory_length, init_memory_length + result["total"]):
            if (i - init_memory_length) % 250 == 0 and i > init_memory_length:
                click.secho(f"Evaluating: {i - init_memory_length}/{result['total']}", fg="blue")
            item = sequence_data[i].copy()
            next_item = sequence_data[i + 1]

            prediction_tree = MCTSPredictionTree(
                root_output=item,
                initial_lstm_states=neural_net.get_lstm_states(),
                predict_function=neural_net.generate_gmm,
                sample_function=neural_net.sample_gmm,
                initial_memory=memory[:-1],
                heuristic_functions=heuristic_functions or [],
                simulation_depth=simulation_depth,
                greedy_weight=greedy_weight,
                exploration_weight=exploration_weight,
                progressive_widening_k=2.5,
                progressive_widening_alpha=0.25,
                min_originality_distances=np.array([0.08, None]),
                expansion_samples=10,
                max_progressive_widening=5,
                snap_dp=[None, 2],
            )
            start = time.time()
            best_output = prediction_tree.search(memory=memory[:-1], time_limit_ms=time_limit_ms, max_iterations=max_iterations)[0]
            result["search_time"] += time.time() - start
            if prediction_matches(best_output, next_item, use_duration_match, use_pitch_match):
                result["correct_mcts"] += 1

            mdrnn_output = neural_net.generate(sequence_data[i].copy())
            if prediction_matches(mdrnn_output, next_item, use_duration_match, use_pitch_match):
                result["correct_mdrnn"] += 1

            memory.append(next_item)
            memory.pop(0)

        return result

    def run_evaluation(self, models_dir: Path, logs_dir: Path, **evaluate_kwargs) -> dict:
        """Evaluates every model/log pair found in the given directories and prints per-pair and overall accuracy."""
        pairs = pair_models_with_logs(Path(models_dir), Path(logs_dir), self.dimension)
        if not pairs:
            click.secho(f"No model/log pairs found in {models_dir} and {logs_dir}", fg="red")
            return {"total": 0, "correct_mcts": 0, "correct_mdrnn": 0, "search_time": 0.0}

        grand = {"total": 0, "correct_mcts": 0, "correct_mdrnn": 0, "search_time": 0.0}
        for model_file, log_file, label in pairs:
            click.secho(f"Evaluating {label}...", fg="cyan")
            result = self.evaluate_model(model_file, log_file, **evaluate_kwargs)
            for key in grand:
                grand[key] += result[key]
            self.print_accuracy(label.capitalize(), result)

        self.print_accuracy("Grand total", grand, fg="magenta")
        if grand["total"] > 0:
            click.secho(f"Average prediction time: {1000 * grand['search_time'] / grand['total']:.2f} ms", fg="magenta")
        click.secho("Evaluation complete!", fg="magenta")
        return grand

    @staticmethod
    def print_accuracy(label: str, result: dict, fg: str = "green") -> None:
        total = result["total"]
        for key, name in (("correct_mcts", "MCTS"), ("correct_mdrnn", "MDRNN")):
            accuracy = result[key] / total if total > 0 else 0.0
            click.secho(f"{label} accuracy ({name}): {result[key]}/{total} ({accuracy:.2%})", fg=fg)


@click.command(name="evaluate")
@click.option("-M", "--models-dir", default="evaluation/models", show_default=True, type=click.Path(file_okay=False),
              help="Directory of .tflite models. A single model is evaluated against every log; several models pair with logs by leading number.")
@click.option("-L", "--logs-dir", default="evaluation/logs", show_default=True, type=click.Path(file_okay=False),
              help="Directory of 2D IMPSY .log files.")
@click.option("-P", "--preset", default="improv", show_default=True, type=click.Choice(list(heuristics.HEURISTIC_PRESETS)),
              help="Heuristic weights tuned for a corpus.")
@click.option("-H", "--heuristic", "heuristic_names", multiple=True, type=click.Choice(heuristics.HEURISTIC_NAMES),
              help="Heuristic to enable; repeat for several. Default is all of them.")
@click.option("--match", default="both", show_default=True, type=click.Choice(["both", "pitch", "duration"]),
              help="Which parts of the next note must match for a prediction to count as correct.")
@click.option("--time-limit-ms", default=100.0, show_default=True, type=float, help="Search time per prediction.")
@click.option("--max-iterations", default=None, type=int, help="Cap on MCTS iterations per prediction (default: time limit only).")
def evaluate(models_dir, logs_dir, preset, heuristic_names, match, time_limit_ms, max_iterations):
    """Compare MCTS-guided prediction accuracy against the plain MDRNN on logged performances."""
    models_path = Path(models_dir)
    model_file = next(models_path.glob("*.tflite"), None)
    if model_file is None:
        click.secho(f"No .tflite models found in {models_path}", fg="red")
        raise click.Abort()
    params = model_params_from_filename(model_file)
    click.secho(f"Model params found: Dim: {params['dim']}, Layers: {params['layers']}, Units: {params['units']}, Mixtures: {params['mixtures']}")
    evaluator = MCTSEvaluator(dimension=params["dim"], units=params["units"], mixtures=params["mixtures"], layers=params["layers"])

    heuristic_functions = heuristics.build_heuristics(preset, heuristic_names)
    click.secho(f"Heuristics ({preset}): {', '.join(heuristic_names or heuristics.HEURISTIC_NAMES)}", fg="cyan")
    evaluator.run_evaluation(
        models_path,
        Path(logs_dir),
        heuristic_functions=heuristic_functions,
        use_duration_match=match in ("both", "duration"),
        use_pitch_match=match in ("both", "pitch"),
        time_limit_ms=time_limit_ms,
        max_iterations=max_iterations,
    )
