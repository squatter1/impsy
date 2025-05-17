import click
import numpy as np
from pathlib import Path
import datetime
import time
import mdrnn
import heuristics
from typing import Callable, Optional, Tuple
from mcts_prediction_tree import MCTSPredictionTree

class MCTSEvaluator:
    def __init__(self, dimension=2, units=64, mixtures=5, layers=2, pi_temp=1.5, sigma_temp=0.01):
        self.dimension = dimension
        self.units = units
        self.mixtures = mixtures
        self.layers = layers
        self.pi_temp = pi_temp
        self.sigma_temp = sigma_temp
        
    def load_model(self, model_file: Path):
        """Load a model from file."""
        if model_file.suffix == ".keras" or model_file.suffix == ".h5":
            click.secho(f"MDRNN Loading from .keras or .h5 file: {model_file}", fg="green")
            return mdrnn.KerasMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)
        elif model_file.suffix == ".tflite":
            click.secho(f"MDRNN Loading from .tflite file: {model_file}", fg="green")
            return mdrnn.TfliteMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)
        else:
            click.secho(f"MDRNN Loading dummy model: {model_file}", fg="yellow")
            return mdrnn.DummyMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers, pi_temp=self.pi_temp, sigma_temp=self.sigma_temp)
        
    def change_temperatures(self, pi_temp: float, sigma_temp: float):
        """Change the temperatures of the model."""
        self.pi_temp = pi_temp
        self.sigma_temp = sigma_temp
    
    def parse_log_file(self, log_file: Path) -> np.ndarray:
        """Parse log file and return as numpy array of (duration, pitch) pairs."""
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        sequence = []
        timestamps = []
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                timestamp = datetime.datetime.fromisoformat(parts[0])
                timestamps.append(timestamp)
                pitch = float(parts[2])
                sequence.append(pitch)
        
        # Calculate durations from timestamps
        durations = []
        for i in range(len(timestamps) - 1):
            duration = (timestamps[i+1] - timestamps[i]).total_seconds()
            durations.append(duration)
        
        # Add a final duration (could be average or last duration)
        if durations:
            durations.append(durations[-1])  # Use the last duration for the final element
        
        # Create the 2D array with (duration, pitch) pairs
        if len(durations) == len(sequence):
            return np.array(list(zip(durations, sequence)))
        else:
            # Handle case where lengths don't match (shouldn't happen if log format is consistent)
            click.secho(f"Warning: Mismatch in durations ({len(durations)}) and sequence ({len(sequence)}) lengths", fg="yellow")
            return np.array([])
    
    def evaluate_model(self, model_file: Path, log_file: Path, init_memory_length: int = 45, heuristics: Optional[Tuple[Callable, Callable]] = None
                           , use_duration_match: bool = True, use_pitch_match: bool = True, simulation_depth: int = 2, greedy_weight: float = 0.4, exploration_weight: float = 0.1
                           , time_limit_ms: float = 100,  max_iterations: int = None) -> float:
        """Evaluate a model against a log file and return accuracy."""
        # Load the model
        neural_net = self.load_model(model_file)
        
        # Parse the log file
        sequence_data = self.parse_log_file(log_file)
        
        if len(sequence_data) < 42:
            click.secho(f"Not enough data in log file: {log_file}", fg="red")
            return 0.0
        
        # Set up evaluation
        total_predictions = len(sequence_data) - init_memory_length - 1
        total_prediction_time = 0.0
        correct_predictions = 0
        correct_predictions_mdrnn = 0
        
        # Initialize memory with first 'init_memory_length' items
        rnn_output_memory = sequence_data[:init_memory_length].tolist()
        
        # Feed the sequence to the model one by one and make predictions
        for i in range(total_predictions):
            # Print a progress message every 250 iterations
            if i % 250 == 0 and i > 0:
                click.secho(f"Evaluating: {i}/{total_predictions}", fg="blue")
            # Set i to correct index given init_memory_length
            i += init_memory_length
            # Current item to base prediction on
            item = sequence_data[i]
            
            # Actual next item we want to predict
            next_item = sequence_data[i+1]
            
            # Create prediction tree
            prediction_tree = MCTSPredictionTree(
                root_output=item,
                initial_lstm_states=neural_net.get_lstm_states(),
                predict_function=neural_net.generate_gmm, 
                sample_function=neural_net.sample_gmm,
                initial_memory=rnn_output_memory[:-1],
                heuristic_functions=heuristics,
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
            
            # Run search
            start = time.time()
            best_output = prediction_tree.search(
                memory=rnn_output_memory[:-1],  # Copy of memory up to this point
                time_limit_ms=time_limit_ms,
                max_iterations=max_iterations,
            )[0] 
            total_prediction_time += time.time() - start
            
            # Check if prediction matches actual next item
            # Duration match if the absolute difference in the multiple of the pitches is less than 0.1
            duration_match = max(best_output[0], next_item[0]) / max(min(best_output[0], next_item[0]),0.001) < 1.1
            # Roudn pitches to check exactly
            output_pitch = round(best_output[1]*127)
            next_item_pitch = round(next_item[1]*127)
            pitch_match = output_pitch == next_item_pitch  # Exact match for pitch
            
            if (not use_duration_match or duration_match) and (not use_pitch_match or pitch_match):
                correct_predictions += 1

            # Get the pure mdrnn prediction
            mdrnn_output = neural_net.generate(item)

            # Check if prediction matches actual next item
            # Duration match if the absolute difference in the multiple of the pitches is less than 0.1
            duration_match_mdrnn = max(mdrnn_output[0], next_item[0]) / max(min(mdrnn_output[0], next_item[0]),0.001) < 1.1
            # Roudn pitches to check exactly
            output_pitch_mdrnn = round(mdrnn_output[1]*127)
            next_item_pitch_mdrnn = round(next_item[1]*127)
            pitch_match_mdrnn = output_pitch_mdrnn == next_item_pitch_mdrnn

            if (not use_duration_match or duration_match_mdrnn) and (not use_pitch_match or pitch_match_mdrnn):
                correct_predictions_mdrnn += 1
            
            # Update memory with actual next item for next iteration
            rnn_output_memory.append(next_item)
            # Remove the first item to keep memory length constant
            rnn_output_memory.pop(0)

        return total_predictions, correct_predictions, correct_predictions_mdrnn, total_prediction_time
    
    def run_evaluation(self, models_dir: Path, logs_dir: Path, heuristics: Optional[Tuple[Callable, Callable]] = None, use_duration_match: bool = True
                           , use_pitch_match: bool = True, init_memory_length: int = 45, simulation_depth: int = 2, greedy_weight: float = 0.4, exploration_weight: float = 0.1
                           , time_limit_ms: float = 100,  max_iterations: int = None):
        """Run evaluation on all model and log file pairs."""
        # Find all .tflite model files
        model_files = sorted(models_dir.glob("*.tflite"))
        log_files = sorted(logs_dir.glob("*.log"))
        
        grand_total_predictions = 0
        grant_total_prediction_time = 0.0
        grand_correct_predictions = 0
        grand_correct_predictions_mdrnn = 0
        if len(model_files) > 1:
            for model_file in model_files:
                # Split the model stem by '-' and take the first part as model number
                model_num = model_file.stem.split('-')[0]
                log_file = logs_dir / f"{model_num}-{self.dimension}d-mdrnn.log"
                
                if not log_file.exists():
                    click.secho(f"Log file not found: {log_file}", fg="red")
                    continue
                
                click.secho(f"Evaluating model {model_num}...", fg="cyan")
                total_predictions, correct_predictions, correct_predictions_mdrnn, total_prediction_time = self.evaluate_model(model_file, log_file, heuristics=heuristics, use_duration_match=use_duration_match, use_pitch_match=use_pitch_match, init_memory_length=init_memory_length, simulation_depth=simulation_depth, greedy_weight=greedy_weight, exploration_weight=exploration_weight, time_limit_ms=time_limit_ms, max_iterations=max_iterations)
                grand_total_predictions += total_predictions
                grant_total_prediction_time += total_prediction_time
                grand_correct_predictions += correct_predictions
                grand_correct_predictions_mdrnn += correct_predictions_mdrnn
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                accuracy_mdrnn = correct_predictions_mdrnn / total_predictions if total_predictions > 0 else 0.0
                click.secho(f"Model {model_num} accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})", fg="green")
                click.secho(f"Model {model_num} accuracy (MDRNN): {correct_predictions_mdrnn}/{total_predictions} ({accuracy_mdrnn:.2%})", fg="green")
        elif len(model_files) == 1:
            # We are using a single model for all log files, iterate through all log files
            for log_file in log_files:
                # We only have one model, so grab from the first model file
                model_file = model_files[0]
                log_num = log_file.stem.split('-')[0]
                
                click.secho(f"Evaluating log {log_num}...", fg="cyan")
                total_predictions, correct_predictions, correct_predictions_mdrnn, total_prediction_time = self.evaluate_model(model_file, log_file, heuristics=heuristics, use_duration_match=use_duration_match, use_pitch_match=use_pitch_match, init_memory_length=init_memory_length, simulation_depth=simulation_depth, greedy_weight=greedy_weight, exploration_weight=exploration_weight, time_limit_ms=time_limit_ms, max_iterations=max_iterations)
                grand_total_predictions += total_predictions
                grant_total_prediction_time += total_prediction_time
                grand_correct_predictions += correct_predictions
                grand_correct_predictions_mdrnn += correct_predictions_mdrnn
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                accuracy_mdrnn = correct_predictions_mdrnn / total_predictions if total_predictions > 0 else 0.0
                click.secho(f"Log {log_num} accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})", fg="green")
                click.secho(f"Log {log_num} accuracy (MDRNN): {correct_predictions_mdrnn}/{total_predictions} ({accuracy_mdrnn:.2%})", fg="green")
        else:
            click.secho(f"No model files found in {models_dir}", fg="red")
            return

        # Print grand totals and grand total accuracy
        grand_total_accuracy = grand_correct_predictions / grand_total_predictions if grand_total_predictions > 0 else 0.0
        click.secho(f"\nGrand total accuracy: {grand_correct_predictions}/{grand_total_predictions} ({grand_total_accuracy:.2%})", fg="magenta")
        grand_total_accuracy_mdrnn = grand_correct_predictions_mdrnn / grand_total_predictions if grand_total_predictions > 0 else 0.0
        click.secho(f"Grand total accuracy (MDRNN): {grand_correct_predictions_mdrnn}/{grand_total_predictions} ({grand_total_accuracy_mdrnn:.2%})", fg="magenta")
        click.secho(f"Average prediction time: {1000 * grant_total_prediction_time / grand_total_predictions:.2f} ms", fg="magenta")
        click.secho(f"Evaluation complete!", fg="magenta")


def main(models_dir, logs_dir):
    """Evaluate MCTS prediction accuracy for multiple models against their respective log files."""
    models_path = Path(models_dir)
    logs_path = Path(logs_dir)

    # Get the name of the first model file in the models directory
    model_file = next(models_path.glob("*.tflite"), None)
    if model_file:
        # Split by '-', the [2] is 'dim{dimension}', [3] is 'layers{layers}' the [4] is 'units{units}', the [5] is 'mixtures{mixtures}'
        model_name = model_file.stem.split('-')[2:6]
        dimension = int(model_name[0][3:])
        layers = int(model_name[1][6:])
        units = int(model_name[2][5:])
        mixtures = int(model_name[3][8:])
        print(f"Model params found: Dim: {dimension}, Layers: {layers}, Units: {units}, Mixtures: {mixtures}")
        evaluator = MCTSEvaluator(dimension=dimension, units=units, mixtures=mixtures, layers=layers)
    else:
        evaluator = MCTSEvaluator()

    # For improv model use 0.15, 0.05, 1.0, 0.1, 0.2
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.key_and_modal_memory(x, min_key_conformity=0.7),
                            lambda x, y, z: heuristics.key_and_modal_conformity_heuristic(x, y, z, min_mode_conformity=0.25, mode_divisor=6.0, mode_max=0.15),
                            0.15
                        ),
    ], use_pitch_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            heuristics.tempo_and_swing_memory, 
                            lambda x, y, z: heuristics.tempo_and_swing_heuristic(x, y, z, max_tempo_deviation=0.08),
                            0.05
                        ),
    ], use_duration_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.interval_markov_memory(x, order=1),
                            lambda x, y, z: heuristics.interval_markov_heuristic(x, y, z),
                            1.0
                        ),
    ], use_pitch_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.time_multiple_markov_memory(x, order=1),
                            lambda x, y, z: heuristics.time_multiple_markov_heuristic(x, y, z),
                            0.1
                        ),
    ], use_duration_match=False)

    # Switch the nottingham model to use the nottingham heuristics
    models_path = Path('eval_models_nottingham')
    logs_path = Path('eval_logs_nottingham')

    # Get the name of the first model file in the models directory
    model_file = next(models_path.glob("*.tflite"), None)
    if model_file:
        # Split by '-', the [2] is 'dim{dimension}', [3] is 'layers{layers}' the [4] is 'units{units}', the [5] is 'mixtures{mixtures}'
        model_name = model_file.stem.split('-')[2:6]
        dimension = int(model_name[0][3:])
        layers = int(model_name[1][6:])
        units = int(model_name[2][5:])
        mixtures = int(model_name[3][8:])
        print(f"Model params found: Dim: {dimension}, Layers: {layers}, Units: {units}, Mixtures: {mixtures}")
        evaluator = MCTSEvaluator(dimension=dimension, units=units, mixtures=mixtures, layers=layers)
    else:
        evaluator = MCTSEvaluator()

    # For nottingham model use 0.25, 0.3, 0.25, 0.25, 1.0
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.key_and_modal_memory(x, min_key_conformity=0.7),
                            lambda x, y, z: heuristics.key_and_modal_conformity_heuristic(x, y, z, min_mode_conformity=0.25, mode_divisor=6.0, mode_max=0.15),
                            0.25
                        ),
    ], use_pitch_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            heuristics.tempo_and_swing_memory, 
                            lambda x, y, z: heuristics.tempo_and_swing_heuristic(x, y, z, max_tempo_deviation=0.08),
                            0.3
                        ),
    ], use_duration_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.interval_markov_memory(x, order=1),
                            lambda x, y, z: heuristics.interval_markov_heuristic(x, y, z),
                            0.25
                        ),
    ], use_pitch_match=False)
    evaluator.run_evaluation(models_path, logs_path, heuristics=[
                        (
                            lambda x: heuristics.time_multiple_markov_memory(x, order=1),
                            lambda x, y, z: heuristics.time_multiple_markov_heuristic(x, y, z),
                            0.25
                        ),
    ], use_duration_match=False)
    

    click.secho("All evaluations complete!", fg="magenta", bold=True)


if __name__ == "__main__":
    main('eval_models', 'eval_logs')
    #main('eval_models_shortened', 'eval_logs_shortened')
    #main('eval_models_nottingham', 'eval_logs_nottingham')
    #main('eval_models_nottingham', 'eval_logs_shortened_nottingham')