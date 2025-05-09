import click
import numpy as np
from pathlib import Path
import datetime
import mdrnn
import heuristics
from mcts_prediction_tree import MCTSPredictionTree

class MCTSEvaluator:
    def __init__(self, dimension=2, units=64, mixtures=5, layers=2):
        self.dimension = dimension
        self.units = units
        self.mixtures = mixtures
        self.layers = layers
        self.results = {}
        
    def load_model(self, model_file: Path):
        """Load a model from file."""
        if model_file.suffix == ".keras" or model_file.suffix == ".h5":
            click.secho(f"MDRNN Loading from .keras or .h5 file: {model_file}", fg="green")
            return mdrnn.KerasMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers)
        elif model_file.suffix == ".tflite":
            click.secho(f"MDRNN Loading from .tflite file: {model_file}", fg="green")
            return mdrnn.TfliteMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers)
        else:
            click.secho(f"MDRNN Loading dummy model: {model_file}", fg="yellow")
            return mdrnn.DummyMDRNN(model_file, self.dimension, self.units, self.mixtures, self.layers)
    
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
    
    def evaluate_model(self, model_file: Path, log_file: Path, init_memory_length: int = 40) -> float:
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
        correct_predictions = 0
        correct_predictions_mdrnn = 0
        
        # Initialize memory with first 'init_memory_length' items
        rnn_output_memory = sequence_data[:init_memory_length].tolist()
        
        # Feed the sequence to the model one by one and make predictions
        for i in range(total_predictions):
            # Print a progress message every 100 iterations
            if i % 100 == 0:
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
                simulation_depth=2,
                exploration_weight=0.1,
            )
            
            # Run search
            best_output = prediction_tree.search(
                memory=rnn_output_memory[:-1],  # Copy of memory up to this point
                heuristic_functions=[
                    heuristics.pitch_height_heuristic,
                    heuristics.pitch_range_heuristic,
                    heuristics.pitch_proximity_heuristic,
                    heuristics.key_and_modal_conformity_heuristic,
                    #heuristics.tempo_swing_time_heuristic,
                ],
                time_limit_ms=100
            )[0]  
            
            # Check if prediction matches actual next item
            # Duration match if the absolute difference in the multiple of the pitches is less than 0.2
            duration_match = max(best_output[0], next_item[0]) / max(min(best_output[0], next_item[0]),0.001) < 1.2
            # Roudn pitches to check exactly
            output_pitch = round(best_output[1]*127)
            next_item_pitch = round(next_item[1]*127)
            pitch_match = output_pitch == next_item_pitch  # Exact match for pitch
            
            #if duration_match and pitch_match: TODO readd
            #    correct_predictions += 1
            # TODO currently only have pitcxh heuristics, so only check pitch match
            if pitch_match:
                correct_predictions += 1

            # Get the pure mdrnn prediction
            mdrnn_output = neural_net.generate(item)

            # Check if prediction matches actual next item
            # Duration match if the absolute difference in the multiple of the pitches is less than 0.2
            duration_match_mdrnn = max(mdrnn_output[0], next_item[0]) / max(min(mdrnn_output[0], next_item[0]),0.001) < 1.2
            # Roudn pitches to check exactly
            output_pitch_mdrnn = round(mdrnn_output[1]*127)
            next_item_pitch_mdrnn = round(next_item[1]*127)
            pitch_match_mdrnn = output_pitch_mdrnn == next_item_pitch_mdrnn

            #TODO change
            if pitch_match_mdrnn:
                correct_predictions_mdrnn += 1
            
            # Update memory with actual next item for next iteration
            rnn_output_memory.append(next_item)
            # Remove the first item to keep memory length constant
            rnn_output_memory.pop(0)
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_mdrnn = correct_predictions_mdrnn / total_predictions if total_predictions > 0 else 0
        return accuracy, accuracy_mdrnn
    
    def run_evaluation(self, models_dir: Path, logs_dir: Path):
        """Run evaluation on all model and log file pairs."""
        # Find all .tflite model files
        model_files = sorted(models_dir.glob("*.tflite"))
        
        for model_file in model_files:
            # Split the model stem by '-' and take the first part as model number
            model_num = model_file.stem.split('-')[0]
            log_file = logs_dir / f"{model_num}-{self.dimension}d-mdrnn.log"
            
            if not log_file.exists():
                click.secho(f"Log file not found: {log_file}", fg="red")
                continue
            
            click.secho(f"Evaluating model {model_num}...", fg="blue")
            accuracy, accuracy_mdrnn = self.evaluate_model(model_file, log_file)
            self.results[model_num] = accuracy
            click.secho(f"Model {model_num} accuracy: {accuracy:.2%}", fg="green")
            click.secho(f"Model {model_num} accuracy (MDRNN): {accuracy_mdrnn:.2%}", fg="green")
        
        # Print overall results
        click.secho("\nEvaluation Results:", fg="blue", bold=True)
        for model_num, accuracy in sorted(self.results.items(), key=lambda x: int(x[0])):
            click.secho(f"Model {model_num}: {accuracy:.2%}", fg="green")
        
        # Calculate average accuracy
        if self.results:
            avg_accuracy = sum(self.results.values()) / len(self.results)
            click.secho(f"\nAverage accuracy across all models: {avg_accuracy:.2%}", fg="blue", bold=True)


def main(models_dir, logs_dir):
    """Evaluate MCTS prediction accuracy for multiple models against their respective log files."""
    models_path = Path(models_dir)
    logs_path = Path(logs_dir)
    
    evaluator = MCTSEvaluator()
    evaluator.run_evaluation(models_path, logs_path)
    
    click.secho("Evaluation complete!", fg="green", bold=True)


if __name__ == "__main__":
    main('eval_models', 'eval_logs')