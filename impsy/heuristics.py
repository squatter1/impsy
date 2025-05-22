import numpy as np
from typing import Callable, Tuple
from collections import defaultdict

#######################################
# FINAL HEURISTIC PARAMETER FUNCTIONS #
#######################################

def parameter_to_heuristic(memory: np.ndarray, branch: np.ndarray, parameter_function: Callable) -> float:
    return -abs(parameter_function(memory) - parameter_function(branch))

# Calculate the average pitch
def pitch_height(branch: np.ndarray, scaling_factor: float = 88) -> float:
    return 100 * np.mean(branch[:, 1]) / scaling_factor
def pitch_height_heuristic (memory: np.ndarray, branch: np.ndarray) -> float:
    return parameter_to_heuristic(memory, branch, pitch_height)

# Calculate the standard deviation of the pitch
def pitch_range(branch: np.ndarray, scaling_factor: float = 22) -> float:
    return 100 * np.std(branch[:, 1]) / scaling_factor
def pitch_range_heuristic (memory: np.ndarray, branch: np.ndarray) -> float:
    return parameter_to_heuristic(memory, branch, pitch_range)

# Average distance between the pitch of each note and its predecessor
def pitch_proximity(branch: np.ndarray, scaling_factor: float = 22) -> float:
    return 100 * np.mean(np.abs(np.diff(branch[:, 1]))) / scaling_factor
def pitch_proximity_heuristic (memory: np.ndarray, branch: np.ndarray) -> float:
    return parameter_to_heuristic(memory, branch, pitch_proximity)

###########################
# KEY AND MODAL HEURISTIC #
###########################

class Scale:
    def __init__(self, notes: np.ndarray, name: str = None):
        self.notes = notes
        self.name = name
        self.n = len(notes)

    def root_conformity(self, branch: np.ndarray, root: int) -> float:
        # Calculate the branch notes in the scale
        branch = (branch + root) % 12
        # Calculate the portion of notes in the branch that are also in the scale as k
        in_scale = np.isin(branch, self.notes)
        k = np.sum(in_scale) / len(branch)
        # Adjust conformity value based on the number of notes in the scale conformity = (12k-n)/(12-n)
        return (12 * k - self.n) / (12 - self.n)

    def conformity(self, branch: np.ndarray) -> float:
        conformity_values = []
        for root in range(12):
            conformity_values.append(self.root_conformity(branch, root))
            
        # Return the maximum root note and its conformity value
        max_conformity = np.max(conformity_values)
        max_root = np.argmax(conformity_values)
        return max_conformity, max_root
    
    def mode_conformity(self, branch: np.ndarray, root: int, mode: int) -> float:
        # Calculate the branch notes in the scale
        branch = (branch + root) % 12
        in_scale = np.isin(branch, self.notes)
        if in_scale.sum() == 0:
            # No notes in the scale, return 0
            return 0
        root_triad = None
        # Check for major triad
        if np.isin(np.array([0, 4, 7]) + mode, self.notes).all():
            root_triad = np.array([0, 4, 7]) + mode
        # Check for minor triad
        elif np.isin(np.array([0, 3, 7]) + mode, self.notes).all():
            root_triad = np.array([0, 3, 7]) + mode
        # Check for diminished triad
        elif np.isin(np.array([0, 3, 6]) + mode, self.notes).all():
            root_triad = np.array([0, 3, 6]) + mode
        # Check for augmented triad
        elif np.isin(np.array([0, 4, 8]) + mode, self.notes).all():
            root_triad = np.array([0, 4, 8]) + mode
        # If no triad is found, return 0
            return -1
        in_root_triad = np.isin(branch, root_triad)
        # Calculate the portion of notes in the scale that are also in the root triad as k_t
        k_t = np.sum(in_root_triad) / np.sum(in_scale)
        # Adjust conformity value based on the number of notes in the scale = (n*k_t-3)/(n-3)
        return (self.n * k_t - 3) / (self.n - 3)
    
    def mode_conformity_all(self, branch: np.ndarray, root: int) -> float:
        mode_conformity_values = []
        for mode in range(12):
            # Check this root note mode is in the scale
            if np.isin(mode, self.notes):
                mode_conformity_values.append(self.mode_conformity(branch, root, mode))
            else:
                mode_conformity_values.append(-1)
            
        # Return the maximum root note and its conformity value
        max_mode_conformity = np.max(mode_conformity_values)
        max_mode = np.argmax(mode_conformity_values)
        return max_mode_conformity, max_mode
    
    def __str__(self):
        return f"Scale: {self.notes}"


# How well do the keys match?
def key_conformity(branch: np.ndarray, min_key_conformity: float = 0.75) -> (float, Scale, int):
    # Calculate conformity to the major scale
    major_scale = Scale(np.array([0, 2, 4, 5, 7, 9, 11]), name="Major")
    major_conformity, major_root = major_scale.conformity(branch)
    # Calculate conformity to the pentatonic scale
    pentatonic_scale = Scale(np.array([0, 2, 4, 7, 9]), name="Pentatonic")
    pentatonic_conformity, pentatonic_root = pentatonic_scale.conformity(branch)
    # Calculate conformity to the blues scale
    blues_scale = Scale(np.array([0, 3, 5, 6, 7, 10]), name="Blues")
    blues_conformity, blues_root = blues_scale.conformity(branch)
    # Calculate conformity to the harmonic minor scale
    harmonic_minor_scale = Scale(np.array([0, 2, 3, 5, 7, 8, 11]), name="Harmonic Minor")
    harmonic_minor_conformity, harmonic_minor_root = harmonic_minor_scale.conformity(branch)

    # If any of these conformity values are above the min_key_conformity, return the max conformity value
    max_conformity = max(major_conformity, pentatonic_conformity, blues_conformity, harmonic_minor_conformity)
    argmax_conformity = np.argmax([major_conformity, pentatonic_conformity, blues_conformity, harmonic_minor_conformity])
    max_scale = [major_scale, pentatonic_scale, blues_scale, harmonic_minor_scale][argmax_conformity]
    max_root = [major_root, pentatonic_root, blues_root, harmonic_minor_root][argmax_conformity]

    if max_conformity > min_key_conformity:
        return max_conformity, max_scale, max_root
    return 0, None, None

def midi_to_note(midi_number):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_number // 12 - 1
    note = notes[midi_number % 12]
    return note, str(octave)

def key_and_modal_memory(memory: np.ndarray, min_key_conformity: float = 0.7) -> (float, Scale, int):
    memory = np.round(memory[:, 1]*127).astype(int) # *127 converts to 12 note scale
    # Calculate key conformity for memory
    memory_conformity, memory_scale, memory_root = key_conformity(memory, min_key_conformity)
    if memory_scale is not None:
        # Calculate modal conformity
        memory_mode_conformity, memory_mode = memory_scale.mode_conformity_all(memory, memory_root)
        return memory_conformity, memory_scale, memory_root, memory_mode_conformity, memory_mode, min_key_conformity
    return memory_conformity, memory_scale, memory_root, -1, None, min_key_conformity

def key_and_modal_conformity_heuristic(memory_tuple: np.ndarray, branch: np.ndarray, multiplier: float = 1.0, min_mode_conformity: float = 0.25, mode_divisor: float = 6.0, mode_max: float = 0.15) -> float:
    memory_conformity, memory_scale, memory_root, memory_mode_conformity, memory_mode, min_key_conformity = memory_tuple
    if memory_scale is None:
        # Chromatic scale, don't use key conformity as a heuristic
        return 0
    
    branch = np.round(branch[:, 1]*127).astype(int) # *127 converts to 12 note scale

    # Calculate conformity of the branch to the memory scale
    branch_conformity = memory_scale.root_conformity(branch, memory_root)

    # If no memory mode, only use key conformity
    if memory_mode_conformity < min_mode_conformity:
        # Not enough memory to establish a mode
        return (abs(branch_conformity - memory_conformity) / 2 + mode_max / 2) * 3.3 * multiplier

    # If branch_conformity < min_key_conformity, no point in calculating mode conformity, assume it is bad (as it will be calculated for other branches)
    if branch_conformity < min_key_conformity and abs(branch_conformity - memory_conformity) > 1 - min_key_conformity:
        # Not enough memory to establish a key
        return (abs(branch_conformity - memory_conformity) / 2 + mode_max) * 3.3 * multiplier

    # Calculate conformity of the branch to the memory mode
    branch_mode_conformity = memory_scale.mode_conformity(branch, memory_root, memory_mode)

    # Return the difference between the branch conformity and the memory conformity for key and mode
    return (abs(branch_conformity - memory_conformity) / 2 + min(abs(branch_mode_conformity - memory_mode_conformity) / mode_divisor, mode_max)) * 3.3 * multiplier

#############################
# TEMPO AND SWING HEURISTIC #
#############################

def tempo_and_swing_deviation(durations: np.ndarray, tempo: int, swing_name: str, swing_ratio: str) -> float:
    beat_duration = 60 / tempo  # Duration of one quarter note in seconds
    score = 0
    
    # Expected durations in this tempo/swing combination
    # Common note durations: whole, half, quarter, 8th, 16th notes and their dotted variants
    expected_durations = []
    
    # Regular note durations
    expected_durations.append((beat_duration * 4, 0))  # Whole note
    expected_durations.append((beat_duration * 2, 0))  # Half note
    expected_durations.append((beat_duration, 0))      # Quarter note
    
    # Eighth notes (no swing)
    if swing_name == "none":
        expected_durations.append((beat_duration / 2, 0))  # Eighth note
        expected_durations.append((beat_duration / 4, 0))  # Sixteenth note
    
    # Add dotted variants with slight penalty
    dotted_durations = [(d[0] * 1.5, 0.05) for d in expected_durations]
    expected_durations.extend(dotted_durations)

    # Add swing durations
    if swing_name != "none":
        # First eighth note in a pair (longer in swing)
        expected_durations.append((beat_duration * swing_ratio / (1 + swing_ratio), 0))
        # Second eighth note in a pair (shorter in swing)
        expected_durations.append((beat_duration / (1 + swing_ratio), 0))

    closest_durations = []
    for duration in durations:
        # Find closest expected duration
        closest_duration = min(expected_durations, key=lambda x: x[1] + (abs(x[0] - duration) / x[0]))
        closest_durations.append(closest_duration)
        # Calculate score based on how close the duration is to the expected duration
        score += (closest_duration[1] + abs(closest_duration[0] - duration) / closest_duration[0])

    # Disincentivise swing with long swing notes next to each other (as this usually isn't actually swing)
    if swing_name != "none":
        for i in range(len(closest_durations) - 1):
            if closest_durations[i][0] == closest_durations[i + 1][0] and closest_durations[i][0] == expected_durations[6][0]:
                score += 0.2

    score /= len(durations)  # Normalize by number of durations for average multiple from note in tempo

    return score

def estimate_tempo_and_swing(durations: np.ndarray, 
                             tempo_range: Tuple[int, int] = (70, 138), 
                             tempo_step: int = 2) -> Tuple[float, str, float]:
    """
    Estimate the tempo and swing ratio from a sequence of durations.
    Returns the most likely tempo (bpm), swing type, and confidence score.
    """
    # Convert durations to seconds (assuming they're in seconds already)
    durations = durations.flatten()
    
    best_score = float('inf')
    best_tempo = 0
    best_swing = "none"
    best_swing_duration = 0
    
    # Test different tempos and swing values
    tempos = range(tempo_range[0], tempo_range[1]+1, tempo_step)
    # Swing options: none, triplet (2:1), golden ratio (1.618:1) (unused)
    swing_options = {
        "none": 1.0,
        "triplet": 2.0
    }
    
    for tempo in tempos: 
        for swing_name, swing_ratio in swing_options.items():
            # Calculate the score for this tempo and swing
            score = tempo_and_swing_deviation(durations, tempo, swing_name, swing_ratio)
            # If the score is better than the best score, update the best values
            if score < best_score:
                best_score = score
                best_tempo = tempo
                best_swing = swing_name
                best_swing_duration = swing_ratio
    
    return best_tempo, best_swing, best_swing_duration, best_score

def tempo_and_swing_memory(memory: np.ndarray) -> Tuple:
    # Extract durations from memory and branch
    memory_durations = memory[:, 0]

    # Estimate tempo and swing from memory
    memory_tempo, memory_swing_name, memory_swing_duration, average_deviation = estimate_tempo_and_swing(memory_durations)
    return (memory_tempo, memory_swing_name, memory_swing_duration, average_deviation)

def tempo_and_swing_heuristic(memory_tuple: np.ndarray, branch: np.ndarray, multiplier: float = 1.0, 
                              max_tempo_deviation: float = 0.08) -> float:
    """
    Calculate heuristic based on tempo and swing deviation from memory.
    """
    memory_tempo, memory_swing_name, memory_swing_duration, average_deviation = memory_tuple

    if average_deviation > max_tempo_deviation:
        # If the average deviation is too high, return 0
        return 0
    tempo_conformity = 1 - (average_deviation / max_tempo_deviation)
    
    # Calculate the conformity of the branch to the memory tempo
    branch_durations = branch[:, 0]
    branch_average_deviation = tempo_and_swing_deviation(branch_durations, memory_tempo, memory_swing_name, memory_swing_duration)
    if branch_average_deviation < average_deviation:
        # Better than memory, return 0
        return 0

    # In this case, we assume a tempo and swing, and just want to return the deviation from that established tempo and swing.
    return tempo_conformity * (branch_average_deviation - average_deviation) * 20 * multiplier

#############################
# INTERVAL MARKOV HEURISTIC #
#############################

def interval_markov_memory(memory: np.ndarray, order: int = 2) -> Tuple:
    """
    Generate an n-order Markov chain model of intervals between notes from memory.
    
    Args:
        memory: np.ndarray of shape (n, 2) where memory[:, 1] contains pitch values
        order: The order of the Markov chain (default: 2)
        
    Returns:
        A tuple containing:
        - 'model': The Markov model as a nested dictionary
        - 'order': The order of the Markov model
        - 'total_transitions': Total number of transitions recorded
        - 'smoothing': Laplace smoothing parameter
        - 'valid': Boolean indicating if the model is valid (i.e., has enough data)
    """
    # Extract pitch values as integers
    pitches = np.round(memory[:, 1] * 127).astype(int)

    # Calculate intervals between consecutive notes
    intervals = np.diff(pitches)

    # Initialize the Markov model
    model = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    
    # Build the Markov model
    if len(intervals) < order + 1:
        # Not enough data to build the model of this order
        return dict(model), order, total_transitions, False, pitches[-order:]
    
    # Convert intervals to tuple states for easier dictionary handling
    for i in range(len(intervals) - order):
        # The state is the sequence of intervals before the transition
        state = tuple(intervals[i:i+order])
        # The next interval is the transition
        next_interval = intervals[i+order]
        
        # Update the model
        model[state][next_interval] += 1
        total_transitions += 1
    
    # Convert defaultdict to regular dict for better serialization
    model_dict = {state: dict(transitions) for state, transitions in model.items()}
    
    return model_dict, order, total_transitions, True, pitches[-order:]

def interval_markov_heuristic(memory_model: Tuple, branch: np.ndarray, multiplier: float = 1.0, smoothing: float = 0.1) -> float:
    """
    Calculate how well the branch conforms to the interval Markov model from memory.
    
    Args:
        memory_model: The Markov model dictionary returned by interval_markov_memory
        branch: np.ndarray of shape (n, 2) where branch[:, 1] contains pitch values
        
    Returns:
        A float value between 0 and 1, where higher values indicate greater deviation
        from the expected interval patterns (i.e., worse conformity)
    """
    model, order, total_transitions, valid, last_pitches = memory_model
    # If the model is not valid, return 0 (no penalty)
    if not valid:
        return 0.0
    
    # Extract pitch values and convert to integers if necessary
    pitches = np.concatenate((last_pitches, np.round(branch[:, 1] * 127).astype(int)))

    # Calculate intervals between consecutive notes
    intervals = np.diff(pitches)

    # Calculate probability for each transition in the branch
    probabilities = []
    
    for i in range(len(intervals) - order):
        # The state is the sequence of intervals before the transition
        state = tuple(intervals[i:i+order])
        # The next interval is the transition we're evaluating
        next_interval = intervals[i+order]
        
        # Get the transition probabilities for this state
        if state in model:
            state_transitions = model[state]
            # Count total transitions from this state
            total_state_transitions = sum(state_transitions.values())
            
            # Calculate probability with Laplace smoothing
            if next_interval in state_transitions:
                prob = (state_transitions[next_interval] + smoothing) / (total_state_transitions + smoothing * len(state_transitions))
            else:
                # If transition never observed, use smoothing
                prob = smoothing / (total_state_transitions + smoothing * len(state_transitions))
        else:
            # If state never observed, use a low probability
            # This could be adjusted based on the diversity of the training data
            prob = smoothing / (total_transitions / len(model) + smoothing)
        probabilities.append(prob)
    
    # Average probability across all transitions
    avg_probability = np.mean(probabilities)

    # Return 1 - avg_probability as the heuristic (higher value = worse conformity)
    return (1.0 - avg_probability) * 2 * multiplier

##################################
# TIME MULTIPLE MARKOV HEURISTIC #
##################################

def time_multiple_markov_memory(memory: np.ndarray, order: int = 2) -> Tuple:
    """
    Generate an n-order Markov chain model of time multiples between notes from memory.
    
    Args:
        memory: np.ndarray of shape (n, 2) where memory[:, 0] contains time intervals
        order: The order of the Markov chain (default: 2)
        
    Returns:
        A tuple containing:
        - 'model': The Markov model as a nested dictionary
        - 'order': The order of the Markov model
        - 'total_transitions': Total number of transitions recorded
        - 'smoothing': Laplace smoothing parameter
        - 'valid': Boolean indicating if the model is valid (i.e., has enough data)
        - 'last_intervals': The last 'order' intervals from memory for concatenation
    """
    # Extract time intervals directly (memory[:, 0] already contains intervals)
    time_intervals = memory[:, 0]
    last_intervals = time_intervals[-order:]
    
    # Calculate multiples between consecutive time intervals
    multiples = []
    valid_multiples = [0.125, 0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    common_multiples = [0.25, 0.5, 1.0, 2.0, 4.0]  # Common multiples for music
    
    for i in range(len(time_intervals) - 1): 
        if time_intervals[i] == 0 or time_intervals[i+1] == 0:
            # Avoid division by zero, set to 1.0 (no change in rhythm)
            multiples.append(1.0)
            continue
        multiple = time_intervals[i+1] / time_intervals[i]
        # Round to the nearest valid multiple
        rounded_multiple = min(valid_multiples, key=lambda x: abs(1 - (x / multiple)) if multiple in common_multiples else abs(1 - (x / multiple)) * 1.25) # sight disincentive for multiples that are not common
        multiples.append(rounded_multiple)
    
    # Initialize the Markov model
    model = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    
    # Build the Markov model
    if len(multiples) < order + 1:
        return dict(model), order, total_transitions, False, last_intervals
    
    # Convert multiples to tuple states for easier dictionary handling
    for i in range(len(multiples) - order):
        # The state is the sequence of multiples before the transition
        state = tuple(multiples[i:i+order])
        # The next multiple is the transition
        next_multiple = multiples[i+order]
        
        # Update the model
        model[state][next_multiple] += 1
        total_transitions += 1
    
    # Convert defaultdict to regular dict for better serialization
    model_dict = {state: dict(transitions) for state, transitions in model.items()}
    
    return model_dict, order, total_transitions, True, last_intervals

def time_multiple_markov_heuristic(memory_model: Tuple, branch: np.ndarray, multiplier: float = 1.0, smoothing: float = 0.1) -> float:
    """
    Calculate how well the branch conforms to the time multiple Markov model from memory.
    
    Args:
        memory_model: The Markov model tuple returned by time_multiple_markov_memory
        branch: np.ndarray of shape (n, 2) where branch[:, 0] contains time intervals
        
    Returns:
        A float value between 0 and 1, where higher values indicate greater deviation
        from the expected rhythmic patterns (i.e., worse conformity)
    """
    model, order, total_transitions, valid, last_intervals = memory_model
    # If the model is not valid, return 0 (no penalty)
    if not valid:
        return 0.0
    
    # Extract time intervals with padding at start
    time_intervals = np.concatenate((last_intervals, branch[:, 0]))
    
    # Calculate multiples between consecutive time intervals
    valid_multiples = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    common_multiples = [0.25, 0.5, 1.0, 2.0, 4.0]  # Common multiples for music
    multiples = []
    
    for i in range(len(time_intervals) - 1):
        if time_intervals[i] == 0 or time_intervals[i+1] == 0:
            # Avoid division by zero, set to 1.0 (no change in rhythm)
            multiples.append(1.0)
            continue
        multiple = time_intervals[i+1] / time_intervals[i]
        # Round to the nearest valid multiple
        rounded_multiple = min(valid_multiples, key=lambda x: abs(1 - (x / multiple)) if multiple in common_multiples else abs(1 - (x / multiple)) * 1.25) # sight disincentive for multiples that are not common
        multiples.append(rounded_multiple)
    
    # Calculate probability for each transition in the branch
    probabilities = []
    
    for i in range(len(multiples) - order):
        # The state is the sequence of multiples before the transition
        state = tuple(multiples[i:i+order])
        # The next multiple is the transition we're evaluating
        next_multiple = multiples[i+order]

        # Get the transition probabilities for this state
        if state in model:
            state_transitions = model[state]
            # Count total transitions from this state
            total_state_transitions = sum(state_transitions.values())
            
            # Calculate probability with Laplace smoothing
            if next_multiple in state_transitions:
                prob = (state_transitions[next_multiple] + smoothing) / (total_state_transitions + smoothing * len(state_transitions))
            else:
                # If transition never observed, use smoothing
                prob = smoothing / (total_state_transitions + smoothing * len(state_transitions))
        else:
            # If state never observed, use a low probability
            if len(model) > 0:
                prob = smoothing / (total_transitions / len(model) + smoothing)
            else:
                prob = 0.5  # Default probability if model is empty
        probabilities.append(prob)
    
    # Average probability across all transitions
    avg_probability = np.mean(probabilities)

    # Return 1 - avg_probability as the heuristic (higher value = worse conformity)
    return (1.0 - avg_probability) * 2 * multiplier

########################
# REPETITION HEURISTIC #
########################

def repetition_markov_memory(memory: np.ndarray, order: int = 2) -> Tuple:
    """
    Generate an n-order Markov chain model that combines both pitch intervals
    and time multiples between notes from memory.
    
    Args:
        memory: np.ndarray of shape (n, 2) where memory[:, 0] contains time intervals
            and memory[:, 1] contains pitch values
        order: The order of the Markov chain (default: 2)
        
    Returns:
        A tuple containing:
        - 'model': The Markov model as a nested dictionary
        - 'order': The order of the Markov model
        - 'total_transitions': Total number of transitions recorded
        - 'valid': Boolean indicating if the model is valid (i.e., has enough data)
        - 'last_pitches': The last 'order' pitches from memory for concatenation
        - 'last_intervals': The last 'order' time intervals from memory for concatenation
    """
    # Extract pitch values as integers
    pitches = np.round(memory[:, 1] * 127).astype(int)

    # Extract time intervals directly
    time_intervals = memory[:, 0]

    # Calculate pitch intervals between consecutive notes
    pitch_intervals = np.diff(pitches)

    # Calculate time multiples between consecutive intervals
    valid_multiples = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    common_multiples = [0.25, 0.5, 1.0, 2.0, 4.0]  # Common multiples for music
    time_multiples = []
    
    for i in range(len(time_intervals) - 1):
        if time_intervals[i] == 0 or time_intervals[i+1] == 0:
            # Avoid division by zero, set to 1.0 (no change in rhythm)
            time_multiples.append(1.0)
            continue
        multiple = time_intervals[i+1] / time_intervals[i]
        # Round to the nearest valid multiple
        rounded_multiple = min(valid_multiples, key=lambda x: abs(1 - (x / multiple)) if multiple in common_multiples else abs(1 - (x / multiple)) * 1.25)
        time_multiples.append(rounded_multiple)

    # Combine pitch intervals and time multiples into tuples
    combined_events = [(pitch_intervals[i], time_multiples[i]) for i in range(len(pitch_intervals))]
    
    # Initialize the Markov model
    model = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    
    # Store the last 'order' values for both pitches and time intervals
    last_pitches = pitches[-order:]
    last_intervals = time_intervals[-order:]
    
    # Build the Markov model
    if len(combined_events) < order + 1:
        # Not enough data to build the model of this order
        return dict(model), order, total_transitions, False, last_pitches, last_intervals
    
    # Convert combined events to tuple states for easier dictionary handling
    for i in range(len(combined_events) - order):
        # The state is the sequence of combined events before the transition
        state = tuple(combined_events[i:i+order])
        # The next combined event is the transition
        next_event = combined_events[i+order]
        
        # Update the model
        model[state][next_event] += 1
        total_transitions += 1
    
    # Convert defaultdict to regular dict for better serialization
    model_dict = {state: dict(transitions) for state, transitions in model.items()}

    return model_dict, order, total_transitions, True, last_pitches, last_intervals

def repetition_markov_heuristic(memory_model: Tuple, branch: np.ndarray, multiplier: float = 1.0) -> float:
    """
    Calculate how well the branch conforms to the combined pitch interval and time multiple
    Markov model from memory.
    
    Args:
        memory_model: The Markov model tuple returned by combined_markov_memory
        branch: np.ndarray of shape (n, 2) where branch[:, 0] contains time intervals
            and branch[:, 1] contains pitch values
        
    Returns:
        A float value between 0 and 1, where higher values indicate greater deviation
        from the expected combined patterns (i.e., worse conformity)
    """
    model, order, total_transitions, valid, last_pitches, last_intervals = memory_model
    
    # If the model is not valid, return 0 (no penalty)
    if not valid:
        return 0.0
    
    # Extract pitch values and convert to integers
    pitches = np.concatenate((last_pitches, np.round(branch[:, 1] * 127).astype(int)))

    # Extract time intervals with padding at start
    time_intervals = np.concatenate((last_intervals, branch[:, 0]))

    # Calculate pitch intervals between consecutive notes
    pitch_intervals = np.diff(pitches)

    # Calculate time multiples between consecutive intervals
    valid_multiples = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    common_multiples = [0.25, 0.5, 1.0, 2.0, 4.0]
    time_multiples = []
    
    for i in range(len(time_intervals) - 1):
        if time_intervals[i] == 0 or time_intervals[i+1] == 0:
            # Avoid division by zero, set to 1.0 (no change in rhythm)
            time_multiples.append(1.0)
            continue
        multiple = time_intervals[i+1] / time_intervals[i]
        # Round to the nearest valid multiple
        rounded_multiple = min(valid_multiples, key=lambda x: abs(1 - (x / multiple)) if multiple in common_multiples else abs(1 - (x / multiple)) * 1.25)
        time_multiples.append(rounded_multiple)

    # Combine pitch intervals and time multiples into tuples
    combined_events = [(pitch_intervals[i], time_multiples[i]) for i in range(len(pitch_intervals))]
    
    # Calculate probability for each transition in the branch
    probabilities = []
    
    for i in range(len(combined_events) - order):
        # The state is the sequence of combined events before the transition
        state = tuple(combined_events[i:i+order])
        # The next combined event is the transition we're evaluating
        next_event = combined_events[i+order]

        # Get the transition probabilities for this state
        if state in model:
            state_transitions = model[state]
            # Count total transitions from this state
            total_state_transitions = sum(state_transitions.values())
            
            # Calculate probability
            if next_event in state_transitions:
                prob = state_transitions[next_event] / total_state_transitions
            else:
                # No transition observed, no repetition
                prob = 0
        else:
            # If state never observed, no repetition
            prob = 0
        probabilities.append(prob)
    
    # Average probability across all transitions
    avg_probability = np.mean(probabilities)
    
    # Return 1 - avg_probability as the heuristic (higher value = worse conformity)
    return (1.0 - avg_probability) * 5 * multiplier

######################################################
# TESTING HEURISTICS (not used in the final version) #
######################################################

def rand_heuristic(memory: np.ndarray, branch: np.ndarray) -> float:
    # Return rand for all branches
    return np.random.rand()

def null_heuristic(memory: np.ndarray, branch: np.ndarray) -> float:
    # Return 0 for all branches
    return 0

def rhythmic_consistency(memory: np.ndarray, branch: np.ndarray) -> float:
    # Return the standard deviation of the first elements in each array within this 2d array
    return -np.std(branch[:, 0])

def rhythmic_range(memory: np.ndarray, branch: np.ndarray) -> float:
    # Return the difference between the min and max of the first elements in each array within this 2d array, + 0.01 * length
    return (np.min(branch[:, 0]) - np.max(branch[:, 0])) + 0.01 * len(branch)

def rhythmic_consistency_to_value(memory: np.ndarray, branch: np.ndarray, value=0.25, verbose=False) -> float:
    # Return the average deviation of the first elements in each array within this 2d array from the value
    if verbose:
        print("Running heuristic")
        print("Intervals: ", branch[:, 0])
        # print the deviation of each element from the value
        print("Deviations:", np.abs(branch[:, 0] - value))
        print("Value:", value)
        print("Heuristic value:", -np.mean(np.abs(branch[:, 0] - value)))
    return -np.mean(np.abs(branch[:, 0] - value))

def pitch_consistency(memory: np.ndarray, branch: np.ndarray) -> float:
    # Return the standard deviation of the second elements in each array within this 2d array
    return -np.std(branch[:, 1])

def overall_consistency(memory: np.ndarray, branch: np.ndarray) -> float:
    return rhythmic_consistency(branch) + pitch_consistency(branch)

def four_note_repetition(memory: np.ndarray, branch: np.ndarray) -> float: #TODO make this work for more than 2D
    branches_mod_4 = [branch[np.arange(len(branch)) % 4 == i] for i in range(4)]
    std_total = 0
    # Subtract the standard deviation of each mod arrays time and pitch, add the overall time and pitch standard deviation for the last 4 elements
    for b in branches_mod_4:
        if b.size > 0:
            std_total -= np.std(b[:, 0])
            std_total -= np.std(b[:, 1])
    std_total += np.std(branch[-4:, 0])
    std_total += np.std(branch[-4:, 1])
    return std_total
