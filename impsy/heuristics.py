import numpy as np
from typing import Callable, Tuple

#######################################
# FINAL HEURISTIC PARAMETER FUNCTIONS #
#######################################

def parameter_to_heuristic(memory: np.ndarray, branch: np.ndarray, parameter_function: Callable) -> float:
    return abs(parameter_function(memory) - parameter_function(branch))

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

class Scale:
    def __init__(self, notes: np.ndarray):
        self.notes = notes
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
    major_scale = Scale(np.array([0, 2, 4, 5, 7, 9, 11]))
    major_conformity, major_root = major_scale.conformity(branch)
    # Calculate conformity to the pentatonic scale
    pentatonic_scale = Scale(np.array([0, 2, 4, 7, 9]))
    pentatonic_conformity, pentatonic_root = pentatonic_scale.conformity(branch)
    # Calculate conformity to the blues scale
    blues_scale = Scale(np.array([0, 3, 5, 6, 7, 10]))
    blues_conformity, blues_root = blues_scale.conformity(branch)
    # Calculate conformity to the harmonic minor scale
    harmonic_minor_scale = Scale(np.array([0, 2, 3, 5, 7, 8, 11]))
    harmonic_minor_conformity, harmonic_minor_root = harmonic_minor_scale.conformity(branch)

    # If any of these conformity values are above the min_key_conformity, return the max conformity value
    max_conformity = max(major_conformity, pentatonic_conformity, blues_conformity, harmonic_minor_conformity)
    argmax_conformity = np.argmax([major_conformity, pentatonic_conformity, blues_conformity, harmonic_minor_conformity])
    max_scale = [major_scale, pentatonic_scale, blues_scale, harmonic_minor_scale][argmax_conformity]
    max_root = [major_root, pentatonic_root, blues_root, harmonic_minor_root][argmax_conformity]
    if max_conformity > min_key_conformity:
        return max_conformity, max_scale, max_root
    return 0, None, None

def key_and_modal_conformity_heuristic(memory: np.ndarray, branch: np.ndarray, min_key_conformity: float = 0.75, min_mode_conformity: float = 0.25) -> float:
    if len(memory) < 10:
        # Not enough memory to establish a key
        return 0
    memory = np.round(memory[:, 1]*127).astype(int) # *127 converts to 12 note scale
    branch = np.round(branch[:, 1]*127).astype(int) # *127 converts to 12 note scale
    # Calculate key conformity for memory
    memory_conformity, memory_scale, memory_root = key_conformity(memory, min_key_conformity)
    if memory_scale is None:
        # Chromatic scale, don't use key conformity as a heuristic
        return 0
    
    # Calculate conformity of the branch to the memory scale
    branch_conformity = memory_scale.root_conformity(branch, memory_root)

    # Calculate modal conformity
    memory_mode_conformity, memory_mode = memory_scale.mode_conformity_all(memory, memory_root)
    if memory_mode_conformity < min_mode_conformity:
        # Not enough memory to establish a mode
        return abs(branch_conformity - memory_conformity) / 2

    # Calculate conformity of the branch to the memory mode
    branch_mode_conformity = memory_scale.mode_conformity(branch, memory_root, memory_mode)

    # Return the difference between the branch conformity and the memory conformity for key and mode
    return (abs(branch_conformity - memory_conformity) + abs(branch_mode_conformity - memory_mode_conformity)) / 2

def estimate_tempo_and_swing(durations: np.ndarray, 
                             tempo_range: Tuple[int, int] = (80, 158), 
                             tempo_step: int = 2) -> Tuple[float, str, float]:
    """
    Estimate the tempo and swing ratio from a sequence of durations.
    Returns the most likely tempo (bpm), swing type, and confidence score.
    """
    # Convert durations to seconds (assuming they're in seconds already)
    durations = durations.flatten()
    
    best_score = -float('inf')
    best_tempo = 0
    best_swing = "none"
    
    # Test different tempos and swing values
    tempos = range(tempo_range[0], tempo_range[1]+1, tempo_step)
    # Swing options: none, triplet (2:1), golden ratio (1.618:1) (unused)
    swing_options = {
        "none": 1.0,
        "triplet": 2.0
    }
    
    for tempo in tempos:
        beat_duration = 60 / tempo  # Duration of one quarter note in seconds
        
        for swing_name, swing_ratio in swing_options.items():
            score = 0
            
            # Expected durations in this tempo/swing combination
            # Common note durations: whole, half, quarter, 8th, 16th notes and their dotted variants
            expected_durations = []
            
            # Regular note durations
            expected_durations.append(beat_duration * 4)  # Whole note
            expected_durations.append(beat_duration * 2)  # Half note
            expected_durations.append(beat_duration)      # Quarter note
            
            # Eighth notes (with swing)
            if swing_name == "none":
                expected_durations.append(beat_duration / 2)  # Eighth note
                expected_durations.append(beat_duration / 4)  # Sixteenth note
            else:
                # First eighth note in a pair (longer in swing)
                expected_durations.append(beat_duration * swing_ratio / (1 + swing_ratio))
                # Second eighth note in a pair (shorter in swing)
                expected_durations.append(beat_duration / (1 + swing_ratio))
            
            # Add dotted variants
            dotted_durations = [d * 1.5 for d in expected_durations]
            expected_durations.extend(dotted_durations)
            
            # Calculate score based on how well durations match expected durations
            for duration in durations:
                # Find closest expected duration
                score -= min(expected_durations, key=lambda x: abs(x - duration) / duration)
            
            # Normalize score by number of notes
            score /= len(durations)

            # Normalise by expected duration density
            score *= beat_duration
            
            if score > best_score:
                best_score = score
                best_tempo = tempo
                best_swing = swing_name
    
    return best_tempo, best_swing, best_score

def detect_time_signature(onsets: np.ndarray, tempo: float) -> Tuple[int, float]:
    """
    Detect the time signature based on note onsets and the estimated tempo.
    Returns the most likely beats per bar and confidence score.
    """
    beat_duration = 60 / tempo
    
    # Convert absolute onsets to beats
    beat_positions = onsets / beat_duration
    
    # Test different bar lengths
    bar_lengths = [3, 4, 5, 7]  # 3/4, 4/4, 5/4, 7/4 time signatures
    
    best_score = -float('inf')
    best_bar_length = 4  # Default to 4/4
    
    for bar_length in bar_lengths:
        scores = np.zeros(bar_length)
        
        # Test each possible downbeat offset
        for offset in range(bar_length):
            # Calculate modular positions within the bar
            positions = (beat_positions + offset) % bar_length
            
            # Count notes at each beat position
            for pos in positions:
                beat_index = int(pos)
                if beat_index < bar_length:
                    scores[beat_index] += 1
        
        # Normalize counts
        if np.sum(scores) > 0:
            scores = scores / np.sum(scores)
        
        # Calculate score based on:
        # 1. Stronger first beat (downbeat)
        # 2. Overall distribution matching expected patterns
        downbeat_strength = scores[0] if np.sum(scores) > 0 else 0
        
        # Expected distribution (first beat is strongest, other beats have typical patterns)
        if bar_length == 4:
            # 4/4: Strong-weak-medium-weak pattern
            expected = np.array([0.4, 0.15, 0.3, 0.15])
        elif bar_length == 3:
            # 3/4: Strong-weak-weak pattern
            expected = np.array([0.5, 0.25, 0.25])
        elif bar_length == 5:
            # 5/4 often grouped as 3+2 or 2+3
            expected = np.array([0.3, 0.15, 0.2, 0.15, 0.2])
        elif bar_length == 7:
            # 7/4 often grouped as 4+3 or 3+4
            expected = np.array([0.25, 0.1, 0.15, 0.1, 0.2, 0.1, 0.1])
        
        # Calculate similarity to expected pattern
        pattern_similarity = 1 - np.mean(np.abs(scores - expected))
        
        # Combined score
        total_score = 0.7 * downbeat_strength + 0.3 * pattern_similarity
        
        if total_score > best_score:
            best_score = total_score
            best_bar_length = bar_length
    
    return best_bar_length, best_score

def tempo_swing_time_heuristic(memory: np.ndarray, branch: np.ndarray, 
                              min_confidence: float = 0.6) -> float:
    """
    Calculate heuristic based on tempo, swing and time signature consistency.
    """
    if len(memory) < 8:  # Need enough notes to reliably estimate tempo
        return 0
    
    # Extract durations from memory and branch
    memory_durations = memory[:, 0]
    branch_durations = branch[:, 0]
    
    # Calculate absolute note onset times
    memory_onsets = np.cumsum(np.hstack(([0], memory_durations[:-1])))
    branch_onsets = np.cumsum(np.hstack(([0], branch_durations[:-1])))
    
    # Estimate tempo and swing from memory
    memory_tempo, memory_swing, tempo_confidence = estimate_tempo_and_swing(memory_durations)
    print("Memory tempo:", memory_tempo, "Swing:", memory_swing, "Confidence:", tempo_confidence)
    return 0
    
    ## Only proceed if confidence is high enough
    #if tempo_confidence < min_confidence:
    #    return 0
    #
    ## Detect time signature
    #memory_time_sig, time_sig_confidence = detect_time_signature(memory_onsets, memory_tempo)
    #
    ## Calculate same for branch
    #branch_tempo, branch_swing, _ = estimate_tempo_and_swing(branch_durations)
    #branch_time_sig, _ = detect_time_signature(branch_onsets, branch_tempo)
    #
    ## Calculate tempo similarity (normalized to 0-1)
    #tempo_diff = abs(memory_tempo - branch_tempo) / max(memory_tempo, branch_tempo)
    #tempo_similarity = 1 - min(tempo_diff, 1)
    #
    ## Swing similarity (1 if same, 0 if different)
    #swing_similarity = 1.0 if memory_swing == branch_swing else 0.0
    #
    ## Time signature similarity (1 if same, 0 if different)
    #time_sig_similarity = 1.0 if memory_time_sig == branch_time_sig else 0.0
    #
    ## Weighted combination
    #if time_sig_confidence < min_confidence:
    #    # If time signature detection is unreliable, only use tempo and swing
    #    return (2 * tempo_similarity + swing_similarity) / 3
    #else:
    #    return (0.5 * tempo_similarity + 0.25 * swing_similarity + 0.25 * time_sig_similarity)





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
