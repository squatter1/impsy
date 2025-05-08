import numpy as np
from typing import Callable

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
    memory = np.round(memory[:, 1]*100).astype(int)
    branch = np.round(branch[:, 1]*100).astype(int)
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
