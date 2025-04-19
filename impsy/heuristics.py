import numpy as np

def null_heuristic(branch: np.ndarray) -> float:
    # Return 0 for all branches
    return 0.0

def rhythmic_consistency(branch: np.ndarray) -> float:
    # Return the standard deviation of the first elements in each array within this 2d array
    return -np.std(branch[:, 0])

def rhythmic_range(branch: np.ndarray) -> float:
    # Return the difference between the min and max of the first elements in each array within this 2d array, + 0.01 * length
    return (np.min(branch[:, 0]) - np.max(branch[:, 0])) + 0.01 * len(branch)

def rhythmic_consistency_to_value(branch: np.ndarray, value=0.25) -> float:
    # Return the average deviation of the first elements in each array within this 2d array from the value
    return -np.mean(np.abs(branch[:, 0] - value))

def pitch_consistency(branch: np.ndarray) -> float:
    # Return the standard deviation of the second elements in each array within this 2d array
    return -np.std(branch[:, 1])

def overall_consistency(branch: np.ndarray) -> float:
    return rhythmic_consistency(branch) + pitch_consistency(branch)

def four_note_repetition(branch: np.ndarray) -> float: #TODO make this work for more than 2D
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
