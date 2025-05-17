#!/usr/bin/env python3
import os
import re
import random
from pathlib import Path

def random_file_selection(directory_path='nottingham-dataset-logs/', 
                          min_notes=200, 
                          total_notes_threshold=10000):
    """
    Randomly select files from directory that have over min_notes notes,
    collect their i and n values without repeats, until the total notes
    exceeds the threshold.
    
    Args:
        directory_path (str): Path to the directory containing log files
        min_notes (int): Minimum number of notes required for a file to be considered
        total_notes_threshold (int): Target threshold for total notes
        
    Returns:
        list: Selected (i, n) pairs
        int: Total number of notes in the selected files
    """
    # Ensure the directory path exists
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory '{directory_path}' does not exist or is not a directory")
        return [], 0
    
    # Pattern to match i-n.log files and extract i and n values
    pattern = re.compile(r'(\d+)-(\d+)\.log')
    
    # Get all eligible files
    eligible_files = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            i_value = int(match.group(1))
            n_value = int(match.group(2))
            
            # Include only files with more than min_notes
            if n_value > min_notes:
                eligible_files.append((i_value, n_value))
    
    if not eligible_files:
        print(f"No files found with more than {min_notes} notes.")
        return [], 0
    
    # Randomly select files until we reach the threshold
    selected_files = []
    selected_i_values = set()  # To ensure no repeats
    total_notes = 0
    
    # Shuffle the eligible files list for random selection
    random.shuffle(eligible_files)
    
    for i_value, n_value in eligible_files:
        if i_value not in selected_i_values:
            selected_files.append((i_value, n_value))
            selected_i_values.add(i_value)
            total_notes += n_value
            
            if total_notes > total_notes_threshold:
                break
    
    return selected_files, total_notes

def main():
    # Get randomly selected files
    selected_files, total_notes = random_file_selection()
    
    # Print the results
    print(f"Files selected from nottingham-dataset-logs/ with more than 200 notes:")
    print(f"{'i':<10} {'n':<10}")
    print("-" * 20)
    
    for i_value, n_value in selected_files:
        print(f"{i_value:<10} {n_value:<10}")
    
    print(f"\nTotal files selected: {len(selected_files)}")
    print(f"Total notes: {total_notes}")
    print(f"Target threshold: 10,000 notes")
    
    # Print the i values in order
    i_values = sorted([i for i, n in selected_files])
    print("\nList of i values in order:")
    print(i_values)

if __name__ == "__main__":
    main()