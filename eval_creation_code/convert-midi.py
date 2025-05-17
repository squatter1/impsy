#!/usr/bin/env python3
import os
import sys
import mido
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import random

def midi_to_log(midi_file_path, output_file_path):
    """
    Convert a MIDI file to a log file with the specified format.
    Each MIDI note is converted to a log entry with:
    - timestamp (starting from 01/01/2000 for the first note)
    - "interface" as the second column
    - MIDI note value / 127 as a float for the third column
    
    Returns the number of MIDI notes processed
    """
    mid = mido.MidiFile(midi_file_path)
    
    # Start time is January 1, 2000
    current_time = datetime(2000, 1, 1)
    
    # Default tempo (microseconds per beat) - will be updated when tempo events are encountered
    # Set bpm to random between 60 and 180 initially
    rand_bpm = random.randint(60, 180)
    tempo = mido.bpm2tempo(rand_bpm)
    
    # Flatten all tracks and sort by absolute time
    events = []
    current_ticks = 0
    
    for track in mid.tracks:
        track_ticks = 0
        for msg in track:
            track_ticks += msg.time
            events.append((track_ticks, msg))
    
    # Sort events by absolute time
    events.sort(key=lambda x: x[0])
    
    # Process events in chronological order
    notes = []
    prev_ticks = 0
    
    for ticks, msg in events:
        # Calculate time difference
        delta_ticks = ticks - prev_ticks
        delta_seconds = mido.tick2second(delta_ticks, mid.ticks_per_beat, tempo)
        current_time += timedelta(seconds=delta_seconds)
        prev_ticks = ticks
        
        # Check for tempo changes
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        
        # Process note events
        elif msg.type == 'note_on' and msg.velocity > 0:
            # Format as specified: timestamp,interface,note_value/127
            timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            normalized_note = msg.note / 127.0
            note_entry = f"{timestamp},interface,{normalized_note}"
            notes.append(note_entry)
    
    # Write to output file
    with open(output_file_path, 'w') as f:
        for note in notes:
            f.write(note + '\n')
    
    return len(notes)

def process_midi_folder(input_folder, output_folder):
    """
    Process all .mid files in the input folder and convert them to .log files
    in the output folder with the naming format i-n.log, where:
    - i is the sequential number of the file processed
    - n is the number of MIDI notes in the file
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    
    # Get all .mid files
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mid')]
    
    # Process each MIDI file
    for idx, midi_file in enumerate(midi_files, start=1):
        input_path = os.path.join(input_folder, midi_file)
        
        # Process the file and get the note count
        note_count = midi_to_log(input_path, "temp_output.log")
        
        # Create the output filename with the required format
        output_filename = f"{idx}-{note_count}.log"
        output_path = os.path.join(output_folder, output_filename)
        
        # Rename the temporary file to the final filename
        os.rename("temp_output.log", output_path)
        
        print(f"Converted {midi_file} to {output_filename} ({note_count} notes)")

if __name__ == "__main__":
    # Process the MIDI files
    process_midi_folder("nottingham-dataset-midi-melodies", "nottingham-dataset-logs")
    
    print("Conversion complete!")