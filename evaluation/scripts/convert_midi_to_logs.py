"""Converts a folder of MIDI melodies to IMPSY 2D logs named i-n.log, where i is the file number and n its note count."""

import argparse
import mido
import random
from datetime import datetime, timedelta
from pathlib import Path


def midi_to_log(midi_file: Path, output_file: Path, tempo_range=(60, 180)) -> int:
    """Writes each note-on as `timestamp,interface,note/127`, starting from 2000-01-01 at a random initial tempo.
    Returns the number of notes written."""
    mid = mido.MidiFile(midi_file)
    current_time = datetime(2000, 1, 1)
    tempo = mido.bpm2tempo(random.randint(*tempo_range))

    # Flatten all tracks and sort by absolute time
    events = []
    for track in mid.tracks:
        track_ticks = 0
        for msg in track:
            track_ticks += msg.time
            events.append((track_ticks, msg))
    events.sort(key=lambda event: event[0])

    notes = []
    prev_ticks = 0
    for ticks, msg in events:
        current_time += timedelta(seconds=mido.tick2second(ticks - prev_ticks, mid.ticks_per_beat, tempo))
        prev_ticks = ticks
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            notes.append(f"{current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')},interface,{msg.note / 127.0}")

    output_file.write_text("\n".join(notes) + "\n")
    return len(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path, help="Folder of .mid files")
    parser.add_argument("output_folder", type=Path, help="Folder to write the .log files to")
    parser.add_argument("--seed", type=int, help="Seed for the random initial tempo of each file")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    args.output_folder.mkdir(exist_ok=True, parents=True)
    midi_files = sorted(f for f in args.input_folder.iterdir() if f.suffix.lower() == ".mid")
    for idx, midi_file in enumerate(midi_files, start=1):
        # Write to a temporary name first since the final name needs the note count
        temp_file = args.output_folder / f"{idx}.tmp"
        note_count = midi_to_log(midi_file, temp_file)
        output_file = args.output_folder / f"{idx}-{note_count}.log"
        temp_file.rename(output_file)
        print(f"Converted {midi_file.name} to {output_file.name} ({note_count} notes)")
    print(f"Converted {len(midi_files)} files")


if __name__ == "__main__":
    main()
