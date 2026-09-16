"""Randomly selects converted logs (named i-n.log) with more than a minimum number of notes until a total note count is reached."""

import argparse
import random
import re
from pathlib import Path

LOG_PATTERN = re.compile(r"(\d+)-(\d+)\.log")


def random_file_selection(directory: Path, min_notes: int, total_notes_threshold: int):
    """Returns the selected (i, n) pairs and their total note count."""
    eligible = []
    for path in directory.iterdir():
        match = LOG_PATTERN.fullmatch(path.name)
        if match and int(match.group(2)) > min_notes:
            eligible.append((int(match.group(1)), int(match.group(2))))
    random.shuffle(eligible)

    selected, total_notes = [], 0
    for i_value, n_value in eligible:
        selected.append((i_value, n_value))
        total_notes += n_value
        if total_notes > total_notes_threshold:
            break
    return selected, total_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("nottingham-dataset-logs"))
    parser.add_argument("--min-notes", type=int, default=200)
    parser.add_argument("--total-notes", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")

    selected, total_notes = random_file_selection(args.directory, args.min_notes, args.total_notes)
    if not selected:
        parser.exit(1, f"No files found with more than {args.min_notes} notes\n")

    print(f"{'i':<10} {'n':<10}")
    print("-" * 20)
    for i_value, n_value in selected:
        print(f"{i_value:<10} {n_value:<10}")
    print(f"\nTotal files selected: {len(selected)}")
    print(f"Total notes: {total_notes} (target {args.total_notes})")
    print("Selected i values:", sorted(i for i, _ in selected))


if __name__ == "__main__":
    main()
