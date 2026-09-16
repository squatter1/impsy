"""Renames i-n.log files to i-n-2d-mdrnn.log so the dataset command picks them up."""

import argparse
import re
from pathlib import Path

LOG_PATTERN = re.compile(r"^(\d+)-(\d+)\.log$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=Path("logs"))
    args = parser.parse_args()

    for path in sorted(args.folder.iterdir()):
        match = LOG_PATTERN.match(path.name)
        if match:
            new_path = path.with_name(f"{match.group(1)}-{match.group(2)}-2d-mdrnn.log")
            print(f"Renaming {path.name} -> {new_path.name}")
            path.rename(new_path)


if __name__ == "__main__":
    main()
