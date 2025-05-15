# rename_logs.py

import os
import re

# Define the folder path
folder = 'logs'

# Define the regex pattern for files like i-n.log (e.g., 1-5.log)
pattern = re.compile(r'^(\d+)-(\d+)\.log$')

# List all files in the folder
for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        i, n = match.groups()
        new_filename = f"{i}-{n}-2d-mdrnn.log"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)
        print(f"Renaming {filename} -> {new_filename}")
        os.rename(old_path, new_path)
