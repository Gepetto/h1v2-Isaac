#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <base_directory>"
  exit 1
fi

BASE_DIR="$1"

# Check if the directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory '$BASE_DIR' does not exist."
  exit 1
fi

# Loop through all subdirectories
for dir in "$BASE_DIR"/*/; do
  # Skip if not a directory
  [ -d "$dir" ] || continue

  # Extract the subfolder name
  folder_name=$(basename "$dir")

  # Run the Python command
  python scripts/clean_rl/play.py \
    --task Isaac-Velocity-CaT-Flat-H12_12dof-Play-v0 \
    --headless \
    --video \
    --video_length 200 \
    --experiment_name "jz/${folder_name}"
done
