#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00                         # Time limit (D-HH:MM:SS
#SBATCH --propagate=NONE
#SBATCH --ntasks=1                              # Number of tasks (usually 1 for single-node jobs)
#SBATCH --time=02:00:00                         # Time limit (D-HH:MM:SS)
#SBATCH --verbose
#SBATCH --output="$SCRIPT_DIR"/logs/synthmorph-cpu-%j.out


# Anaconda
# source /home/giannoua/miniconda3/bin/activate
source activate mri

# Positional args:
#   $1 = subject image path
#   $2 = subject segmentation path
#   $3 = template image path
#   $4 = seg-method
#   $5 = target-roi
SUBJECT_IMAGE="$1"
SUBJECT_SEG="$2"
TEMPLATE_IMAGE="$3"
SEG_METHOD="$4"
TARGET_ROI="$5"

# Run the synthmorph registration script on CPU
python synthmorph_ravens_pipeline.py \
  "$SUBJECT_IMAGE" \
  "$SUBJECT_SEG" \
  "$TEMPLATE_IMAGE" \
  --seg-method "$SEG_METHOD" \
  --target-roi "$TARGET_ROI"

# Optional: Wait for all background processes to finish
wait
