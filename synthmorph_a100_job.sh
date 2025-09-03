#!/bin/bash
#SBATCH --gres=gpu:a100:1 
#SBATCH -c 4
#SBATCH --mem 160G
#SBATCH --time=00:30:00                         # Time limit (D-HH:MM:SS
#SBATCH --propagate=NONE
#SBATCH --partition=aishort                         # Replace with your cluster's GPU partition name
#SBATCH --ntasks=1                              # Number of tasks (usually 1 for single-node jobs)
#SBATCH --mail-type=NONE                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=giannoua@pennmedicine.upenn.edu
#SBATCH --verbose

# Anaconda
# source /home/giannoua/miniconda3/bin/activate
source activate mri

# Load necessary modules
#module unload cuda
module load cuda/12.2

module unload cudnn
module load cudnn/8.9.7

#module load freesurfer/8.1.0

# Start TensorBoard (optional, useful for monitoring)
# tensorboard --logdir=logs/tensorboard --port=6006 &    # Uncomment to start TensorBoard in the background

nvidia-smi
nvcc --version

# Set TensorFlow environment variables for A100 GPU compatibility
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

# Disable problematic cuDNN optimizations that cause narrowing errors
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=1
export TF_CUDNN_DETERMINISTIC=1

# Additional CUDA settings for stability
export CUDA_VISIBLE_DEVICES=0

echo "TensorFlow environment variables set for A100 compatibility"

# Positional args:
#   $1 = subject image path
#   $2 = subject segmentation path
#   $3 = template image path
#   $4 = seg-method
#   $5 = target-roi
#   $6 = subject-id
SUBJECT_IMAGE="$1"
SUBJECT_SEG="$2"
TEMPLATE_IMAGE="$3"
SEG_METHOD="$4"
TARGET_ROI="$5"
SUBJECT_ID="$6"

# Run the synthmorph registration script on GPU
python synthmorph_ravens_pipeline.py \
  "$SUBJECT_IMAGE" \
  "$SUBJECT_SEG" \
  "$TEMPLATE_IMAGE" \
  --subject-id "$SUBJECT_ID" \
  --seg-method "$SEG_METHOD" \
  --target-roi "$TARGET_ROI"

# Optional: Wait for all background processes to finish
wait
