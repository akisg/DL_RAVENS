#!/bin/bash

set -euo pipefail

# Activate env
source activate mri

# Optional modules (cluster-specific)
module load freesurfer/8.1.0 || true

export FSLOUTPUTTYPE=NIFTI_GZ

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

print_usage() {
  echo "Usage: $0 --pipeline ants|synthmorph --template <template_t1.nii.gz> --subject <subject_t1.nii.gz> --outdir <output_dir> [--seg-method fast|synthseg_freesurfer|synthseg_github|dlicv] [--target-roi csf|gray_matter|white_matter|background] [--shape 256,256,256] [--voxel-size 1] [--force-cpu] [--device cpu|gpu]"
}

PIPELINE="synthmorph"
TEMPLATE=""
SUBJECT=""
OUTDIR=""
SEG_METHOD="fast"
TARGET_ROI="csf"
SHAPE="256,256,256"
FORCE_CPU=false
DEVICE="cpu"
VOX_MM=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --subject) SUBJECT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --seg-method) SEG_METHOD="$2"; shift 2 ;;
    --target-roi) TARGET_ROI="$2"; shift 2 ;;
    --shape) SHAPE="$2"; shift 2 ;;
    --force-cpu) FORCE_CPU=true; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --voxel-size) VOX_MM="$2"; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown option: $1"; print_usage; exit 1 ;;
  esac
done

if [[ -z "$TEMPLATE" || -z "$SUBJECT" || -z "$OUTDIR" ]]; then
  echo "ERROR: Missing required args"; print_usage; exit 1
fi

mkdir -p "$OUTDIR"

# 0) Early resample/regrid: reshape to SHAPE and set voxel size to VOX_MM
# Resample subject and template using dlravens_preprocessing.py
ORIG_SUBJECT_ID=$(basename "$SUBJECT")
# strip extensions and known suffixes
ORIG_SUBJECT_ID="${ORIG_SUBJECT_ID%.nii.gz}"
ORIG_SUBJECT_ID="${ORIG_SUBJECT_ID%.nii}"
ORIG_SUBJECT_ID="${ORIG_SUBJECT_ID%_reshaped}"
ORIG_SUBJECT_ID="${ORIG_SUBJECT_ID%_T1_LPS}"
ORIG_SUBJECT_ID="${ORIG_SUBJECT_ID%_T1}"

PREPROC_SUBDIR="$OUTDIR/init/${ORIG_SUBJECT_ID}/"


RESAMPLE_OUTPUT=$(python "$SCRIPT_DIR"/dlravens_preprocessing.py \
  --subject "$SUBJECT" \
  --template "$TEMPLATE" \
  --outdir "$PREPROC_SUBDIR" \
  --shape "$SHAPE" \
  --voxel-size "$VOX_MM" | cat)

SUBJECT_RESHAPED=$(printf "%s\n" "$RESAMPLE_OUTPUT" | awk -F= '/^SUBJECT_OUT=/{print $2}')
TEMPLATE_RESHAPED=$(printf "%s\n" "$RESAMPLE_OUTPUT" | awk -F= '/^TEMPLATE_OUT=/{print $2}')

if [[ -z "$SUBJECT_RESHAPED" || -z "$TEMPLATE_RESHAPED" ]]; then
  echo "ERROR: Failed to obtain reshaped outputs from dlravens_preprocessing.py"; exit 1
fi

# Use reshaped inputs for all downstream steps
SUBJECT="$SUBJECT_RESHAPED"
TEMPLATE="$TEMPLATE_RESHAPED"

# 1) Preprocess + segmentation (submit via sbatch and wait)
echo "Submitting preprocessing job via sbatch (blocking until completion)..."
mkdir -p "$SCRIPT_DIR"/logs
sbatch --wait \
  --output="$SCRIPT_DIR"/logs/init-%j.out \
  --cpus-per-task=8 \
  --mem-per-cpu=4G \
  --time=06:00:00 \
  --propagate=NONE \
  --ntasks=1 \
  --verbose \
  -- \
  "$SCRIPT_DIR"/preprocessing_segmentation_pipeline.sh \
  ${FORCE_CPU:+--force-cpu} \
  --shape "$SHAPE" \
  --seg-method "$SEG_METHOD" \
  --target-roi "$TARGET_ROI" \
  "$SUBJECT" \
  "$PREPROC_SUBDIR" \
  "$ORIG_SUBJECT_ID"

# Determine outputs from preprocessing (preserve original subject id)
SUBJECT_ID="$ORIG_SUBJECT_ID"
REORIENTED="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS.nii.gz"
SEG_OUT_FAST="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS_fast_seg.nii.gz"
SEG_OUT_SYNTHFS="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS_synthseg_freesurfer_seg.nii.gz"
SEG_OUT_SYNTHGH="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS_synthseg_github_seg.nii.gz"
SEG_OUT_DLICV="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS_dlicvmask.nii.gz"
DLICV_OUT="$PREPROC_SUBDIR/${SUBJECT_ID}_T1_LPS_dlicv.nii.gz"

case "$SEG_METHOD" in
  fast) SUBJECT_SEG="$SEG_OUT_FAST" ;;
  synthseg_freesurfer) SUBJECT_SEG="$SEG_OUT_SYNTHFS" ;;
  synthseg_github) SUBJECT_SEG="$SEG_OUT_SYNTHGH" ;;
  dlicv) SUBJECT_SEG="$SEG_OUT_DLICV" ;;
 esac

if [[ ! -f "$SUBJECT_SEG" ]]; then
  echo "ERROR: Expected segmentation not found: $SUBJECT_SEG"; exit 1
fi

# 2) Choose pipeline
if [[ "$PIPELINE" == "ants" ]]; then
  # ANTs/GenerateRAVENS pipeline
  OUT_SUBDIR="$OUTDIR/ants/${SUBJECT_ID}"
  mkdir -p "$OUT_SUBDIR"
  "$SCRIPT_DIR"/run_calc_csf_ravens.sh \
    -s "$DLICV_OUT" \
    -l "$SUBJECT_SEG" \
    -t "$TEMPLATE" \
    -o "$OUT_SUBDIR" \
    -m ants
else
  # SynthMorph registration pipeline
  export OUTPUT_DIR="$OUTDIR/synthmorph"
  mkdir -p "$OUTPUT_DIR"

  # Download weights file
  WEIGHTS_URL="https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5"
  WEIGHTS_FILE=$(basename "$WEIGHTS_URL")
  if [[ -f "$WEIGHTS_FILE" ]]; then
    echo "Model weights already present: $WEIGHTS_FILE"
  else
    echo "Downloading model weights: $WEIGHTS_FILE"
    curl -fL -O "$WEIGHTS_URL"
  fi

  if [[ "$DEVICE" == "gpu" ]]; then
    # Submit GPU SynthMorph job (A100)
    sbatch --output="$SCRIPT_DIR"/logs/synthmorph-a100-%j.out -- \
      "$SCRIPT_DIR"/synthmorph_a100_job.sh \
        "$DLICV_OUT" \
        "$SUBJECT_SEG" \
        "$TEMPLATE" \
        "$SEG_METHOD" \
        "$TARGET_ROI" \
        "$SUBJECT_ID"
  else
    # Submit CPU SynthMorph job
    sbatch --output="$SCRIPT_DIR"/logs/synthmorph-cpu-%j.out -- \
      "$SCRIPT_DIR"/synthmorph_cpu_job.sh \
        "$DLICV_OUT" \
        "$SUBJECT_SEG" \
        "$TEMPLATE" \
        "$SEG_METHOD" \
        "$TARGET_ROI" \
        "$SUBJECT_ID"
  fi
fi

wait