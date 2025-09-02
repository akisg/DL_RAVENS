#!/bin/bash

# =============================================================================
# Brain MRI Segmentation Pipeline
# =============================================================================
# This script performs complete preprocessing and segmentation of T1 brain MRI
# Includes: Reorientation, DLICV brain masking, FAST tissue segmentation,
#          SynthSeg segmentation, shape resizing, and skull stripping
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths (modify these for your environment)
DLICV_MODELS="/cbica/home/erusg/doshijim_files/TensorFlow/LifeSpanData_FinalModels/Models"
DLICV_CONTAINER="/cbica/home/erusg/doshijim_files/Singularity/deepmrseg/deepmrseg_1.0.0.Alpha2.sif"

# FAST parameters
FAST_N_CLASSES=3
FAST_VERBOSE=1

# SynthSeg parameters
SYNTHSEG_THREADS=8
SYNTHSEG_COMMAND="mri_synthseg"  # or "python ../SynthSeg/scripts/commands/SynthSeg_predict.py"

# Shape parameters (must be divisible by 16 for SynthMorph)
DEFAULT_SHAPE="256,256,256"
ALTERNATIVE_SHAPE="128,128,128"

# =============================================================================
# FUNCTIONS
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS] <input_t1> <output_dir> [subject_id]"
    echo ""
    echo "Required arguments:"
    echo "  input_t1      Path to input T1 NIFTI file (.nii.gz)"
    echo "  output_dir    Directory to save all outputs"
    echo ""
    echo "Optional arguments:"
    echo "  subject_id    Subject identifier (default: extracted from filename)"
    echo ""
    echo "Options:"
    echo "  --scale       Enable intensity scaling (0-2048 range)"
    echo "  --no-dlicv    Skip DLICV brain masking (use original image)"
    echo "  --skull-strip Enable skull stripping using ANTs"
    echo "  --force-cpu   Force CPU mode for DLICV (skip GPU)"
    echo "  --shape <x,y,z> Resize image to specified shape (default: 256,256,256)"
    echo "  --seg-method <method> Segmentation method: fast, synthseg_freesurfer, synthseg_github, dlicv"
    echo "  --target-roi <roi> Target ROI for segmentation: csf, gray_matter, white_matter, background"
    echo "  --freesurfer-prefix <prefix> FreeSurfer environment prefix (e.g., 'source /path/to/freesurfer')"
    echo "  --help        Show this help message"
    echo ""
    echo "Segmentation Methods:"
    echo "  fast              FSL FAST tissue segmentation"
    echo "  synthseg_freesurfer  FreeSurfer SynthSeg (requires FreeSurfer)"
    echo "  synthseg_github   GitHub SynthSeg implementation"
    echo "  dlicv            DLICV deep learning segmentation"
    echo ""
    echo "Target ROIs:"
    echo "  csf              Cerebrospinal fluid"
    echo "  gray_matter      Gray matter"
    echo "  white_matter     White matter"
    echo "  background       Background"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/subject_T1.nii.gz /path/to/output subject_001"
    echo "  $0 --scale --skull-strip /path/to/subject_T1.nii.gz /path/to/output"
    echo "  $0 --seg-method synthseg_freesurfer --target-roi csf /path/to/subject_T1.nii.gz /path/to/output"
    echo "  $0 --shape 128,128,128 --seg-method fast /path/to/subject_T1.nii.gz /path/to/output"
}

check_dependencies() {
    local missing_deps=()
    
    # Check for required commands
    for cmd in 3dresample nifti1_test nifti_tool 3dBrickStat 3dcalc fast; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done
    
    # Check for DLICV container if not skipped and DLICV is selected
    if [[ "$SKIP_DLICV" != "true" ]] && [[ "$SEG_METHOD" == "dlicv" ]] && [[ ! -f "$DLICV_CONTAINER" ]]; then
        echo "ERROR: DLICV container not found at: $DLICV_CONTAINER"
        echo "Use --no-dlicv to skip DLICV processing or choose different segmentation method"
        exit 1
    fi
    
    # Check for FreeSurfer if SynthSeg is selected
    if [[ "$SEG_METHOD" == "synthseg_freesurfer" ]]; then
        if ! command -v mri_synthseg &> /dev/null; then
            echo "ERROR: FreeSurfer SynthSeg not found. Please ensure FreeSurfer is properly installed and sourced."
            echo "Use --freesurfer-prefix to specify FreeSurfer environment setup."
            exit 1
        fi
    fi
    
    # Check for ANTs if skull stripping is enabled
    if [[ "$SKULL_STRIP" == "true" ]]; then
        if ! command -v antsApplyTransforms &> /dev/null; then
            echo "ERROR: ANTs not found. Please install ANTs for skull stripping functionality."
            exit 1
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "ERROR: Missing required dependencies:"
        printf '  %s\n' "${missing_deps[@]}"
        echo ""
        echo "Please install AFNI and FSL:"
        echo "  AFNI: https://afni.nimh.nih.gov/"
        echo "  FSL: https://fsl.fmrib.ox.ac.uk/"
        exit 1
    fi
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

create_temp_dir() {
    local temp_dir=$(mktemp -d)
    echo "$temp_dir"
}

cleanup_temp() {
    if [[ -n "$TEMP_DIR" ]] && [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        log_message "Cleaned up temporary directory"
    fi
}

# Function to resize image to specified shape
resize_image() {
    local input_file="$1"
    local output_file="$2"
    local target_shape="$3"
    
    log_message "Resizing image to shape: $target_shape"
    
    # Parse shape dimensions
    IFS=',' read -ra SHAPE_DIMS <<< "$target_shape"
    local dim_x="${SHAPE_DIMS[0]}"
    local dim_y="${SHAPE_DIMS[1]}"
    local dim_z="${SHAPE_DIMS[2]}"
    
    # Use 3dresample to resize the image
    if [[ "$dim_x" == "128" && "$dim_y" == "128" && "$dim_z" == "128" ]]; then
        # For 128x128x128, we might need to downsample
        log_message "Downsampling to 128x128x128 (2mm voxels)"
        3dresample -dxyz 2 2 2 -prefix "$output_file" -inset "$input_file"
    else
        # For other shapes, use the specified dimensions
        log_message "Resizing to ${dim_x}x${dim_y}x${dim_z}"
        3dresample -dxyz "$dim_x" "$dim_y" "$dim_z" -prefix "$output_file" -inset "$input_file"
    fi
}

# Function to perform skull stripping using ANTs
skull_strip_ants() {
    local input_file="$1"
    local output_file="$2"
    
    log_message "Performing skull stripping using ANTs"
    
    # Use ANTs brain extraction
    antsBrainExtraction.sh -d 3 -a "$input_file" -e /usr/share/ants/templates/OASIS-30_Atropos_template/T_template0.nii.gz -m /usr/share/ants/templates/OASIS-30_Atropos_template/T_template0_BrainCerebellumProbabilityMask.nii.gz -o "${output_file%.nii.gz}_"
    
    # Move the brain-extracted image to the desired output location
    mv "${output_file%.nii.gz}_BrainExtractionBrain.nii.gz" "$output_file"
    
    # Clean up temporary files
    rm -f "${output_file%.nii.gz}_"*
}

# Function to run SynthSeg segmentation
run_synthseg() {
    local input_file="$1"
    local output_file="$2"
    local seg_method="$3"
    
    log_message "Running SynthSeg segmentation using method: $seg_method"
    
    local cmd=""
    if [[ "$seg_method" == "synthseg_freesurfer" ]]; then
        cmd="$FREESURFER_PREFIX $SYNTHSEG_COMMAND --i $input_file --o $output_file --robust --vol ${output_file%.nii.gz}_vol.csv --qc ${output_file%.nii.gz}_qc.csv --threads $SYNTHSEG_THREADS --cpu"
    elif [[ "$seg_method" == "synthseg_github" ]]; then
        cmd="python ../SynthSeg/scripts/commands/SynthSeg_predict.py --i $input_file --o $output_file --robust --vol ${output_file%.nii.gz}_vol.csv --qc ${output_file%.nii.gz}_qc.csv --threads $SYNTHSEG_THREADS --cpu"
    fi
    
    if [[ -n "$cmd" ]]; then
        log_message "Running command: $cmd"
        eval "$cmd"
    else
        echo "ERROR: Unknown SynthSeg method: $seg_method"
        exit 1
    fi
}

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

SCALE_INTENSITY=false
SKIP_DLICV=false
SKULL_STRIP=false
FORCE_CPU=false
SHAPE="$DEFAULT_SHAPE"
SEG_METHOD="fast"
TARGET_ROI="csf"
FREESURFER_PREFIX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scale)
            SCALE_INTENSITY=true
            shift
            ;;
        --no-dlicv)
            SKIP_DLICV=true
            shift
            ;;
        --skull-strip)
            SKULL_STRIP=true
            shift
            ;;
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        --shape)
            SHAPE="$2"
            shift 2
            ;;
        --seg-method)
            SEG_METHOD="$2"
            shift 2
            ;;
        --target-roi)
            TARGET_ROI="$2"
            shift 2
            ;;
        --freesurfer-prefix)
            FREESURFER_PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option $1"
            print_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "ERROR: Missing required arguments"
    print_usage
    exit 1
fi

INPUT_T1="$1"
OUTPUT_DIR="$2"
SUBJECT_ID="${3:-$(basename "$INPUT_T1" .nii.gz)}"
# Sanitize SUBJECT_ID in case an extension or known suffix slipped in
SUBJECT_ID="${SUBJECT_ID%.nii.gz}"
SUBJECT_ID="${SUBJECT_ID%.nii}"
SUBJECT_ID="${SUBJECT_ID%_reshaped}"
SUBJECT_ID="${SUBJECT_ID%_T1_LPS}"
SUBJECT_ID="${SUBJECT_ID%_T1}"

# =============================================================================
# VALIDATION
# =============================================================================

log_message "Starting enhanced brain segmentation pipeline for subject: $SUBJECT_ID"
log_message "Configuration:"
log_message "  Segmentation method: $SEG_METHOD"
log_message "  Target ROI: $TARGET_ROI"
log_message "  Shape: $SHAPE"
log_message "  Skull stripping: $SKULL_STRIP"
log_message "  Scale intensity: $SCALE_INTENSITY"
log_message "  Skip DLICV: $SKIP_DLICV"

# Check input file
if [[ ! -f "$INPUT_T1" ]]; then
    echo "ERROR: Input T1 file not found: $INPUT_T1"
    exit 1
fi

# Check file extension
if [[ ! "$INPUT_T1" =~ \.(nii|nii\.gz)$ ]]; then
    echo "ERROR: Input file must be NIFTI format (.nii or .nii.gz)"
    exit 1
fi

# Validate shape format
if [[ ! "$SHAPE" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "ERROR: Invalid shape format. Use format: x,y,z (e.g., 256,256,256)"
    exit 1
fi

# Validate segmentation method
case "$SEG_METHOD" in
    fast|synthseg_freesurfer|synthseg_github|dlicv)
        ;;
    *)
        echo "ERROR: Invalid segmentation method: $SEG_METHOD"
        echo "Valid options: fast, synthseg_freesurfer, synthseg_github, dlicv"
        exit 1
        ;;
 esac

# Validate target ROI
case "$TARGET_ROI" in
    csf|gray_matter|white_matter|background)
        ;;
    *)
        echo "ERROR: Invalid target ROI: $TARGET_ROI"
        echo "Valid options: csf, gray_matter, white_matter, background"
        exit 1
        ;;
 esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check dependencies
check_dependencies

# Set up cleanup trap
TEMP_DIR=""
trap cleanup_temp EXIT

# =============================================================================
# STEP 1: REORIENTATION AND PREPROCESSING
# =============================================================================

log_message "Step 1: Reorienting and preprocessing T1 image"

TEMP_DIR=$(create_temp_dir)
cd "$TEMP_DIR"

# Copy and prepare input file
input_base=$(basename "$INPUT_T1")
cp "$INPUT_T1" "$input_base"

# Determine base name without extension
if [[ "$input_base" == *.nii.gz ]]; then
    base_name="${input_base%.nii.gz}"
elif [[ "$input_base" == *.nii ]]; then
    base_name="${input_base%.nii}"
else
    base_name="$input_base"
fi

# Convert to NIFTI format if needed
nifti1_test -n2 "$input_base" "temp_${base_name}" 2>/dev/null || cp "$input_base" "temp_${base_name}.nii.gz"

# Reorient to LPS (Left-Posterior-Superior)
log_message "  Reorienting to LPS coordinate system"
3dresample -orient rai -prefix "temp_LPS.hdr" -inset "temp_${base_name}.hdr" 2>/dev/null || \
3dresample -orient rai -prefix "temp_LPS.nii.gz" -inset "temp_${base_name}.nii.gz"

# Clear sform matrix
log_message "  Clearing sform matrix"
nifti_tool -mod_hdr -mod_field sform_code 0 -prefix "temp_LPS_nosform.hdr" -infiles "temp_LPS.hdr" 2>/dev/null || \
nifti_tool -mod_hdr -mod_field sform_code 0 -prefix "temp_LPS_nosform.nii.gz" -infiles "temp_LPS.nii.gz"

# Convert to compressed NIFTI
nifti1_test -zn1 "temp_LPS_nosform.img" "reoriented" 2>/dev/null || \
nifti1_test -zn1 "temp_LPS_nosform.nii.gz" "reoriented"

# Intensity scaling (optional)
if [[ "$SCALE_INTENSITY" == "true" ]]; then
    log_message "  Scaling intensity range to 0-2048"
    min=$(3dBrickStat -slow -min "reoriented.nii.gz")
    max=$(3dBrickStat -slow -max "reoriented.nii.gz")
    3dcalc -a "reoriented.nii.gz" -expr "((a-${min})/(${max}-${min}))*2048" -short -nscale -prefix "reoriented_scaled.nii.gz"
    REORIENTED_FILE="reoriented_scaled.nii.gz"
else
    REORIENTED_FILE="reoriented.nii.gz"
fi

# Copy to output directory
REORIENTED_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS.nii.gz"
cp "$REORIENTED_FILE" "$REORIENTED_OUTPUT"
log_message "  Reoriented image saved to: $REORIENTED_OUTPUT"

# =============================================================================
# STEP 1.5: SHAPE RESIZING (if different from default)
# =============================================================================

if [[ "$SHAPE" != "$DEFAULT_SHAPE" ]]; then
    log_message "Step 1.5: Resizing image to specified shape"
    RESIZED_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_${SHAPE//,/_}.nii.gz"
    resize_image "$REORIENTED_OUTPUT" "$RESIZED_OUTPUT" "$SHAPE"
    PROCESSING_INPUT="$RESIZED_OUTPUT"
else
    PROCESSING_INPUT="$REORIENTED_OUTPUT"
fi

# =============================================================================
# STEP 1.6: SKULL STRIPPING (if enabled)
# =============================================================================

if [[ "$SKULL_STRIP" == "true" ]]; then
    log_message "Step 1.6: Performing skull stripping"
    SKULL_STRIPPED_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_skullstripped.nii.gz"
    skull_strip_ants "$PROCESSING_INPUT" "$SKULL_STRIPPED_OUTPUT"
    PROCESSING_INPUT="$SKULL_STRIPPED_OUTPUT"
fi

# =============================================================================
# STEP 2: DLICV BRAIN MASKING (OPTIONAL)
# =============================================================================

if [[ "$SKIP_DLICV" == "true" ]]; then
    log_message "Step 2: Skipping DLICV brain masking (using preprocessed image)"
    BRAIN_MASKED_FILE="$PROCESSING_INPUT"
    BRAIN_MASK_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_dlicvmask.nii.gz"
    # Create a simple mask (all ones) if DLICV is skipped
    3dcalc -a "$PROCESSING_INPUT" -expr "1" -prefix "$BRAIN_MASK_OUTPUT"
else
    log_message "Step 2: Running DLICV deep learning brain masking"
    
    # Check if singularity is available
    if ! command -v singularity &> /dev/null; then
        echo "ERROR: Singularity not found. Install singularity or use --no-dlicv"
        exit 1
    fi
    
    # Run DLICV segmentation
    BRAIN_MASK_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_dlicvmask.nii.gz"
    
    if [[ ! -f "$BRAIN_MASK_OUTPUT" ]]; then
        log_message "  Running DLICV model..."
        # Determine singularity flags based on CPU/GPU preference
        if [[ "$FORCE_CPU" == "true" ]]; then
            SINGULARITY_FLAGS="-B $DLICV_MODELS -B $TEMP_DIR -B $OUTPUT_DIR"
            log_message "  Running DLICV in CPU mode"
        else
            SINGULARITY_FLAGS="--nv -B $DLICV_MODELS -B $TEMP_DIR -B $OUTPUT_DIR"
            log_message "  Running DLICV in GPU mode"
        fi
        
        singularity run \
            $SINGULARITY_FLAGS \
            "$DLICV_CONTAINER" \
            deepmrseg_test \
            --mdlDir "$DLICV_MODELS/DLICV/Final/LPS/" \
            --mdlDir "$DLICV_MODELS/DLICV/Final/PSL/" \
            --mdlDir "$DLICV_MODELS/DLICV/Final/SLP/" \
            --inImg "$PROCESSING_INPUT" \
            --outImg "$BRAIN_MASK_OUTPUT" \
            --nJobs 4
    else
        log_message "  DLICV mask already exists, skipping"
    fi
    
    # Apply brain mask to T1 image
    log_message "  Applying brain mask to T1 image"
    BRAIN_MASKED_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_dlicv.nii.gz"
    3dcalc -prefix "$BRAIN_MASKED_OUTPUT" -a "$PROCESSING_INPUT" -b "$BRAIN_MASK_OUTPUT" -expr "a*step(b)" -verbose -nscale
    BRAIN_MASKED_FILE="$BRAIN_MASKED_OUTPUT"
fi

# =============================================================================
# STEP 3: SEGMENTATION
# =============================================================================

log_message "Step 3: Running segmentation using method: $SEG_METHOD"

# Determine input file for segmentation
# if [[ "$SEG_METHOD" == "dlicv" ]]; then
#     SEG_INPUT="$BRAIN_MASKED_FILE"
# else
#     SEG_INPUT="$PROCESSING_INPUT"
# fi

SEG_INPUT="$BRAIN_MASKED_FILE"


# Run appropriate segmentation method
case "$SEG_METHOD" in
    fast)
        FAST_OUTPUT_PREFIX="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_${SEG_METHOD}"
        FAST_SEG_OUTPUT="$FAST_OUTPUT_PREFIX"_seg.nii.gz
        
        if [[ ! -f "$FAST_SEG_OUTPUT" ]]; then
            log_message "  Running FAST segmentation with FAST_OUTPUT_PREFIX = $FAST_OUTPUT_PREFIX and SEG_INPUT = $SEG_INPUT"
            fast --nopve -o "$FAST_OUTPUT_PREFIX" --verbose "$SEG_INPUT"
            
            # Verify output
            if [[ ! -f "$FAST_SEG_OUTPUT" ]]; then
                echo "ERROR: FAST segmentation failed to produce output: $FAST_SEG_OUTPUT"
                exit 1
            fi
        else
            log_message "  FAST segmentation already exists, skipping"
        fi
        SEG_OUTPUT="$FAST_SEG_OUTPUT"
        ;;
    
    synthseg_freesurfer|synthseg_github)
        SYNTHSEG_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_${SEG_METHOD}_seg.nii.gz"
        
        if [[ ! -f "$SYNTHSEG_OUTPUT" ]]; then
            log_message "  Running SynthSeg segmentation..."
            run_synthseg "$SEG_INPUT" "$SYNTHSEG_OUTPUT" "$SEG_METHOD"
        else
            log_message "  SynthSeg segmentation already exists, skipping"
        fi
        SEG_OUTPUT="$SYNTHSEG_OUTPUT"
        ;;
    
    dlicv)
        # DLICV segmentation is already done in Step 2
        SEG_OUTPUT="$BRAIN_MASK_OUTPUT"
        ;;
 esac

# =============================================================================
# STEP 4: CREATE BINARY MASK FOR TARGET ROI
# =============================================================================

log_message "Step 4: Creating binary mask for target ROI: $TARGET_ROI"

# Define label mappings for different segmentation methods
case "$SEG_METHOD" in
    fast)
        case "$TARGET_ROI" in
            csf) LABELS="1" ;;
            gray_matter) LABELS="2" ;;
            white_matter) LABELS="3" ;;
            background) LABELS="0" ;;
        esac
        ;;
    
    synthseg_freesurfer|synthseg_github)
        case "$TARGET_ROI" in
            csf) LABELS="4,5,14,15,24,43,44" ;;
            gray_matter) LABELS="3,8,10,11,12,13,16,17,18,26,28,42,47,49,50,51,52,53,54,58,60" ;;
            white_matter) LABELS="2,7,41,46" ;;
            background) LABELS="0" ;;
        esac
        ;;
    
    dlicv)
        case "$TARGET_ROI" in
            csf) LABELS="1,4,11,46,49,50,51,52" ;;
            gray_matter|white_matter|background) LABELS="0" ;;
        esac
        ;;
 esac

# Create binary mask
BINARY_MASK_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_${SEG_METHOD}_${TARGET_ROI}_mask.nii.gz"
log_message "  Creating binary mask with labels: $LABELS"

# Use 3dcalc to create binary mask
3dcalc -a "$SEG_OUTPUT" -expr "ispositive(amongst(a,$LABELS))" -prefix "$BINARY_MASK_OUTPUT"

# =============================================================================
# STEP 5: CALCULATE TISSUE VOLUMES
# =============================================================================

log_message "Step 5: Calculating tissue volumes"

VOLUME_OUTPUT="$OUTPUT_DIR/${SUBJECT_ID}_T1_LPS_${SEG_METHOD}_${TARGET_ROI}_vol.csv"

if [[ ! -f "$VOLUME_OUTPUT" ]]; then
    # Calculate volume for the target ROI
    volume=$(3dBrickStat -non-zero -volume "$BINARY_MASK_OUTPUT" 2>/dev/null || echo "0")
    
    echo "SubjectID,SegmentationMethod,TargetROI,Volume_mm3" > "$VOLUME_OUTPUT"
    echo "$SUBJECT_ID,$SEG_METHOD,$TARGET_ROI,$volume" >> "$VOLUME_OUTPUT"
    
    log_message "  Target ROI volume (mm³): $volume"
else
    log_message "  Volume calculations already exist, skipping"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

log_message "Enhanced pipeline completed successfully!"
log_message "Output files:"
log_message "  Reoriented T1: $REORIENTED_OUTPUT"

if [[ "$SHAPE" != "$DEFAULT_SHAPE" ]]; then
    log_message "  Resized T1: $RESIZED_OUTPUT"
fi

if [[ "$SKULL_STRIP" == "true" ]]; then
    log_message "  Skull-stripped T1: $SKULL_STRIPPED_OUTPUT"
fi

if [[ "$SKIP_DLICV" != "true" ]]; then
    log_message "  Brain mask: $BRAIN_MASK_OUTPUT"
    log_message "  Brain-masked T1: $BRAIN_MASKED_OUTPUT"
fi

log_message "  Segmentation: $SEG_OUTPUT"
log_message "  Binary mask: $BINARY_MASK_OUTPUT"
log_message "  Tissue volumes: $VOLUME_OUTPUT"

# Clean up temporary directory
cleanup_temp

log_message "Enhanced brain segmentation pipeline finished for subject: $SUBJECT_ID"
