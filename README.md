# DL_RAVENS

## Overview
This repository contains an end‑to‑end MRI processing pipeline for brain T1 images. It performs preprocessing, brain masking and tissue segmentation, registration, and RAVENS (Regional Analysis of Volumes Examined in Normalized Space) analysis using either an ANTs-based workflow or a SynthMorph/VoxelMorph workflow.

The entrypoint is the Slurm job wrapper `pipeline_job.sh`, which orchestrates all steps and submits the necessary sub‑jobs.

## Entrypoint
`pipeline_job.sh` (Slurm-based; runs other scripts under the hood)

### Usage
```
./pipeline_job.sh \
  --pipeline ants|synthmorph \
  --template <template_t1.nii.gz> \
  --subject <subject_t1.nii.gz> \
  --outdir <output_dir> \
  [--seg-method fast|synthseg_freesurfer|synthseg_github|dlicv] \
  [--target-roi csf|gray_matter|white_matter|background] \
  [--shape 256,256,256] \
  [--voxel-size 1] \
  [--force-cpu] \
  [--device cpu|gpu]
```

### Quick start
- CPU SynthMorph example:
```
./pipeline_job.sh \
  --pipeline synthmorph \
  --template /path/to/template_T1.nii.gz \
  --subject  /path/to/subject_T1.nii.gz \
  --outdir   /path/to/output_dir \
  --seg-method fast \
  --target-roi csf \
  --device cpu
```

- GPU SynthMorph example (A100 config provided in job script):
```
./pipeline_job.sh \
  --pipeline synthmorph \
  --template /path/to/template_T1.nii.gz \
  --subject  /path/to/subject_T1.nii.gz \
  --outdir   /path/to/output_dir \
  --seg-method synthseg_freesurfer \
  --target-roi csf \
  --device gpu
```

- ANTs + GenerateRAVENS example:
```
./pipeline_job.sh \
  --pipeline ants \
  --template /path/to/template_T1.nii.gz \
  --subject  /path/to/subject_T1.nii.gz \
  --outdir   /path/to/output_dir \
  --seg-method dlicv \
  --target-roi csf
```

### Arguments
- `--pipeline` (required): choose `ants` for the ANTs/GenerateRAVENS path or `synthmorph` for the SynthMorph/VoxelMorph path.
- `--template`, `--subject`, `--outdir` (required): input T1s and output directory.
- `--seg-method` (optional): `fast` (FSL), `synthseg_freesurfer` (FreeSurfer SynthSeg), `synthseg_github` (Python SynthSeg), or `dlicv`.
- `--target-roi` (optional): `csf`, `gray_matter`, `white_matter`, or `background` for ROI mask and volume stats.
- `--shape` (optional): target shape for early reshape/regrid, e.g. `256,256,256`.
- `--voxel-size` (optional): isotropic voxel size in mm (default 1.0) for early resample.
- `--force-cpu` (optional): force CPU for steps that support GPU (e.g., DLICV inside Singularity).
- `--device` (optional): `cpu` or `gpu` for the SynthMorph branch. Ignored by `ants`.

### What the wrapper does
1. Early resample/regrid using `dlravens_preprocessing.py`
   - Reorients to LPS, resamples to the provided voxel size, and reshapes to the target shape for both subject and template.
   - Emits `<basename>_reshaped.nii.gz` in `--outdir` and passes those downstream.
2. Preprocessing + segmentation via Slurm (`preprocessing_segmentation_pipeline.sh`)
   - Reorient, (optional) skull strip, DLICV brain mask (unless `--no-dlicv` is used inside the script), run segmentation (`fast`, `synthseg_*`, or `dlicv`), create ROI mask, and compute ROI volumes.
3. Branch by `--pipeline`:
   - `ants`: calls `run_calc_csf_ravens.sh`, which submits `GenerateRAVENS.sh` (external) to compute RAVENS with ANTs.
   - `synthmorph`: submits either `synthmorph_a100_job.sh` (GPU) or `synthmorph_cpu_job.sh` (CPU), which in turn run `synthmorph_ravens_pipeline.py` to perform deformable registration, Jacobians, and RAVENS.

Logs are written under `logs/` with Slurm job IDs.

## Outputs
Given `SUBJECT_ID` derived from the subject filename (suffixes like `_T1`, `_T1_LPS`, `_reshaped` stripped), the preprocessing stage writes (examples):
- Reoriented T1: `<outdir>/<SUBJECT_ID>_T1_LPS.nii.gz`
- Brain mask: `<outdir>/<SUBJECT_ID>_T1_LPS_dlicvmask.nii.gz` (if DLICV used)
- Brain‑masked T1: `<outdir>/<SUBJECT_ID>_T1_LPS_dlicv.nii.gz`
- Segmentation: `<outdir>/<SUBJECT_ID>_T1_LPS_<seg-method>_seg.nii.gz`
- ROI mask: `<outdir>/<SUBJECT_ID>_T1_LPS_<seg-method>_<target-roi>_mask.nii.gz`
- ROI volume CSV: `<outdir>/<SUBJECT_ID>_T1_LPS_<seg-method>_<target-roi>_vol.csv`

Pipeline‑specific outputs:
- ANTs branch: results are organized under `<outdir>/ants/<SUBJECT_ID>/` and include the ANTs/GenerateRAVENS outputs for the selected label/ROI.
- SynthMorph branch: results are organized under `<outdir>/synthmorph/` (set via `OUTPUT_DIR` in the wrapper) and include registration products (warps, Jacobians) and RAVENS maps from `synthmorph_ravens_pipeline.py`.

## Dependencies
This project mixes Python tools with external neuroimaging toolkits. Minimum requirements:

- Python packages (see `requirements.txt`). Notably: TensorFlow/Keras, SimpleITK, nibabel, nilearn.
- Git packages (installed separately; see Dockerfile or install manually):
  - `surfa` (required by `dlravens_preprocessing.py`)
  - `voxelmorph` and `neurite` (used by SynthMorph/VoxelMorph)
- External tools:
  - AFNI and FSL (required by `preprocessing_segmentation_pipeline.sh`)
  - FreeSurfer (for `synthseg_freesurfer`)
  - Singularity and DLICV models (for `dlicv` option)
  - Slurm (`sbatch`) for job submission
  - CUDA/cuDNN (GPU SynthMorph)

The wrapper sets `FSLOUTPUTTYPE=NIFTI_GZ`. Some cluster environments may need modules loaded (e.g., `module load freesurfer/8.1.0`). The SynthMorph weights file is downloaded automatically at runtime via `curl` the first time `pipeline_job.sh` runs.

## Environment setup (example)
Using conda:
```
conda create -n mri python=3.10 -y
conda activate mri
pip install -r requirements.txt
# Install Git deps (pin revisions as needed)
pip install git+https://github.com/freesurfer/surfa.git
pip install git+https://github.com/adalca/neurite.git
pip install git+https://github.com/voxelmorph/voxelmorph.git
```

For FreeSurfer SynthSeg support, ensure FreeSurfer is installed and sourced so `mri_synthseg` is on `PATH`.

For DLICV, set paths at the top of `preprocessing_segmentation_pipeline.sh`:
- `DLICV_MODELS`: directory containing the DLICV models
- `DLICV_CONTAINER`: path to the Singularity image

## Components (for reference)
- `pipeline_job.sh`: Orchestrator and Slurm submission wrapper (entrypoint)
- `dlravens_preprocessing.py`: Early resample/regrid of subject and template using `surfa`
- `preprocessing_segmentation_pipeline.sh`: Reorientation, (optional) skull stripping, DLICV masking, segmentation, ROI mask and volume CSV
- `run_calc_csf_ravens.sh`: Submits `GenerateRAVENS.sh` (external) for ANTs workflow
- `synthmorph_cpu_job.sh` / `synthmorph_a100_job.sh`: Slurm job scripts that call
  `synthmorph_ravens_pipeline.py` (registration, Jacobians, RAVENS) on CPU/GPU

## Troubleshooting
- Missing `surfa` import when running `dlravens_preprocessing.py`:
  install via `pip install git+https://github.com/freesurfer/surfa.git`.
- AFNI/FSL not found: install and ensure binaries are on `PATH`.
- FreeSurfer SynthSeg errors: verify FreeSurfer is sourced; optionally pass a prefix via `--freesurfer-prefix` to `preprocessing_segmentation_pipeline.sh`.
- Singularity not found or DLICV paths invalid: install Singularity or run with `--seg-method` other than `dlicv`.
- GPU errors on A100: the GPU job script sets TensorFlow/CUDA environment variables for stability; fall back to `--device cpu` if needed.

## Notes
- All heavy work is executed via Slurm (`sbatch`). Running without Slurm is not supported by the wrapper as written.
- File naming deliberately preserves your original subject ID (common suffixes are stripped) for consistent downstream outputs.

