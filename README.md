# mri_pipeline

## Overview
This repository contains an MRI processing pipeline that performs brain image registration and RAVENS (Regional Analysis of Volumes Examined in Normalized Space) analysis using SynthMorph and VoxelMorph.

## Main Pipeline: synthmorph_ravens_pipeline.py

### Purpose
The `synthmorph_ravens_pipeline.py` script implements a comprehensive MRI processing pipeline that:
- Performs brain image segmentation using SynthSeg
- Conducts affine and deformable registration using SynthMorph/VoxelMorph
- Calculates Jacobian determinants for volume change analysis
- Generates RAVENS maps for regional volume analysis

### Key Components

#### 1. **Image Preprocessing**
- Resizes and reorients input T1-weighted MRI images to LPS orientation
- Supports skull stripping (optional) using ANTs
- Preprocesses both subject and template images

#### 2. **Segmentation**
- Uses SynthSeg for automatic brain tissue segmentation
- Supports multiple segmentation backends:
  - `synthseg_freesurfer`: FreeSurfer SynthSeg installation
  - `synthseg_github`: GitHub SynthSeg (currently in progress)
  - `fast`: FSL FAST segmentation
- Generates segmentation masks and quality control reports

#### 3. **Affine Registration**
- Performs linear registration between subject and template images
- Supports multiple registration methods:
  - `synthmorph_freesurfer`: SynthMorph affine registration
  - `itk`: ITK-based affine registration using SimpleITK
  - `flirt`: FSL FLIRT registration (currently in progress)
- Calculates volume change factors from transformation matrices

#### 4. **Deformable Registration**
- Performs non-linear registration using SynthMorph
- Supports two approaches:
  - `synthmorph_freesurfer`: FreeSurfer SynthMorph command-line interface
  - `synthmorph_voxelmorph`: VoxelMorph-based implementation
- Generates deformation fields and Jacobian determinants

#### 5. **RAVENS Analysis**
- Calculates RAVENS maps by multiplying Jacobian determinants with segmentation masks
- Applies volume correction factors from affine registration
- Focuses on CSF (Cerebrospinal Fluid) analysis by default


