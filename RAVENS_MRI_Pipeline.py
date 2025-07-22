#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !rm -rf output/*


# In[2]:


# affine = 'flirt'
affine = 'synthmorph'

# deformable = 'synthmorph_cmd'
deformable = 'synthmorph_vxm'

SHOW_IMAGES = False


# In[3]:


# Clone Guray's repo
# !rm -rf SingleScanAnalysis

# !git clone https://github.com/gurayerus/SingleScanAnalysis.git
# !cp SingleScanAnalysis/test/input/* .
# !cp SingleScanAnalysis/src/pipeline_dl/utils_mri.py .

# Skull stripped images
# !cp SingleScanAnalysis/src/pipeline_istag/RAVENS_Pipeline_Simple/input/Subj1/Subj1_T1_LPS_dlicv.nii.gz .
# !cp SingleScanAnalysis/src/pipeline_istag/RAVENS_Pipeline_Simple/templates/colin27_t1_tal_lin.nii.gz .


# In[4]:


# # Packages from GitHub.
# !pip -q install git+https://github.com/adalca/neurite.git@0776a575eadc3d10d6851a4679b7a305f3a31c65
# !pip -q install git+https://github.com/freesurfer/surfa.git@ec5ddb193fd1caf22ec654c457b5678f6bd8e460
# !pip -q install git+https://github.com/voxelmorph/voxelmorph.git@2cd706189cb5919c6335f34feec672f05203e36b


# In[5]:


# !pip install tensorflow==2.18
# !pip install keras==3.08

# get_ipython().system('pip show tensorflow')
# get_ipython().system('pip show keras')
# !conda env list


# In[6]:


# !pip -q install nilearn


# In[7]:


import numpy as np
import os
import io
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm
import matplotlib.pyplot as plt


# In[8]:


# Downloads.
# !curl -O https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5


# In[9]:


# Helper functions. The shape has to be divisible by 16.
# shape = (128, 128, 128)
shape = (256, 256, 256)

def normalize(x):
    x = np.float32(x)
    x -= x.min()
    x /= x.max()
    return x[None, ..., None]

def show(x, title=None):
    if not SHOW_IMAGES:
        return
    x = np.squeeze(x)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    i, j, k = (f // 2 for f in x.shape)

    slices = x[:, j, :], x[:, :, k], x[i, :, :].T
    for x, ax in zip(slices, axes):
        ax.imshow(x, cmap='gray')
        ax.set_axis_off()

    if title:
        axes[1].text(0.50, 1.05, title, ha='center', transform=axes[1].transAxes, size=14)


# In[10]:


# # Load, transform, and display the first high VN scan

# ###### Do we need reshape? (or even resize?)

# # high_vn_1 = sf.load_volume('/Volumes/home/mri_samples/T1/137_S_4227_2011-09-21_T1_LPS.nii.gz').resize(voxsize=2).reshape(target_shape).reorient('LPS')
# high_vn_1 = sf.load_volume('../mri_samples/T1/137_S_4227_2011-09-21_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(high_vn_1, title='Extremely High VN: 137_S_4227')

# # Load, transform, and display the second high VN scan
# high_vn_2 = sf.load_volume('../mri_samples/T1/013_S_4236_2011-10-13_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(high_vn_2, title='Extremely High VN: 013_S_4236')


# In[11]:


# # Load, transform, and display the first small VN scan
# small_vn_1 = sf.load_volume('../mri_samples/T1/019_S_6635_2020-01-28_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(small_vn_1, title='Extremely Small VN: 019_S_6635')

# # Load, transform, and display the second small VN scan
# small_vn_2 = sf.load_volume('../mri_samples/T1/072_S_4226_2012-10-12_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(small_vn_2, title='Extremely Small VN: 072_S_4226')


# In[12]:


# # Load, transform, and display the first mean VN scan
# mean_vn_1 = sf.load_volume('../mri_samples/T1/023_S_1247_2007-02-21_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(mean_vn_1, title='Mean VN: 023_S_1247')

# # Load, transform, and display the second mean VN scan
# mean_vn_2 = sf.load_volume('../mri_samples/T1/027_S_0118_2008-02-23_T1_LPS.nii.gz').resize(voxsize=2).reorient('LPS')
# show(mean_vn_2, title='Mean VN: 027_S_0118')


# In[13]:


# subj1_path = "../mri_samples/T1/023_S_1247_2007-02-21_T1_LPS.nii.gz" # Yuhan's mean VN scan
# template_path = "../mri_samples/template/BLSA_SPGR+MPRAGE_averagetemplate.nii.gz" # Yuhan's template
# template_path = "../mri_samples/T1/137_S_4227_2011-09-21_T1_LPS.nii.gz" # Yuhan's high VN scan


# In[14]:


# # Guray's Inverted Input Files and Template
# subject_nifti_paths = {
#     "subj1": "in/subj1/subj1_T1_LPS.nii.gz",
#     "subj2": "in/subj2/subj2_T1_LPS.nii.gz",
#     # Add more as needed
# }
# template_nifti_path = "template/colin27_t1_tal_lin_INV.nii.gz"


# Yuhan's Input Files and Template
subject_nifti_paths = {
    "subj1": "../mri_samples/T1/023_S_1247_2007-02-21_T1_LPS.nii.gz",
    "subj2": "../mri_samples/T1/027_S_0118_2008-02-23_T1_LPS.nii.gz",
    # "subj3": "../mri_samples/T1/072_S_4226_2012-10-12_T1_LPS.nii.gz",
    # "subj4": "../mri_samples/T1/019_S_6635_2020-01-28_T1_LPS.nii.gz",
    # "subj5": "../mri_samples/T1/013_S_4236_2011-10-13_T1_LPS.nii.gz",
    # "subj6": "../mri_samples/T1/137_S_4227_2011-09-21_T1_LPS.nii.gz",
    # Add more as needed
}
template_nifti_path = "../mri_samples/template/BLSA_SPGR+MPRAGE_averagetemplate.nii.gz"

# template_path = "out_synth/template/colin27_t1_tal_lin_INV.nii.gz"  # Guray's Inverted Template
shape = (256, 256, 256)

# Helper function to get all paths for a subject
def get_subject_paths(subj_id):
    base = f"out_synth/{subj_id}/init/"
    lin_reg = f"out_synth/{subj_id}/lin_reg/"
    def_reg = f"out_synth/{subj_id}/def_reg/"
    return {
        "t1": base + f"{subj_id}_t1.nii.gz",
        "t1_seg": base + f"{subj_id}_t1_seg.nii.gz",
        "t1_mask": base + f"{subj_id}_t1_mask.nii.gz",
        "t1_reg": lin_reg + f"{subj_id}_t1_reg.nii.gz",
        "t1_trans": lin_reg + f"{subj_id}_t1_trans.lta",
        "t1_seg_reg": lin_reg + f"{subj_id}_t1_seg_reg.nii.gz",
        "t1_def": def_reg + f"{subj_id}_t1_def.nii.gz",
        "jac_det": def_reg + f"{subj_id}_jac_det.nii.gz",
        "ravens": def_reg + f"{subj_id}_t1_RAVENS.nii.gz",
        "ravens_scaled": def_reg + f"{subj_id}_t1_RAVENS_scaled.nii.gz",
    }



# Cerebrospinal Fluid (CSF)
csf_labels = [4, 5, 14, 15, 24, 43, 44]

# Gray Matter (includes cortex and deep gray matter structures)
gray_matter_labels = [3, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 42, 47, 49, 50, 51, 52, 53, 54, 58, 60]

# White Matter
white_matter_labels = [2, 7, 41, 46]

# Background
background_label = [0]



def calculate_volume_change_from_matrix(matrix: np.ndarray) -> float:
    """
    Calculates the volume change factor from a 4x4 affine matrix.

    The factor is the absolute value of the determinant of the top-left
    3x3 submatrix (the linear transformation part).
    
    Args:
        matrix: A 4x4 NumPy array representing the affine transformation.
        
    Returns:
        The volume change factor k, where new_volume = k * old_volume.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")
    
    # Extract the top-left 3x3 linear transformation submatrix
    linear_part = matrix[0:3, 0:3]
    
    # The volume change factor is the determinant of this submatrix
    determinant = np.linalg.det(linear_part)
    
    return abs(determinant)

def parse_flirt_mat_file(filepath: str) -> np.ndarray:
    """
    Parses a simple space-delimited text file, like a FSL FLIRT .mat file.
    """
    print(f"📄 Reading FLIRT file: {filepath}")
    try:
        # np.loadtxt is perfect for simple text-based matrices
        matrix = np.loadtxt(filepath)
        return matrix
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def parse_freesurfer_lta_file(filepath: str) -> np.ndarray:
    """
    Parses a FreeSurfer LTA file to extract the 4x4 transformation matrix.
    """
    print(f"📄 Reading LTA file: {filepath}")
    matrix_lines = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # The line '1 4 4' signals that the next 4 lines are the matrix
        start_index = lines.index('1 4 4\n') + 1
        matrix_lines = lines[start_index : start_index + 4]

        # Use numpy to parse the extracted lines
        matrix = np.loadtxt(io.StringIO("".join(matrix_lines)))
        return matrix
        
    except (ValueError, IndexError) as e:
        # Handles cases where '1 4 4' is not found or file is malformed
        print(f"Error parsing {filepath}: Could not find the matrix start signal. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with {filepath}: {e}")
        return None


# In[27]:


import numpy as np
import nibabel as nib
import os

def calculate_physical_volume_change(
    flirt_matrix: np.ndarray,
    moving_image_path: str,
    ref_image_path: str
) -> float:
    """
    Calculates the true physical volume change factor from a FLIRT matrix
    by incorporating the voxel dimensions of the moving and reference images.
    
    Args:
        flirt_matrix: 4x4 NumPy array from the FLIRT .mat file.
        moving_image_path: Filepath to the original moving image (.nii or .nii.gz).
        ref_image_path: Filepath to the reference image (.nii or .nii.gz).
        
    Returns:
        The physical volume change factor k.
    """
    # 1. Calculate the determinant of the linear part of the matrix
    if flirt_matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")
    
    linear_part = flirt_matrix[0:3, 0:3]
    matrix_determinant = np.linalg.det(linear_part)
    
    # 2. Get voxel dimensions from the NIfTI headers
    try:
        moving_img = nib.load(moving_image_path)
        ref_img = nib.load(ref_image_path)
        
        # get_zooms() returns voxel dims (dx, dy, dz, ...)
        moving_vox_dims = moving_img.header.get_zooms()[:3]
        ref_vox_dims = ref_img.header.get_zooms()[:3]
        
    except FileNotFoundError as e:
        print(f"🚨 Error: Could not find image file - {e}")
        return None
    except Exception as e:
        print(f"🚨 Error loading NIfTI files: {e}")
        return None

    # 3. Calculate the physical volume of a single voxel for each image
    moving_voxel_volume = np.prod(moving_vox_dims)
    ref_voxel_volume = np.prod(ref_vox_dims)
    
    # 4. Calculate the voxel volume ratio
    if moving_voxel_volume == 0:
        raise ValueError("Moving image voxel volume cannot be zero.")
    voxel_volume_ratio = ref_voxel_volume / moving_voxel_volume
    
    # 5. Compute the final physical volume change factor
    k_physical = abs(matrix_determinant) * voxel_volume_ratio
    
    print("--- Calculation Details ---")
    print(f"Matrix Determinant    : {matrix_determinant:.4f}")
    print(f"Moving Voxel Volume   : {moving_voxel_volume:.4f} mm³")
    print(f"Reference Voxel Volume: {ref_voxel_volume:.4f} mm³")
    print(f"Voxel Volume Ratio    : {voxel_volume_ratio:.4f}")
    print("-------------------------")
    
    return k_physical




# ## Validate Scale Factor Value from the actual data

# In[29]:


import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Union

def calculate_nifti_volume(filepath: Union[str, Path], verbose: bool = False) -> float:
    """
    Calculates the total volume of non-zero voxels in a NIfTI file.

    This function loads a NIfTI file, counts the number of voxels with a
    value other than zero, determines the volume of a single voxel from the
    file's header, and computes the total volume.

    Args:
        filepath (str | Path): The full path to the input NIfTI file 
                               (e.g., 'data/subject_01.nii.gz').
        verbose (bool): If True, prints the voxel count, volume per voxel,
                        and total volume to the console. Defaults to False.

    Returns:
        float: The total volume of the non-zero voxels, typically in mm³.
    
    Raises:
        FileNotFoundError: If the file specified by filepath does not exist.
        Exception: Catches and re-raises other potential errors from nibabel.
    """
    try:
        # Ensure the filepath is a Path object for robust handling
        nii_path = Path(filepath)
        if not nii_path.exists():
            raise FileNotFoundError(f"Error: The file was not found at {filepath}")

        # Load the NIfTI file
        nii_file = nib.load(nii_path)

        # Get the image data as a NumPy array
        data = nii_file.get_fdata()

        # 1. Count the number of non-zero voxels
        voxel_count = np.count_nonzero(data)

        # 2. Get the volume of a single voxel from the header's zoom info
        # The first three zoom values correspond to the dimensions (x, y, z)
        voxel_volume = np.prod(nii_file.header.get_zooms()[:3])

        # 3. Calculate the total volume
        total_volume = voxel_count * voxel_volume

        if verbose:
            print(f"--- Volume Calculation Details ---")
            print(f"File: {nii_path.name}")
            print(f"Number of non-zero voxels: {voxel_count}")
            print(f"Volume per voxel: {voxel_volume:.4f} mm³")
            print(f"Total volume of structure: {total_volume:.4f} mm³")
            print(f"------------------------------------")

        return total_volume

    except FileNotFoundError as e:
        print(e)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise



# In[30]:


# Calculate Volume 1
# volume_1 = calculate_nifti_volume('out_synth/subj1/lin_reg/subj1_t1_seg_reg.nii.gz')
# print(f"\nCalculated Volume 1: {volume_1:.4f} mm³")
# # Calculate Volume 2
# volume_2 = calculate_nifti_volume('out_synth/subj1/init/subj1_t1_mask.nii.gz')
# print(f"\nCalculated Volume 2: {volume_2:.4f} mm³")

# # Calculate the scale factor by dividing the volumes
# actual_scale_factor = volume_1 / volume_2
# print(f"\nCalculated Scale Factor: {actual_scale_factor:.4f} mm³")


# # Deformable Registration

# In[31]:


# Shapes model. Assumes affine initialization and may require fine tuning.

print(f"The shape variable is: {shape}, and its type is: {type(shape)}")


model = vxm.networks.VxmDense(
    nb_unet_features=([256] * 4, [256] * 6),
    int_steps=5,
    int_resolution=2,
    svf_resolution=2,
    inshape=shape,
  )
model = tf.keras.Model(model.inputs, model.references.pos_flow)
model.load_weights('shapes-dice-vel-3-res-8-16-32-256f.h5')




import pandas as pd
import json
import os
import argparse
import nibabel as nib
import numpy as np



for subj_id in subject_nifti_paths.keys():
    print(f"\n===== Processing subject: {subj_id} =====\n")
    paths = get_subject_paths(subj_id)
    subj1_path = paths["t1"]
    out_seg = paths["t1_seg"]
    output_filename = paths["t1_mask"]
    affine_moved = paths["t1_reg"]
    matrix_filepath = paths["t1_trans"]
    seg_affine_moved = paths["t1_seg_reg"]
    def_moved = paths["t1_def"]
    jac_det_path = paths["jac_det"]
    ravens_path = paths["ravens"]
    ravens_scaled_path = paths["ravens_scaled"]

    # --- Input original NIfTI paths and preprocess (resize/reshape/resample) ---
    orig_subj_nii = subject_nifti_paths[subj_id]
    orig_template_nii = template_nifti_path
    preproc_subj_nii = f"out_synth/{subj_id}/init/{subj_id}_t1.nii.gz"
    preproc_template_nii = "out_synth/template/colin27_t1_tal_lin_INV.nii.gz"

    # Preprocess subject image
    if not os.path.exists(preproc_subj_nii):
        print(f"Preprocessing subject image for {subj_id} ...")
        # subj_vol = sf.load_volume(orig_subj_nii).resize(voxsize=2).reshape(shape).reorient('LPS')
        subj_vol = sf.load_volume(orig_subj_nii).reorient('LPS')
        os.makedirs(os.path.dirname(preproc_subj_nii), exist_ok=True)
        subj_vol.save(preproc_subj_nii)
    else:
        print(f"Preprocessed subject image exists: {preproc_subj_nii}")

    # Preprocess template image
    if not os.path.exists(preproc_template_nii):
        print("Preprocessing template image ...")
        # template_vol = sf.load_volume(orig_template_nii).resize(voxsize=2).reshape(shape).reorient('LPS')
        template_vol = sf.load_volume(orig_template_nii).reorient('LPS')
        os.makedirs(os.path.dirname(preproc_template_nii), exist_ok=True)
        template_vol.save(preproc_template_nii)
    else:
        print(f"Preprocessed template image exists: {preproc_template_nii}")

    # Update paths for downstream steps
    subj1_path = preproc_subj_nii
    template_path = preproc_template_nii

    # --- Segmentation with SynthSeg (template and subject) ---
    import utils_mri as utilmri
    NUMTHD = 4
    # For each subject, segment both the template and the subject's T1 image
    seg_targets = [
        (f"template", template_path, f"out_synth/template/"),
        (subj_id, subj1_path, f"out_synth/{subj_id}/init/")
    ]
    for cur_id, cur_img, out_base in seg_targets:
        # # Reorient images
        # print(f'Reorienting image for {cur_id} ...')
        # out_reorient = os.path.join(out_base, f'{cur_id}_T1_LPS.nii.gz')
        # if not os.path.exists(out_reorient):
        #     if not os.path.exists(out_base):
        #         os.makedirs(out_base)
        #     utilmri.reorient_img(cur_img, 'LPS', out_reorient)
        # else:
        #     print(f'Out file exists, skip: {out_reorient}')

        # Segment image
        print(f'Segmenting image for {cur_id} ...')
        out_seg = os.path.join(out_base, f'{cur_id}_t1_seg.nii.gz')
        out_qc = os.path.join(out_base, f'{cur_id}_t1_Seg_QC.csv')
        out_vol = os.path.join(out_base, f'{cur_id}_t1_Seg_Vol.csv')
        out_post = os.path.join(out_base, f'{cur_id}_t1_Seg_Post.nii.gz')
        out_resample = os.path.join(out_base, f'{cur_id}_t1_Seg_Resample.nii.gz')
        cmd = f'mri_synthseg --i {cur_img} --o {out_seg} --robust --vol {out_vol} --qc {out_qc} --resample {out_resample} --threads {NUMTHD} --cpu'
        if not os.path.exists(out_seg):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {out_seg}')

    # Now use the subject's segmentation output for downstream steps
    out_seg = os.path.join(f"out_synth/{subj_id}/init/", f"{subj_id}_t1_seg.nii.gz")
    t1_seg = sf.load_volume(out_seg).reshape(shape).reorient('LPS')
    if SHOW_IMAGES:
        show(t1_seg, title=f'Segmented T1-weighted MRI ({subj_id})')





    # --- Create Binary Mask ---
    import nibabel as nib
    # target_labels = [1]
    target_labels = csf_labels
    seg_data = t1_seg.data
    mask_data = np.isin(seg_data, target_labels).astype(np.uint8)
    affine_matrix = np.asarray(t1_seg.geom.vox2world)
    csf_mask_nii = nib.Nifti1Image(mask_data, affine_matrix)
    nib.save(csf_mask_nii, output_filename)
    print(f"Binary mask saved to: {output_filename}")
    temp1 = sf.load_volume(output_filename).reshape(shape).reorient('LPS')
    if SHOW_IMAGES:
        show(temp1, title=f'Binary mask ({subj_id})')

    # --- Affine Registration ---
    affine_moving = subj1_path
    affine_fixed = template_path
    if affine == 'synthmorph':
        # Ensure output directory exists for the transform file
        os.makedirs(os.path.dirname(matrix_filepath), exist_ok=True)
        # Estimate and save an affine transform trans.lta in FreeSurfer LTA format
        cmd1 = f'mri_synthmorph register -m affine -t {matrix_filepath} {affine_moving} {affine_fixed}'
        # Apply an existing transform to an image
        cmd2 = f'mri_synthmorph apply {matrix_filepath} {affine_moving} {affine_moved}'
        if not os.path.exists(affine_moved):
            print(f'About to run: {cmd1}')
            os.system(cmd1)
            print(f'About to run: {cmd2}')
            os.system(cmd2)
        else:
            print(f'Out file exists, skip: {affine_moved}')
    elif affine == 'flirt':
        matrix_filepath = paths["t1_trans"].replace('.lta', '.mat')
        cmd1 = f'bet {affine_moving} output/stripped.nii.gz'
        cmd2 = f'flirt -in output/stripped.nii.gz -ref {affine_fixed} -out {affine_moved} -omat {matrix_filepath} -dof 12'
        if not os.path.exists(affine_moved):
            print(f'About to run: {cmd1}')
            os.system(cmd1)
            os.system(cmd2)
        else:
            print(f'Out file exists, skip: {affine_moved}')

    # --- Apply the affine step to the segmentation ---
    seg_affine_moving = output_filename
    if affine == 'synthmorph':
        cmd = f'mri_synthmorph apply -m nearest {matrix_filepath} {seg_affine_moving} {seg_affine_moved}'
        if not os.path.exists(seg_affine_moved):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {seg_affine_moved}')
    elif affine == 'flirt':
        cmd = f'flirt -in {seg_affine_moving} -ref {affine_fixed} -out {seg_affine_moved} -init {matrix_filepath} -applyxfm'
        if not os.path.exists(seg_affine_moved):
            print(f'About to run: {cmd}')
            os.system(cmd)
        else:
            print(f'Out file exists, skip: {seg_affine_moved}')

    # --- Calculate Scale Factor for the Affine Registration ---
    if affine == "synthmorph":
        lta_matrix = parse_freesurfer_lta_file(matrix_filepath)
        if lta_matrix is not None:
            scale_factor = calculate_volume_change_from_matrix(lta_matrix)
            print(f"📈 SynthMorph Volume Change Factor (k): {scale_factor:.4f}\n")
    elif affine == 'flirt':
        flirt_matrix = parse_flirt_mat_file(matrix_filepath)
        if flirt_matrix is not None:
            # scale_factor = calculate_volume_change_from_matrix(flirt_matrix)
            # print(f"📈 FLIRT Volume Change Factor (k): {scale_factor:.4f}\n")

            scale_factor = calculate_physical_volume_change(flirt_matrix, subj1_path, template_path)
            print(f"📈 FLIRT Volume Change Factor (k): {scale_factor:.4f}\n")


    # --- Calculate Volumes for Validation ---
    volume_1 = calculate_nifti_volume(seg_affine_moved)
    print(f"\nCalculated Volume 1: {volume_1:.4f} mm³")
    volume_2 = calculate_nifti_volume(output_filename)
    print(f"\nCalculated Volume 2: {volume_2:.4f} mm³")
    actual_scale_factor = volume_1 / volume_2
    print(f"\nCalculated Scale Factor: {actual_scale_factor:.4f} mm³")

    # --- Deformable Registration ---
    t1_fixed = sf.load_volume(affine_fixed).reshape(shape).reorient('LPS')
    t1_moving = sf.load_volume(affine_moved).resample_like(t1_fixed)
    if SHOW_IMAGES:
        show(t1_fixed, title=f'Fixed T1-weighted MRI ({subj_id})')
        show(t1_moving, title=f'Moving T1-weighted MRI ({subj_id})')
    moving = normalize(t1_moving)
    fixed = normalize(t1_fixed)
    trans = model.predict((moving, fixed))
    moved = vxm.layers.SpatialTransformer(fill_value=0)((moving, trans))
    if SHOW_IMAGES:
        show(t1_moving, title=f'Moving T1-weighted MRI ({subj_id})')
        show(moved, title=f'Moved T1-weighted MRI ({subj_id})')
        show(fixed, title=f'Fixed T1-weighted MRI ({subj_id})')
        show(moved - fixed, title=f'Difference after registration 1 ({subj_id})')

    # --- Deformable Registration of Segmented Image ---
    # Define the template segmentation path based on the segmentation step
    template_seg_path = os.path.join("out_synth/template/", "template_t1_seg.nii.gz")
    t1_fixed_seg = sf.load_volume(template_seg_path).reshape(shape).reorient('LPS')
    t1_moving_seg = sf.load_volume(seg_affine_moved).resample_like(t1_fixed_seg)
    before = normalize(t1_moving_seg) - normalize(t1_fixed_seg)
    if SHOW_IMAGES:
        show(t1_moving_seg, title=f'Moving Registered T1-weighted MRI ({subj_id})')
        show(t1_fixed_seg, title=f'Fixed Registered T1-weighted MRI ({subj_id})')
        show(before, title=f'Difference before Segmented registration ({subj_id})')

    # --- Apply Deformation Field to Segmented Image ---
    moving_seg = normalize(t1_moving_seg)
    fixed_seg = normalize(t1_fixed_seg)
    moved_seg = vxm.layers.SpatialTransformer(interp_method='nearest', fill_value=0)((moving_seg, trans))
    if SHOW_IMAGES:
        show(moved_seg, title=f'Moved T1-weighted MRI ({subj_id})')
        show(fixed_seg, title=f'Fixed T1-weighted MRI ({subj_id})')
        show(moved_seg - fixed_seg, title=f'Difference after Segmented registration ({subj_id})')
    os.makedirs(os.path.dirname(def_moved), exist_ok=True)
    t1_fixed.new(moved_seg[0]).save(def_moved)

    # --- Calculate Jacobian Matrix of the Deformation ---
    jacobian_det = vxm.py.utils.jacobian_determinant(trans[0])
    if SHOW_IMAGES:
        show(jacobian_det, title=f'Jacobian Determinant of the Transformation ({subj_id})')
    t1_fixed.new(jacobian_det).save(jac_det_path)

    # --- Calculate Ravens Map ---
    def calc_ravens(f_jac, f_seg, labels, f_out):
        nii_jac = nib.load(f_jac)
        img_jac = nii_jac.get_fdata()
        nii_seg = nib.load(f_seg)
        nii_seg_data = nii_seg.get_fdata()
        ravens = img_jac * nii_seg_data
        nii_out = nib.Nifti1Image(ravens, nii_jac.affine)
        nib.save(nii_out, f_out)
        print(f"RAVENS map saved to: {f_out}")
        ravens /= scale_factor
        nii_out = nib.Nifti1Image(ravens, nii_jac.affine)
        nib.save(nii_out, ravens_scaled_path)
        print(f"RAVENS map saved to: {ravens_scaled_path}")

    calc_ravens(jac_det_path, def_moved, target_labels, ravens_path)

    # --- Show RAVENS Maps ---
    t1_rav = sf.load_volume(ravens_path)
    if SHOW_IMAGES:
        show(t1_rav, title=f'RAVENS Map ({subj_id})')
        t1_rav = sf.load_volume(ravens_scaled_path)
        show(t1_rav, title=f'RAVENS Map Scaled ({subj_id})')

    # --- Print Sums for Debugging ---
    print(f"The volume of the original mask is: {volume_2}")

    nii_file = nib.load(ravens_scaled_path)
    data = nii_file.get_fdata()
    total_sum = np.sum(data)
    print(f"The sum of the values in RAVENS after scaling is: {total_sum}")

    nii_file = nib.load(ravens_path)
    data = nii_file.get_fdata()
    total_sum = np.sum(data)
    print(f"The sum of the values in temp RAVENS is: {total_sum}")

    nii_file = nib.load(out_seg)
    data = nii_file.get_fdata()
    voxel_count = np.count_nonzero(np.isin(data, target_labels))
    print(f"The number of voxels with a value of X in the Original Image is: {voxel_count}")
    nii_file = nib.load(output_filename)
    data = nii_file.get_fdata()
    total_sum = np.sum(data)
    print(f"The sum of the values in Original MASK is: {total_sum}")
    nii_file = nib.load(output_filename)
    data = nii_file.get_fdata()
    voxel_count = np.count_nonzero(data)
    print(f"The count of the values in Original mask is: {voxel_count}")

