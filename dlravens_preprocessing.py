"""
Early resample/regrid utility for subject and template volumes.

Resamples to requested isotropic voxel size, reorients to LPS, and reshapes to a
target shape compatible with downstream registration.

Outputs are written into --outdir with suffix "_reshaped.nii.gz" for both
subject and template, preserving the original basenames.
"""

import argparse
import os
import sys

try:
    import surfa as sf
except Exception as exc:
    sys.stderr.write(
        f"ERROR: Failed to import 'surfa'. Install it first (e.g., pip install git+https://github.com/freesurfer/surfa.git)\n{exc}\n"
    )
    sys.exit(1)


def parse_shape(shape_arg: str) -> tuple[int, int, int]:
    # Accept "256" or "256,256,256"
    s = shape_arg.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError("--shape must be 'X,Y,Z' or a single integer for cubic shape")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    v = int(s)
    return (v, v, v)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_suffix(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        stem = base[:-7]
    elif base.endswith(".nii"):
        stem = base[:-4]
    else:
        stem = base
    return f"{stem}{suffix}.nii.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resample/regrid subject and template volumes")
    parser.add_argument("--subject", required=True, help="Path to subject T1 .nii.gz")
    parser.add_argument("--template", required=True, help="Path to template T1 .nii.gz")
    parser.add_argument("--outdir", required=True, help="Output directory for reshaped volumes")
    parser.add_argument("--shape", default="256,256,256", help="Target shape (X,Y,Z) or single int, default 256,256,256")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Target isotropic voxel size in mm, default 1.0")
    parser.add_argument("--orient", default="LPS", help="Target orientation (default: LPS)")

    args = parser.parse_args()

    subject_path = args.subject
    template_path = args.template
    outdir = args.outdir
    target_shape = parse_shape(args.shape)
    target_vox = float(args.voxel_size)
    target_orient = args.orient

    if not os.path.isfile(subject_path):
        sys.stderr.write(f"ERROR: Subject file not found: {subject_path}\n")
        return 2
    if not os.path.isfile(template_path):
        sys.stderr.write(f"ERROR: Template file not found: {template_path}\n")
        return 2

    ensure_outdir(outdir)

    # Compute output file paths in outdir
    suffix = "_reshaped"
    subj_out = os.path.join(outdir, add_suffix(subject_path, suffix))
    temp_out = os.path.join(outdir, add_suffix(template_path, suffix))

    # Process template
    temp_vol = (
        sf.load_volume(template_path)
        .reorient(target_orient)
        .resize(voxsize=target_vox)
        .reshape(target_shape)
    )
    # Ensure overwrite behavior across surfa versions
    if os.path.exists(temp_out):
        os.remove(temp_out)
    temp_vol.save(temp_out)

    # Process subject
    subj_vol = (
        sf.load_volume(subject_path)
        .reorient(target_orient)
        .resize(voxsize=target_vox)
        .reshape(target_shape)
    )
    # Ensure overwrite behavior across surfa versions
    if os.path.exists(subj_out):
        os.remove(subj_out)
    subj_vol.save(subj_out)

    # Emit paths for potential upstream parsing
    print(f"TEMPLATE_OUT={temp_out}")
    print(f"SUBJECT_OUT={subj_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
