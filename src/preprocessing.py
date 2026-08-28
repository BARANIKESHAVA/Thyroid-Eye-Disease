"""
MRI Preprocessing Module
=========================
Preprocessing for TOM500 orbital MRI volumes:
- NIfTI loading, orientation verification, spacing extraction
- Intensity normalization (z-score on foreground)
- Optional resampling, denoising, bias-field correction
- Foreground cropping
- Saving preprocessed outputs

Author: TED EOM Research Group
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Dict, Any, List


def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI volume. Returns (data, affine, header)."""
    img = nib.load(filepath)
    return img.get_fdata().astype(np.float32), img.affine.copy(), img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: str,
               header: Optional[nib.Nifti1Header] = None) -> None:
    """Save array as NIfTI."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, filepath)


def get_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    """Extract voxel spacing from affine matrix."""
    return tuple(np.abs(np.diag(affine)[:3]).tolist())


def normalize_intensity(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize intensity on non-zero foreground voxels."""
    if method == "zscore":
        mask = data > 0
        if mask.any():
            mu, sigma = data[mask].mean(), data[mask].std()
            if sigma > 0:
                data = (data - mu) / sigma
    elif method == "minmax":
        mn, mx = data.min(), data.max()
        if mx - mn > 0:
            data = (data - mn) / (mx - mn)
    elif method == "percentile":
        lo, hi = np.percentile(data, [0.5, 99.5])
        data = np.clip(data, lo, hi)
        if hi - lo > 0:
            data = (data - lo) / (hi - lo)
    return data.astype(np.float32)


def resample_volume(data: np.ndarray, affine: np.ndarray,
                    target_spacing: Tuple[float, float, float],
                    interpolation: str = "linear"
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to target spacing using SimpleITK."""
    sitk_img = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    orig_spacing = get_spacing(affine)
    sitk_img.SetSpacing(orig_spacing)

    orig_size = np.array(sitk_img.GetSize())
    new_size = np.round(orig_size * np.array(orig_spacing) / np.array(target_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)

    out = resampler.Execute(sitk_img)
    out_data = sitk.GetArrayFromImage(out).transpose(2, 1, 0).astype(np.float32)

    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * target_spacing[i]

    return out_data, new_affine


def reorient_to_ras(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient to RAS."""
    img = nib.Nifti1Image(data, affine)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata().astype(np.float32), canonical.affine


def denoise_gaussian(data: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    return gaussian_filter(data, sigma=sigma).astype(np.float32)


def preprocess_volume(filepath: str, output_dir: str,
                      target_spacing: Optional[Tuple[float, float, float]] = None,
                      normalize_method: str = "zscore",
                      denoise: bool = False,
                      is_label: bool = False) -> Dict[str, Any]:
    """
    Preprocess a single MRI volume.
    For labels: nearest-neighbor resampling, no intensity ops.
    """
    filename = os.path.basename(filepath)
    data, affine, header = load_nifti(filepath)
    original_shape = data.shape
    original_spacing = get_spacing(affine)

    # Reorient
    data, affine = reorient_to_ras(data, affine)

    # Denoise (images only)
    if denoise and not is_label:
        data = denoise_gaussian(data, sigma=0.5)

    # Resample if requested
    if target_spacing is not None:
        interp = "nearest" if is_label else "linear"
        data, affine = resample_volume(data, affine, target_spacing, interp)

    # Normalize (images only)
    if not is_label:
        data = normalize_intensity(data, method=normalize_method)

    # Save
    output_path = os.path.join(output_dir, filename)
    save_nifti(data, affine, output_path)

    return {
        "filename": filename,
        "original_shape": original_shape,
        "original_spacing": original_spacing,
        "preprocessed_shape": data.shape,
        "new_spacing": get_spacing(affine),
    }


def preprocess_dataset(image_dir: str, label_dir: str,
                       out_image_dir: str, out_label_dir: str,
                       target_spacing: Optional[Tuple[float, float, float]] = None,
                       max_cases: Optional[int] = None) -> List[Dict]:
    """Preprocess all image-label pairs."""
    print("\n" + "=" * 60)
    print("  MRI Preprocessing")
    print("=" * 60)

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))])

    if max_cases:
        images = images[:max_cases]
        labels = labels[:max_cases]

    results = []
    for i, img_file in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_file}")
        info = preprocess_volume(
            os.path.join(image_dir, img_file), out_image_dir,
            target_spacing=target_spacing, is_label=False
        )
        results.append(info)

        # Find matching label
        from src.dataset_manager import extract_case_id
        case_id = extract_case_id(img_file)
        lbl_match = [l for l in labels if extract_case_id(l) == case_id]
        if lbl_match:
            preprocess_volume(
                os.path.join(label_dir, lbl_match[0]), out_label_dir,
                target_spacing=target_spacing, is_label=True
            )

    print(f"  * Preprocessed {len(results)} cases\n")
    return results
