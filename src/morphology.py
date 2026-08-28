"""
Morphological Analysis Module
===============================
Volume, Cross-Sectional Area, and Thickness extraction from EOM masks.

Thickness method: Euclidean distance transform.
  - Each interior voxel gets the distance to the nearest boundary.
  - max_thickness = 2 * max(distance_transform)  (inscribed sphere diameter)
  - mean_thickness = 2 * mean(distance_transform inside mask)

Author: TED EOM Research Group
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Optional, Tuple


def compute_volume(mask: np.ndarray, spacing: Tuple[float, ...]) -> Dict[str, float]:
    """Volume = voxel_count * voxel_volume."""
    voxel_vol = float(np.prod(spacing))
    n = int((mask > 0).sum())
    vol_mm3 = n * voxel_vol
    return {"volume_mm3": round(vol_mm3, 2), "volume_cm3": round(vol_mm3 / 1000.0, 4)}


def compute_csa(mask: np.ndarray, spacing: Tuple[float, ...],
                axis: int = 2) -> Dict[str, float]:
    """
    Cross-sectional area per slice along `axis`.
    Returns mean and max CSA in mm^2.
    """
    dims = list(range(3))
    dims.remove(axis)
    pixel_area = float(spacing[dims[0]] * spacing[dims[1]])

    areas = []
    for s in range(mask.shape[axis]):
        sl = [slice(None)] * 3
        sl[axis] = s
        a = float((mask[tuple(sl)] > 0).sum()) * pixel_area
        if a > 0:
            areas.append(a)

    if not areas:
        return {"mean_csa_mm2": 0.0, "max_csa_mm2": 0.0}
    return {
        "mean_csa_mm2": round(float(np.mean(areas)), 2),
        "max_csa_mm2": round(float(np.max(areas)), 2),
    }


def compute_thickness(mask: np.ndarray, spacing: Tuple[float, ...]) -> Dict[str, float]:
    """
    Thickness via distance transform.
    thickness = 2 * distance_to_boundary
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {"mean_thickness_mm": 0.0, "max_thickness_mm": 0.0}

    dt = distance_transform_edt(binary, sampling=spacing)
    vals = dt[binary > 0]
    return {
        "mean_thickness_mm": round(2.0 * float(np.mean(vals)), 2),
        "max_thickness_mm": round(2.0 * float(np.max(vals)), 2),
    }


def extract_morphology(mask: np.ndarray, spacing: Tuple[float, ...],
                       label_name: str, label_idx: int,
                       case_id: str) -> Dict:
    """Extract all morphological features for one label."""
    binary = (mask == label_idx).astype(np.uint8) if label_idx > 0 else (mask > 0).astype(np.uint8)
    vol = compute_volume(binary, spacing)
    csa = compute_csa(binary, spacing)
    thick = compute_thickness(binary, spacing)
    return {
        "patient_id": case_id,
        "muscle": label_name,
        "label_idx": label_idx,
        **vol, **csa, **thick,
    }


def analyze_all(pred_dir: str, output_dir: str,
                labels: Dict[int, str]) -> pd.DataFrame:
    """
    Extract morphology from all predicted masks.
    """
    print("\n" + "=" * 60)
    print("  Morphological Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(pred_dir)
                        if f.endswith(('.nii', '.nii.gz'))])

    all_results = []
    for i, pred_f in enumerate(pred_files):
        case_id = pred_f.replace(".nii.gz", "").replace(".nii", "")
        pred_nii = nib.load(os.path.join(pred_dir, pred_f))
        data = pred_nii.get_fdata().astype(np.int16)
        spacing = tuple(np.abs(np.diag(pred_nii.affine)[:3]).tolist())

        unique = np.unique(data).astype(int).tolist()
        for idx, name in labels.items():
            if idx == 0 or idx not in unique:
                continue
            morph = extract_morphology(data, spacing, name, idx, case_id)
            all_results.append(morph)

        if (i + 1) % 20 == 0 or i == len(pred_files) - 1:
            print(f"  [{i+1}/{len(pred_files)}] Analyzed")

    if not all_results:
        print("  [WARN] No morphological data extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    cols = ["patient_id", "muscle", "volume_mm3", "volume_cm3",
            "mean_csa_mm2", "max_csa_mm2", "mean_thickness_mm", "max_thickness_mm"]
    df = df[[c for c in cols if c in df.columns]]

    # Print summary
    print(f"\n  {'Muscle':<20s} {'Vol(cm3)':>10s} {'MaxCSA':>10s} {'MaxThick':>10s}")
    print(f"  {'-'*55}")
    for muscle in df["muscle"].unique():
        m = df[df["muscle"] == muscle]
        print(f"  {muscle:<20s} "
              f"{m['volume_cm3'].mean():>9.3f} "
              f"{m['max_csa_mm2'].mean():>9.1f} "
              f"{m['max_thickness_mm'].mean():>9.1f}")

    csv_path = os.path.join(output_dir, "eom_morphology_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    # Also save Excel if openpyxl is available
    try:
        xlsx_path = os.path.join(output_dir, "eom_morphology_results.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"  [SAVED] {xlsx_path}")
    except Exception:
        pass

    print(f"  * Morphological analysis complete ({len(df)} measurements)\n")
    return df
