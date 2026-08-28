"""
Segmentation Evaluation Module
================================
Computes DSC, IoU, Precision, Recall, HD95 per EOM class.

Author: TED EOM Research Group
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion
from typing import Dict, List, Optional, Tuple


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """DSC = 2|A^B| / (|A| + |B|)"""
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    return float(2.0 * np.logical_and(p, g).sum() / (p.sum() + g.sum()))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU = |A^B| / |AuB|"""
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision = TP / (TP + FP)"""
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall = TP / (TP + FN)"""
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    tp = np.logical_and(p, g).sum()
    fn = np.logical_and(~p, g).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                          spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff Distance in mm."""
    p, g = (pred > 0).astype(bool), (gt > 0).astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    # Surface voxels
    p_surf = p ^ _safe_erode(p)
    g_surf = g ^ _safe_erode(g)

    dt_p = distance_transform_edt(~p, sampling=spacing)
    dt_g = distance_transform_edt(~g, sampling=spacing)

    d1 = dt_g[p_surf] if p_surf.any() else np.array([])
    d2 = dt_p[g_surf] if g_surf.any() else np.array([])

    if len(d1) == 0 and len(d2) == 0:
        return float("inf")

    all_d = np.concatenate([d1, d2]) if len(d1) > 0 and len(d2) > 0 else (d1 if len(d1) > 0 else d2)
    return float(np.percentile(all_d, 95))


def _safe_erode(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask)
    return mask if eroded.sum() == 0 else eroded


def evaluate_case(pred_data: np.ndarray, gt_data: np.ndarray,
                  labels: Dict[int, str],
                  spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                  case_id: str = "") -> List[Dict]:
    """Evaluate one case across all labels."""
    results = []
    for idx, name in labels.items():
        if idx == 0:
            continue
        p = (pred_data == idx).astype(np.uint8)
        g = (gt_data == idx).astype(np.uint8)
        results.append({
            "case_id": case_id,
            "muscle": name,
            "label_idx": idx,
            "dice": dice_coefficient(p, g),
            "iou": iou_score(p, g),
            "precision": precision_score(p, g),
            "recall": recall_score(p, g),
            "hausdorff_95": hausdorff_distance_95(p, g, spacing),
        })
    return results


def evaluate_all(pred_dir: str, gt_dir: str, output_dir: str,
                 labels: Dict[int, str]) -> pd.DataFrame:
    """
    Evaluate all prediction-GT pairs.
    Returns DataFrame with per-case per-muscle metrics.
    """
    print("\n" + "=" * 60)
    print("  Segmentation Evaluation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(pred_dir)
                        if f.endswith(('.nii', '.nii.gz'))])

    all_results = []
    for i, pred_f in enumerate(pred_files):
        case_id = pred_f.replace(".nii.gz", "").replace(".nii", "")

        # Find GT
        gt_path = None
        for ext in [".nii.gz", ".nii"]:
            p = os.path.join(gt_dir, case_id + ext)
            if os.path.isfile(p):
                gt_path = p
                break
        if gt_path is None:
            continue

        pred_nii = nib.load(os.path.join(pred_dir, pred_f))
        gt_nii = nib.load(gt_path)
        pred_data = pred_nii.get_fdata().astype(np.int16)
        gt_data = gt_nii.get_fdata().astype(np.int16)
        spacing = tuple(np.abs(np.diag(gt_nii.affine)[:3]).tolist())

        case_results = evaluate_case(pred_data, gt_data, labels, spacing, case_id)
        all_results.extend(case_results)

        if (i + 1) % 20 == 0 or i == len(pred_files) - 1:
            print(f"  [{i+1}/{len(pred_files)}] Evaluated")

    if not all_results:
        print("  [WARN] No cases evaluated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Print per-muscle summary
    print(f"\n  {'Muscle':<20s} {'Dice':>8s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'HD95':>8s}")
    print(f"  {'-'*60}")
    for muscle in df["muscle"].unique():
        m = df[df["muscle"] == muscle]
        hd = m["hausdorff_95"].replace([np.inf], np.nan)
        print(f"  {muscle:<20s} "
              f"{m['dice'].mean():>7.4f} "
              f"{m['iou'].mean():>7.4f} "
              f"{m['precision'].mean():>7.4f} "
              f"{m['recall'].mean():>7.4f} "
              f"{hd.mean():>7.2f}")

    print(f"\n  Overall mean Dice: {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")

    csv_path = os.path.join(output_dir, "segmentation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")
    print(f"  * Evaluation complete ({len(df)} measurements)\n")
    return df
