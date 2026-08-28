"""
EOM Label Extraction Module
=============================
Extracts extraocular muscle (EOM) labels from TOM500 multi-organ masks.

TOM500 has 9 structures (label indices 1-9). This module:
- Reads the original multi-class labels
- Extracts only the 5 EOM classes
- Supports two configurations:
  A) Binary: background=0, all_eom=1
  B) Multi-class: background=0, SR/LPS=1, IR=2, MR=3, LR=4, SO=5

Author: TED EOM Research Group
"""

import os
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple


# Default TOM500 label mapping (will be verified from actual data)
# These are the expected indices based on the TOM500 publication
TOM500_LABEL_MAP = {
    0: "background",
    1: "optic_nerve",
    2: "orbital_fat",
    3: "lacrimal_gland",
    4: "eyeball",
    5: "sr_lps_complex",
    6: "inferior_rectus",
    7: "medial_rectus",
    8: "lateral_rectus",
    9: "superior_oblique",
}

# EOM source indices in TOM500 (labels 5-9)
EOM_SOURCE_INDICES = {
    5: "sr_lps_complex",
    6: "inferior_rectus",
    7: "medial_rectus",
    8: "lateral_rectus",
    9: "superior_oblique",
}

# Multi-class EOM remapping: original -> new index
MULTICLASS_EOM_MAP = {
    5: 1,  # SR/LPS -> 1
    6: 2,  # IR -> 2
    7: 3,  # MR -> 3
    8: 4,  # LR -> 4
    9: 5,  # SO -> 5
}

MULTICLASS_EOM_LABELS = {
    0: "background",
    1: "sr_lps_complex",
    2: "inferior_rectus",
    3: "medial_rectus",
    4: "lateral_rectus",
    5: "superior_oblique",
}

MULTICLASS_EOM_DISPLAY = {
    0: "Background",
    1: "SR/LPS Complex",
    2: "Inferior Rectus",
    3: "Medial Rectus",
    4: "Lateral Rectus",
    5: "Superior Oblique",
}

BINARY_EOM_LABELS = {
    0: "background",
    1: "extraocular_muscle",
}


def extract_eom_labels(label_data: np.ndarray,
                       mode: str = "multiclass",
                       source_map: Optional[Dict[int, int]] = None
                       ) -> np.ndarray:
    """
    Extract EOM labels from a TOM500 multi-organ mask.

    Parameters
    ----------
    label_data : np.ndarray
        Original TOM500 label mask (values 0-9).
    mode : str
        'binary' or 'multiclass'.
    source_map : dict or None
        Custom mapping {original_idx: new_idx}.
        If None, uses default TOM500 EOM indices.

    Returns
    -------
    np.ndarray
        EOM-only label mask.
    """
    if source_map is None:
        source_map = MULTICLASS_EOM_MAP

    eom_mask = np.zeros_like(label_data, dtype=np.uint8)

    if mode == "binary":
        for orig_idx in source_map.keys():
            eom_mask[label_data == orig_idx] = 1
    elif mode == "multiclass":
        for orig_idx, new_idx in source_map.items():
            eom_mask[label_data == orig_idx] = new_idx
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'multiclass'.")

    return eom_mask


def process_label_file(input_path: str, output_path: str,
                       mode: str = "multiclass") -> Dict:
    """
    Read a TOM500 label file, extract EOMs, save new label.
    Returns info about the extraction.
    """
    img = nib.load(input_path)
    label_data = img.get_fdata().astype(np.int16)
    affine = img.affine

    # Detect which labels are present
    unique_original = sorted(np.unique(label_data).astype(int).tolist())

    # Extract EOM
    eom_mask = extract_eom_labels(label_data, mode=mode)
    unique_eom = sorted(np.unique(eom_mask).astype(int).tolist())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img = nib.Nifti1Image(eom_mask.astype(np.int16), affine)
    nib.save(out_img, output_path)

    return {
        "input": os.path.basename(input_path),
        "output": os.path.basename(output_path),
        "original_labels": unique_original,
        "eom_labels": unique_eom,
        "mode": mode,
    }


def process_all_labels(input_dir: str, output_dir: str,
                       mode: str = "multiclass",
                       file_list: Optional[List[str]] = None) -> List[Dict]:
    """
    Process all label files in a directory.

    Parameters
    ----------
    input_dir : str
        Directory with original TOM500 label masks.
    output_dir : str
        Directory to save EOM-only masks.
    mode : str
        'binary' or 'multiclass'.
    file_list : list or None
        Specific files to process. If None, process all.

    Returns
    -------
    list of dict
        Processing summaries.
    """
    print("\n" + "=" * 60)
    print(f"  EOM Label Extraction (mode: {mode})")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    if file_list is None:
        file_list = sorted([f for f in os.listdir(input_dir)
                           if f.endswith(('.nii', '.nii.gz'))])

    results = []
    for i, fname in enumerate(file_list):
        info = process_label_file(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname),
            mode=mode,
        )
        results.append(info)
        if (i + 1) % 50 == 0 or i == len(file_list) - 1:
            print(f"  [{i+1}/{len(file_list)}] Processed")

    # Summary
    if results:
        all_eom = set()
        for r in results:
            all_eom.update(r["eom_labels"])
        print(f"\n  Mode: {mode}")
        print(f"  Files processed: {len(results)}")
        print(f"  EOM labels present: {sorted(all_eom)}")

        label_names = MULTICLASS_EOM_LABELS if mode == "multiclass" else BINARY_EOM_LABELS
        for idx in sorted(all_eom):
            name = label_names.get(idx, f"class_{idx}")
            print(f"    {idx}: {name}")

    print(f"  * Label extraction complete\n")
    return results


def get_eom_label_config(mode: str = "multiclass") -> Dict:
    """Return the label configuration for the given mode."""
    if mode == "multiclass":
        return {
            "labels": MULTICLASS_EOM_LABELS,
            "display_names": MULTICLASS_EOM_DISPLAY,
            "num_classes": 6,  # including background
        }
    else:
        return {
            "labels": BINARY_EOM_LABELS,
            "display_names": {0: "Background", 1: "Extraocular Muscle"},
            "num_classes": 2,
        }
