"""
nnU-Net Setup & Dataset Preparation
======================================
Converts TOM500 EOM data into nnU-Net v2 format and manages environment.

Author: TED EOM Research Group
"""

import os
import sys
import json
import shutil
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional
from src.dataset_manager import extract_case_id, scan_nifti_files
from src.label_extraction import MULTICLASS_EOM_LABELS, BINARY_EOM_LABELS


def setup_nnunet_env(nnunet_base: str) -> Dict[str, str]:
    """Configure nnU-Net environment variables."""
    paths = {
        "nnUNet_raw": os.path.join(nnunet_base, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(nnunet_base, "nnUNet_preprocessed"),
        "nnUNet_results": os.path.join(nnunet_base, "nnUNet_results"),
    }
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
        os.environ[key] = os.path.abspath(path)

    print("\n" + "=" * 60)
    print("  nnU-Net Environment")
    print("=" * 60)
    for key, path in paths.items():
        print(f"  {key}: {os.path.abspath(path)}")
    return paths


def convert_to_nnunet(train_image_dir: str, train_label_dir: str,
                      val_image_dir: str, val_label_dir: str,
                      nnunet_raw_dir: str,
                      dataset_id: int = 501,
                      dataset_name: str = "TOM500EOM",
                      eom_mode: str = "multiclass") -> Dict:
    """
    Convert TOM500 EOM data to nnU-Net format.
    Preserves the official 400/100 train/val split.
    """
    print("\n" + "=" * 60)
    print("  nnU-Net Dataset Conversion")
    print("=" * 60)

    ds_dir = os.path.join(nnunet_raw_dir, f"Dataset{dataset_id:03d}_{dataset_name}")
    img_tr = os.path.join(ds_dir, "imagesTr")
    lbl_tr = os.path.join(ds_dir, "labelsTr")
    img_ts = os.path.join(ds_dir, "imagesTs")

    for d in [img_tr, lbl_tr, img_ts]:
        os.makedirs(d, exist_ok=True)

    # Get label config
    labels = MULTICLASS_EOM_LABELS if eom_mode == "multiclass" else BINARY_EOM_LABELS

    # Copy training files
    train_imgs = scan_nifti_files(train_image_dir)
    train_lbls = scan_nifti_files(train_label_dir)
    train_lbl_ids = {extract_case_id(f): f for f in train_lbls}

    n_train = 0
    for img_file in train_imgs:
        case_id = extract_case_id(img_file)
        if case_id in train_lbl_ids:
            # Image: add _0000 channel suffix
            dst_img = f"{case_id}_0000.nii.gz"
            dst_lbl = f"{case_id}.nii.gz"
            shutil.copy2(os.path.join(train_image_dir, img_file),
                        os.path.join(img_tr, dst_img))
            shutil.copy2(os.path.join(train_label_dir, train_lbl_ids[case_id]),
                        os.path.join(lbl_tr, dst_lbl))
            n_train += 1

    # Copy validation files as test set
    val_imgs = scan_nifti_files(val_image_dir)
    n_val = 0
    for img_file in val_imgs:
        case_id = extract_case_id(img_file)
        dst_img = f"{case_id}_0000.nii.gz"
        shutil.copy2(os.path.join(val_image_dir, img_file),
                    os.path.join(img_ts, dst_img))
        n_val += 1

    print(f"  Training cases:   {n_train}")
    print(f"  Test cases:       {n_val}")

    # Validate label values in a sample
    sample_lbls = scan_nifti_files(lbl_tr)[:5]
    all_unique = set()
    for f in sample_lbls:
        lbl = nib.load(os.path.join(lbl_tr, f)).get_fdata()
        all_unique.update(np.unique(lbl).astype(int).tolist())
    print(f"  Label values found: {sorted(all_unique)}")

    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "T2w_MRI"},
        "labels": {name: idx for idx, name in labels.items()},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "TOM500 - Extraocular Muscle Segmentation for TED",
        "reference": "https://doi.org/10.6084/m9.figshare.27133389",
        "licence": "CC0",
    }

    json_path = os.path.join(ds_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"  dataset.json saved: {json_path}")

    # Verify image-label correspondence
    final_imgs = set(extract_case_id(f) for f in scan_nifti_files(img_tr))
    final_lbls = set(extract_case_id(f) for f in scan_nifti_files(lbl_tr))
    mismatched = final_imgs.symmetric_difference(final_lbls)
    if mismatched:
        print(f"  [WARN] {len(mismatched)} mismatched image-label pairs!")
    else:
        print(f"  All {n_train} image-label pairs verified")

    print(f"  * nnU-Net dataset ready: {ds_dir}\n")

    return {
        "status": "ok",
        "dataset_dir": ds_dir,
        "dataset_id": dataset_id,
        "num_training": n_train,
        "num_testing": n_val,
        "labels": labels,
        "eom_mode": eom_mode,
    }


def detect_gpu() -> Dict:
    """Detect GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU: {name} ({mem:.1f} GB)")
            return {"available": True, "device": "cuda", "name": name, "memory_gb": round(mem, 2)}
    except ImportError:
        pass
    print("  GPU: Not available (CPU mode)")
    return {"available": False, "device": "cpu", "name": None, "memory_gb": None}
