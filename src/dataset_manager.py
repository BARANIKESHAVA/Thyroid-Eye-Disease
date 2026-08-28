"""
TOM500 Dataset Manager
=======================
Handles downloading, verification, and inspection of the TOM500 dataset.

TOM500: A Multi-Organ Annotated Orbital MRI Dataset for Thyroid Eye Disease
- 500 T2-weighted orbital MRI cases (400 train / 100 val)
- 9 segmented orbital structures including 5 extraocular muscles
- NIfTI (.nii.gz) format
- Source: https://doi.org/10.6084/m9.figshare.27133389
- License: CC0

Author: TED EOM Research Group
"""

import os
import sys
import glob
import zipfile
import hashlib
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Figshare download info
FIGSHARE_DOI = "10.6084/m9.figshare.27133389"
FIGSHARE_URL = "https://figshare.com/ndownloader/articles/27133389/versions/1"
DATASET_ZIP_NAME = "TOM500.zip"

# TOM500 full label mapping (9 structures + background)
# The exact integer IDs will be auto-detected from the actual data.
# This is the expected ordering based on the TOM500 publication.
TOM500_EXPECTED_STRUCTURES = [
    "background",
    "optic_nerve",
    "orbital_fat",
    "lacrimal_gland",
    "eyeball",
    "sr_lps_complex",   # Superior rectus + levator palpebrae superioris
    "inferior_rectus",
    "medial_rectus",
    "lateral_rectus",
    "superior_oblique",
]

# EOM-only labels (the 5 muscles we focus on)
EOM_LABEL_NAMES = {
    "sr_lps_complex": "SR/LPS Complex",
    "inferior_rectus": "Inferior Rectus",
    "medial_rectus": "Medial Rectus",
    "lateral_rectus": "Lateral Rectus",
    "superior_oblique": "Superior Oblique",
}


def find_tom500(data_dir: str) -> Optional[str]:
    """
    Search for the TOM500 dataset in common locations.
    Returns the path to the TOM500 root directory if found, else None.
    """
    candidates = [
        os.path.join(data_dir, "TOM500"),
        os.path.join(data_dir, "tom500"),
        data_dir,  # maybe data_dir itself is TOM500 root
    ]
    for c in candidates:
        if _is_valid_tom500_dir(c):
            return c
    return None


def _is_valid_tom500_dir(path: str) -> bool:
    """Check if a directory looks like a valid TOM500 dataset root."""
    if not os.path.isdir(path):
        return False
    # Check for train/val subdirectories or image/label subdirectories
    has_train = os.path.isdir(os.path.join(path, "train"))
    has_val = os.path.isdir(os.path.join(path, "val"))
    if has_train and has_val:
        return True
    # Alternative flat structure: imagesTr/labelsTr
    has_imgtr = os.path.isdir(os.path.join(path, "imagesTr"))
    has_lbltr = os.path.isdir(os.path.join(path, "labelsTr"))
    if has_imgtr and has_lbltr:
        return True
    # Check if it contains NIfTI files directly with train/image pattern
    for sub in ["train/image", "train/images", "training/image", "training/images"]:
        d = os.path.join(path, sub)
        if os.path.isdir(d):
            niftis = glob.glob(os.path.join(d, "*.nii*"))
            if len(niftis) > 0:
                return True
    return False


def download_tom500(data_dir: str) -> Optional[str]:
    """
    Attempt to download TOM500 from Figshare.
    Returns the path to extracted dataset, or None on failure.
    """
    print("\n" + "=" * 60)
    print("  TOM500 Dataset Download")
    print("=" * 60)
    print(f"  Source: https://doi.org/{FIGSHARE_DOI}")
    print(f"  Target: {data_dir}")

    zip_path = os.path.join(data_dir, DATASET_ZIP_NAME)
    os.makedirs(data_dir, exist_ok=True)

    # Try downloading
    try:
        import urllib.request
        print(f"\n  Downloading TOM500.zip (~2.3 GB) ...")
        print(f"  URL: {FIGSHARE_URL}")
        print(f"  This may take several minutes depending on your connection.\n")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {pct:5.1f}% ({mb:.0f}/{total_mb:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(FIGSHARE_URL, zip_path, reporthook=_progress)
        print(f"\n  Download complete: {zip_path}")

    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"\n  === MANUAL DOWNLOAD INSTRUCTIONS ===")
        print(f"  1. Visit: https://doi.org/{FIGSHARE_DOI}")
        print(f"  2. Download TOM500.zip")
        print(f"  3. Extract to: {data_dir}")
        print(f"  4. Re-run the pipeline")
        return None

    # Extract
    if os.path.isfile(zip_path):
        print(f"\n  Extracting TOM500.zip ...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            print(f"  Extraction complete.")
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")
            return None

    return find_tom500(data_dir)


def detect_dataset_structure(tom500_dir: str) -> Dict:
    """
    Auto-detect the folder structure of TOM500.
    Returns a dict with paths to train/val image/label dirs.
    """
    result = {
        "root": tom500_dir,
        "train_image_dir": None,
        "train_label_dir": None,
        "val_image_dir": None,
        "val_label_dir": None,
        "structure_type": "unknown",
    }

    # Try: TOM500/train/image, TOM500/val/image
    search_patterns = [
        # (train_img, train_lbl, val_img, val_lbl)
        ("train/image", "train/label", "val/image", "val/label"),
        ("train/images", "train/labels", "val/images", "val/labels"),
        ("training/image", "training/label", "validation/image", "validation/label"),
        ("training/images", "training/labels", "validation/images", "validation/labels"),
        ("imagesTr", "labelsTr", "imagesTs", "labelsTs"),
        ("imagesTr", "labelsTr", "imagesVal", "labelsVal"),
    ]

    for ti, tl, vi, vl in search_patterns:
        tid = os.path.join(tom500_dir, ti)
        tld = os.path.join(tom500_dir, tl)
        vid = os.path.join(tom500_dir, vi)
        vld = os.path.join(tom500_dir, vl)
        if os.path.isdir(tid) and os.path.isdir(tld):
            result["train_image_dir"] = tid
            result["train_label_dir"] = tld
            result["structure_type"] = "standard"
            if os.path.isdir(vid):
                result["val_image_dir"] = vid
            if os.path.isdir(vld):
                result["val_label_dir"] = vld
            break

    # If no standard structure found, search recursively
    if result["train_image_dir"] is None:
        # Look for any directory containing .nii.gz files
        for root, dirs, files in os.walk(tom500_dir):
            niftis = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
            if len(niftis) > 50:
                rel = os.path.relpath(root, tom500_dir).lower()
                if 'label' in rel or 'mask' in rel or 'seg' in rel:
                    if 'train' in rel:
                        result["train_label_dir"] = root
                    elif 'val' in rel:
                        result["val_label_dir"] = root
                elif 'image' in rel or 'img' in rel:
                    if 'train' in rel:
                        result["train_image_dir"] = root
                    elif 'val' in rel:
                        result["val_image_dir"] = root

    return result


def scan_nifti_files(directory: str) -> List[str]:
    """Return sorted list of NIfTI file basenames in a directory."""
    if directory is None or not os.path.isdir(directory):
        return []
    files = sorted([f for f in os.listdir(directory)
                    if f.endswith(('.nii', '.nii.gz'))])
    return files


def extract_case_id(filename: str) -> str:
    """Extract case/patient ID from a NIfTI filename."""
    base = filename.replace(".nii.gz", "").replace(".nii", "")
    # Remove common suffixes like _0000
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return base


def detect_label_mapping(label_dir: str, max_scan: int = 10) -> Dict[int, str]:
    """
    Scan label files to detect unique integer labels.
    Returns mapping {int_value: structure_name}.
    """
    if label_dir is None or not os.path.isdir(label_dir):
        return {}

    all_labels = set()
    files = scan_nifti_files(label_dir)[:max_scan]
    for f in files:
        lbl = nib.load(os.path.join(label_dir, f)).get_fdata()
        all_labels.update(np.unique(lbl).astype(int).tolist())

    all_labels = sorted(all_labels)

    # Map to expected TOM500 structure names
    mapping = {}
    for idx in all_labels:
        if idx < len(TOM500_EXPECTED_STRUCTURES):
            mapping[idx] = TOM500_EXPECTED_STRUCTURES[idx]
        else:
            mapping[idx] = f"unknown_class_{idx}"

    return mapping


def get_sample_info(image_path: str) -> Dict:
    """Load a sample NIfTI and return its metadata."""
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine
    spacing = tuple(np.abs(np.diag(affine)[:3]).round(4).tolist())
    return {
        "shape": data.shape,
        "spacing": spacing,
        "dtype": str(data.dtype),
        "intensity_range": (float(data.min()), float(data.max())),
    }


def verify_dataset(data_dir: str, dataset_path: Optional[str] = None) -> Dict:
    """
    Full dataset verification pipeline.
    1. Find or download TOM500
    2. Detect structure
    3. Verify image-label correspondence
    4. Report statistics

    Returns a comprehensive dataset info dict.
    """
    print("\n" + "=" * 60)
    print("  TOM500 Dataset Verification")
    print("=" * 60)

    # Step 1: Find dataset
    tom500_dir = dataset_path or find_tom500(data_dir)

    if tom500_dir is None:
        print("  TOM500 not found locally. Attempting download ...")
        tom500_dir = download_tom500(data_dir)

    if tom500_dir is None:
        print("\n  [ERROR] TOM500 dataset is not available.")
        print(f"  Please download from: https://doi.org/{FIGSHARE_DOI}")
        print(f"  Extract to: {data_dir}/TOM500/")
        print("  Expected structure:")
        print("    TOM500/")
        print("      train/")
        print("        image/   (400 .nii.gz files)")
        print("        label/   (400 .nii.gz files)")
        print("      val/")
        print("        image/   (100 .nii.gz files)")
        print("        label/   (100 .nii.gz files)")
        return {"status": "not_found", "path": None}

    print(f"  Found TOM500 at: {tom500_dir}")

    # Step 2: Detect structure
    structure = detect_dataset_structure(tom500_dir)
    print(f"  Structure type: {structure['structure_type']}")
    print(f"  Train images: {structure['train_image_dir']}")
    print(f"  Train labels: {structure['train_label_dir']}")
    print(f"  Val images:   {structure['val_image_dir']}")
    print(f"  Val labels:   {structure['val_label_dir']}")

    if structure["train_image_dir"] is None:
        print("\n  [ERROR] Could not detect train/val structure.")
        print("  Please ensure TOM500 has train/image and train/label directories.")
        return {"status": "structure_error", "path": tom500_dir}

    # Step 3: Scan files
    train_images = scan_nifti_files(structure["train_image_dir"])
    train_labels = scan_nifti_files(structure["train_label_dir"])
    val_images = scan_nifti_files(structure["val_image_dir"])
    val_labels = scan_nifti_files(structure["val_label_dir"])

    print(f"\n  Train images: {len(train_images)}")
    print(f"  Train labels: {len(train_labels)}")
    print(f"  Val images:   {len(val_images)}")
    print(f"  Val labels:   {len(val_labels)}")

    # Step 4: Match IDs
    train_img_ids = set(extract_case_id(f) for f in train_images)
    train_lbl_ids = set(extract_case_id(f) for f in train_labels)
    val_img_ids = set(extract_case_id(f) for f in val_images)
    val_lbl_ids = set(extract_case_id(f) for f in val_labels)

    train_matched = sorted(train_img_ids & train_lbl_ids)
    train_unmatched_img = train_img_ids - train_lbl_ids
    train_unmatched_lbl = train_lbl_ids - train_img_ids
    val_matched = sorted(val_img_ids & val_lbl_ids)

    print(f"\n  Train matched pairs: {len(train_matched)}")
    if train_unmatched_img:
        print(f"  [WARN] Train images without labels: {len(train_unmatched_img)}")
    if train_unmatched_lbl:
        print(f"  [WARN] Train labels without images: {len(train_unmatched_lbl)}")
    print(f"  Val matched pairs:   {len(val_matched)}")

    # Step 5: Detect labels
    label_mapping = detect_label_mapping(structure["train_label_dir"])
    print(f"\n  Detected label mapping:")
    for idx, name in label_mapping.items():
        print(f"    {idx}: {name}")

    # Step 6: Sample metadata
    sample_info = {}
    if train_images:
        sample_path = os.path.join(structure["train_image_dir"], train_images[0])
        sample_info = get_sample_info(sample_path)
        print(f"\n  Sample MRI info ({train_images[0]}):")
        print(f"    Shape:     {sample_info['shape']}")
        print(f"    Spacing:   {sample_info['spacing']} mm")
        print(f"    Intensity: [{sample_info['intensity_range'][0]:.1f}, "
              f"{sample_info['intensity_range'][1]:.1f}]")

    # Build summary
    info = {
        "status": "ok",
        "path": tom500_dir,
        "structure": structure,
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "train_matched_ids": train_matched,
        "val_matched_ids": val_matched,
        "label_mapping": label_mapping,
        "sample_info": sample_info,
        "num_train": len(train_matched),
        "num_val": len(val_matched),
    }

    print(f"\n  {'=' * 50}")
    print(f"  TOM500 DATASET SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Total cases:     {len(train_matched) + len(val_matched)}")
    print(f"  Training cases:  {len(train_matched)}")
    print(f"  Validation cases: {len(val_matched)}")
    print(f"  MRI format:      NIfTI (.nii.gz)")
    print(f"  Structures:      {len(label_mapping)} classes")
    if sample_info:
        print(f"  Voxel spacing:   {sample_info.get('spacing', 'N/A')} mm")
    print(f"  Status:          VERIFIED")
    print(f"  {'=' * 50}\n")

    return info


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    verify_dataset(base)
