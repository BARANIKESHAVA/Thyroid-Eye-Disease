"""
Inference Module
=================
Runs nnU-Net inference on validation cases.

Author: TED EOM Research Group
"""

import os
import sys
import subprocess
import numpy as np
import nibabel as nib
from typing import Dict, List


def run_inference(input_dir: str, output_dir: str,
                  dataset_id: int = 501,
                  configuration: str = "3d_fullres",
                  fold: str = "0",
                  trainer: str = "nnUNetTrainer",
                  device: str = "cuda") -> Dict:
    """
    Run nnU-Net v2 inference.

    Parameters
    ----------
    input_dir : str
        Directory with test images (_0000 suffix).
    output_dir : str
        Directory for predicted masks.
    """
    print("\n" + "=" * 60)
    print("  nnU-Net Inference")
    print("=" * 60)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Config:  {configuration}")
    print(f"  Device:  {device}")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [sys.executable, "-m",
           "nnunetv2.inference.predict_from_raw_data",
           "-i", input_dir, "-o", output_dir,
           "-d", str(dataset_id), "-c", configuration,
           "-f", fold, "-tr", trainer,
           "--device", device]

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
    except FileNotFoundError:
        print("  [ERROR] nnU-Net not installed.")
        success = False

    pred_files = []
    if os.path.isdir(output_dir):
        pred_files = sorted([f for f in os.listdir(output_dir)
                            if f.endswith(('.nii', '.nii.gz'))])

    status = "completed" if success else "failed"
    print(f"  Inference {status}: {len(pred_files)} predictions")

    return {
        "status": status,
        "output_dir": output_dir,
        "num_predictions": len(pred_files),
        "predictions": pred_files,
    }


def prepare_val_for_inference(val_image_dir: str, inference_input_dir: str) -> int:
    """
    Copy validation images to inference input directory with _0000 suffix.
    Returns number of files prepared.
    """
    import shutil
    from src.dataset_manager import extract_case_id

    os.makedirs(inference_input_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(val_image_dir)
                   if f.endswith(('.nii', '.nii.gz'))])

    count = 0
    for f in files:
        case_id = extract_case_id(f)
        dst = f"{case_id}_0000.nii.gz"
        shutil.copy2(os.path.join(val_image_dir, f),
                    os.path.join(inference_input_dir, dst))
        count += 1

    print(f"  Prepared {count} validation images for inference")
    return count
