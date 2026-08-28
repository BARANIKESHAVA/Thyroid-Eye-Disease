"""
nnU-Net Training Module
========================
Wraps nnU-Net v2 plan_and_preprocess and training commands.

Author: TED EOM Research Group
"""

import os
import sys
import subprocess
from typing import Dict, Optional


def plan_and_preprocess(dataset_id: int, verify: bool = True) -> bool:
    """Run nnU-Net experiment planning and preprocessing."""
    print("\n" + "=" * 60)
    print(f"  nnU-Net Planning & Preprocessing (Dataset {dataset_id})")
    print("=" * 60)

    cmd = [sys.executable, "-m",
           "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
           "-d", str(dataset_id)]
    if verify:
        cmd.append("--verify_dataset_integrity")

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
        if result.returncode == 0:
            print("  * Planning & preprocessing complete")
            return True
        else:
            print("  [ERROR] Planning failed")
            return False
    except FileNotFoundError:
        print("  [ERROR] nnU-Net not installed. Run: pip install nnunetv2")
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] Timed out (>2 hours)")
        return False


def train_model(dataset_id: int,
                configuration: str = "3d_fullres",
                fold: int = 0,
                trainer: str = "nnUNetTrainer",
                device: str = "cuda",
                num_epochs: Optional[int] = None) -> Dict:
    """
    Train nnU-Net model.

    Parameters
    ----------
    dataset_id : int
    configuration : str
        '2d', '3d_fullres', '3d_lowres', or '3d_cascade_fullres'.
    fold : int
        0-4, or use 'all'.
    trainer : str
    device : str
        'cuda' or 'cpu'.
    num_epochs : int or None
        Override default epochs.
    """
    print("\n" + "=" * 60)
    print("  nnU-Net Training")
    print("=" * 60)
    print(f"  Dataset:       {dataset_id}")
    print(f"  Configuration: {configuration}")
    print(f"  Fold:          {fold}")
    print(f"  Device:        {device}")
    if num_epochs:
        print(f"  Epochs:        {num_epochs}")

    cmd = [sys.executable, "-m", "nnunetv2.run.run_training",
           str(dataset_id), configuration, str(fold),
           "-tr", trainer, "-device", device, "--npz"]

    print(f"  Command: {' '.join(cmd)}")
    print("  Training started ... (this will take hours)\n")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
    except FileNotFoundError:
        print("  [ERROR] nnU-Net not installed.")
        success = False
    except KeyboardInterrupt:
        print("  [INTERRUPTED]")
        success = False

    status = "completed" if success else "failed"
    print(f"\n  Training {status}")
    return {"status": status, "dataset_id": dataset_id,
            "configuration": configuration, "fold": fold, "device": device}


def check_trained_model(dataset_id: int, dataset_name: str = "TOM500EOM",
                        configuration: str = "3d_fullres",
                        fold: int = 0, trainer: str = "nnUNetTrainer") -> bool:
    """Check if a trained model checkpoint exists."""
    results_dir = os.environ.get("nnUNet_results", "")
    if not results_dir:
        return False
    model_dir = os.path.join(
        results_dir,
        f"Dataset{dataset_id:03d}_{dataset_name}",
        f"{trainer}__{configuration}__nnUNetPlans",
        f"fold_{fold}"
    )
    return os.path.isfile(os.path.join(model_dir, "checkpoint_final.pth"))
