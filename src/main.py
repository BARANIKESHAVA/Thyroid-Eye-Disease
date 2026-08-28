"""
Main Execution Script - TED EOM MRI Segmentation Pipeline (TOM500)
====================================================================

Usage:
    python src/main.py --mode full         # Complete pipeline
    python src/main.py --mode dataset      # Dataset check only
    python src/main.py --mode preprocess   # Preprocessing only
    python src/main.py --mode train        # nnU-Net training only
    python src/main.py --mode inference    # Inference only
    python src/main.py --mode evaluate     # Evaluation only
    python src/main.py --mode morphology   # Morphological analysis only
    python src/main.py --mode visualize    # Generate figures only

Author: TED EOM Research Group
"""

import os
import sys
import argparse
import time
import json
import platform
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

from src.dataset_manager import verify_dataset
from src.preprocessing import preprocess_dataset
from src.label_extraction import (process_all_labels, get_eom_label_config,
                                   MULTICLASS_EOM_LABELS)
from src.nnunet_setup import setup_nnunet_env, convert_to_nnunet, detect_gpu
from src.train import plan_and_preprocess, train_model, check_trained_model
from src.inference import run_inference, prepare_val_for_inference
from src.evaluation import evaluate_all
from src.morphology import analyze_all
from src.visualization import generate_all_figures


# ──────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────

def get_paths(project_dir: str) -> dict:
    return {
        "data": os.path.join(project_dir, "data"),
        "nnunet": os.path.join(project_dir, "nnUNet"),
        "preprocessed": os.path.join(project_dir, "data", "preprocessed"),
        "eom_labels": os.path.join(project_dir, "data", "eom_labels"),
        "seg_output": os.path.join(project_dir, "results", "segmentation"),
        "metrics_output": os.path.join(project_dir, "results", "metrics"),
        "morph_output": os.path.join(project_dir, "results", "morphology"),
        "fig_output": os.path.join(project_dir, "results", "figures"),
        "inference_input": os.path.join(project_dir, "data", "inference_input"),
    }


def print_banner():
    print("\n")
    print("=" * 65)
    print("  TOM500 EOM SEGMENTATION RESEARCH PIPELINE")
    print("  Automated Deep Learning-Based Segmentation &")
    print("  Quantitative Morphological Analysis of Extraocular")
    print("  Muscles in MRI Images of Thyroid Eye Disease")
    print("=" * 65)
    print(f"  Project:  {PROJECT_DIR}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print("=" * 65)
    print()


# ──────────────────────────────────────────────────────────────
# Pipeline Steps
# ──────────────────────────────────────────────────────────────

def step_dataset(paths: dict, args) -> dict:
    """Check/download TOM500."""
    t0 = time.time()
    info = verify_dataset(paths["data"], dataset_path=args.dataset_path)
    if info["status"] != "ok":
        print("\n  [FATAL] Cannot proceed without TOM500 dataset.")
        print(f"  Configure --dataset-path or place TOM500 in: {paths['data']}")
        sys.exit(1)
    return {"time": time.time() - t0, "info": info}


def step_preprocess(paths: dict, dataset_info: dict, args) -> dict:
    """Preprocess train and val sets."""
    t0 = time.time()
    structure = dataset_info["structure"]

    # Preprocess validation images (used for inference/evaluation)
    val_out_img = os.path.join(paths["preprocessed"], "val", "images")
    val_out_lbl = os.path.join(paths["preprocessed"], "val", "labels")

    if structure["val_image_dir"] and structure["val_label_dir"]:
        preprocess_dataset(
            structure["val_image_dir"], structure["val_label_dir"],
            val_out_img, val_out_lbl,
            target_spacing=None,  # keep original spacing for nnU-Net
            max_cases=args.max_cases,
        )

    return {"time": time.time() - t0}


def step_extract_eom(paths: dict, dataset_info: dict, args) -> dict:
    """Extract EOM labels from TOM500 multi-organ masks."""
    t0 = time.time()
    structure = dataset_info["structure"]
    mode = args.eom_mode

    # Train labels
    if structure["train_label_dir"]:
        train_eom_dir = os.path.join(paths["eom_labels"], "train")
        process_all_labels(structure["train_label_dir"], train_eom_dir, mode=mode)

    # Val labels
    if structure["val_label_dir"]:
        val_eom_dir = os.path.join(paths["eom_labels"], "val")
        process_all_labels(structure["val_label_dir"], val_eom_dir, mode=mode)

    return {"time": time.time() - t0, "mode": mode}


def step_nnunet_convert(paths: dict, dataset_info: dict, args) -> dict:
    """Convert to nnU-Net format."""
    t0 = time.time()
    structure = dataset_info["structure"]

    env = setup_nnunet_env(paths["nnunet"])

    # Use EOM-extracted labels for training
    train_eom_dir = os.path.join(paths["eom_labels"], "train")
    val_img_dir = structure["val_image_dir"]

    # Use original images + EOM-extracted labels
    result = convert_to_nnunet(
        train_image_dir=structure["train_image_dir"],
        train_label_dir=train_eom_dir,
        val_image_dir=val_img_dir,
        val_label_dir=os.path.join(paths["eom_labels"], "val"),
        nnunet_raw_dir=env["nnUNet_raw"],
        dataset_id=args.dataset_id,
        eom_mode=args.eom_mode,
    )

    return {"time": time.time() - t0, "result": result}


def step_train(paths: dict, args) -> dict:
    """nnU-Net training."""
    t0 = time.time()
    env = setup_nnunet_env(paths["nnunet"])
    gpu = detect_gpu()
    device = "cuda" if gpu["available"] else "cpu"
    if args.device == "cpu":
        device = "cpu"

    # Plan and preprocess
    plan_and_preprocess(args.dataset_id)

    # Train
    result = train_model(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        device=device,
        num_epochs=args.num_epochs,
    )
    return {"time": time.time() - t0, "result": result, "device": device}


def step_inference(paths: dict, dataset_info: dict, args) -> dict:
    """Run inference on validation set."""
    t0 = time.time()
    env = setup_nnunet_env(paths["nnunet"])
    gpu = detect_gpu()
    device = "cuda" if gpu["available"] else "cpu"
    if args.device == "cpu":
        device = "cpu"

    structure = dataset_info["structure"]

    # Prepare val images with _0000 suffix
    prepare_val_for_inference(structure["val_image_dir"], paths["inference_input"])

    result = run_inference(
        input_dir=paths["inference_input"],
        output_dir=paths["seg_output"],
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=str(args.fold),
        device=device,
    )
    return {"time": time.time() - t0, "result": result}


def step_evaluate(paths: dict, args) -> dict:
    """Evaluate segmentation."""
    t0 = time.time()
    labels = get_eom_label_config(args.eom_mode)["labels"]

    gt_dir = os.path.join(paths["eom_labels"], "val")

    df = evaluate_all(
        pred_dir=paths["seg_output"],
        gt_dir=gt_dir,
        output_dir=paths["metrics_output"],
        labels=labels,
    )
    return {"time": time.time() - t0, "df": df}


def step_morphology(paths: dict, args) -> dict:
    """Morphological analysis."""
    t0 = time.time()
    labels = get_eom_label_config(args.eom_mode)["labels"]

    df = analyze_all(
        pred_dir=paths["seg_output"],
        output_dir=paths["morph_output"],
        labels=labels,
    )
    return {"time": time.time() - t0, "df": df}


def step_visualize(paths: dict, dataset_info: dict,
                   metrics_df, morph_df) -> dict:
    """Generate figures."""
    t0 = time.time()
    generate_all_figures(
        project_dir=PROJECT_DIR,
        dataset_info=dataset_info,
        metrics_df=metrics_df,
        morph_df=morph_df,
    )
    return {"time": time.time() - t0}


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def write_run_summary(paths: dict, dataset_info: dict,
                      metrics_df, morph_df, timings: dict, args):
    """Write run_summary.txt and print terminal summary."""
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("  TOM500 EOM SEGMENTATION RESEARCH PIPELINE - RUN SUMMARY")
    summary_lines.append("=" * 65)
    summary_lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"  Mode: {args.mode}")
    summary_lines.append(f"  EOM Mode: {args.eom_mode}")
    summary_lines.append("")

    # Dataset
    summary_lines.append("  DATASET")
    summary_lines.append(f"    Source: TOM500 (doi:{FIGSHARE_DOI})")
    summary_lines.append(f"    Path: {dataset_info.get('path', 'N/A')}")
    summary_lines.append(f"    Training cases: {dataset_info.get('num_train', 'N/A')}")
    summary_lines.append(f"    Validation cases: {dataset_info.get('num_val', 'N/A')}")
    summary_lines.append("")

    # Timings
    summary_lines.append("  PIPELINE STEPS")
    for step_name, elapsed in timings.items():
        status = f"Complete ({elapsed:.1f}s)" if elapsed >= 0 else "Skipped"
        summary_lines.append(f"    {step_name:<25s}: {status}")
    summary_lines.append("")

    # Metrics
    if metrics_df is not None and not metrics_df.empty:
        summary_lines.append("  SEGMENTATION PERFORMANCE")
        for muscle in metrics_df["muscle"].unique():
            m = metrics_df[metrics_df["muscle"] == muscle]
            summary_lines.append(f"    {muscle}:")
            summary_lines.append(f"      Dice: {m['dice'].mean():.4f} +/- {m['dice'].std():.4f}")
            summary_lines.append(f"      IoU:  {m['iou'].mean():.4f} +/- {m['iou'].std():.4f}")
        summary_lines.append(f"    Overall Dice: {metrics_df['dice'].mean():.4f} +/- {metrics_df['dice'].std():.4f}")
        summary_lines.append("")

    # Morphology
    if morph_df is not None and not morph_df.empty:
        summary_lines.append("  MORPHOLOGICAL ANALYSIS")
        for muscle in morph_df["muscle"].unique():
            m = morph_df[morph_df["muscle"] == muscle]
            summary_lines.append(f"    {muscle}:")
            summary_lines.append(f"      Volume: {m['volume_cm3'].mean():.3f} +/- {m['volume_cm3'].std():.3f} cm3")
            summary_lines.append(f"      Max CSA: {m['max_csa_mm2'].mean():.1f} +/- {m['max_csa_mm2'].std():.1f} mm2")
            summary_lines.append(f"      Max Thickness: {m['max_thickness_mm'].mean():.1f} +/- {m['max_thickness_mm'].std():.1f} mm")
        summary_lines.append("")

    # Configuration
    summary_lines.append("  CONFIGURATION")
    summary_lines.append(f"    nnU-Net dataset ID: {args.dataset_id}")
    summary_lines.append(f"    Configuration: {args.configuration}")
    summary_lines.append(f"    Fold: {args.fold}")
    summary_lines.append(f"    Device: {args.device}")
    summary_lines.append(f"    Python: {sys.version.split()[0]}")
    summary_lines.append(f"    Platform: {platform.system()} {platform.release()}")
    try:
        import torch
        summary_lines.append(f"    PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            summary_lines.append(f"    CUDA: {torch.version.cuda}")
    except ImportError:
        pass
    summary_lines.append("")

    # Output locations
    summary_lines.append("  OUTPUT FILES")
    summary_lines.append(f"    Segmentation masks: results/segmentation/")
    summary_lines.append(f"    Metrics CSV: results/metrics/segmentation_metrics.csv")
    summary_lines.append(f"    Morphology CSV: results/morphology/eom_morphology_results.csv")
    summary_lines.append(f"    Figures: results/figures/")
    summary_lines.append("")

    summary_lines.append("=" * 65)
    summary_lines.append("  PIPELINE COMPLETED")
    summary_lines.append("=" * 65)

    # Print to terminal
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save to file
    summary_path = os.path.join(PROJECT_DIR, "results", "run_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  [SAVED] {summary_path}")


# ──────────────────────────────────────────────────────────────
# Figshare DOI for summary
# ──────────────────────────────────────────────────────────────
FIGSHARE_DOI = "10.6084/m9.figshare.27133389"


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TOM500 EOM Segmentation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="full",
        choices=["full", "dataset", "preprocess", "train",
                 "inference", "evaluate", "morphology", "visualize"],
        help="Pipeline mode")
    parser.add_argument("--dataset-path", type=str, default=None,
        help="Path to TOM500 dataset root directory")
    parser.add_argument("--dataset-id", type=int, default=501,
        help="nnU-Net dataset ID (default: 501)")
    parser.add_argument("--configuration", type=str, default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres"],
        help="nnU-Net configuration")
    parser.add_argument("--fold", type=int, default=0, help="CV fold (default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
        choices=["cuda", "cpu"])
    parser.add_argument("--num-epochs", type=int, default=None,
        help="Override training epochs")
    parser.add_argument("--eom-mode", type=str, default="multiclass",
        choices=["binary", "multiclass"],
        help="EOM segmentation mode")
    parser.add_argument("--max-cases", type=int, default=None,
        help="Limit number of cases (for testing)")

    args = parser.parse_args()
    paths = get_paths(PROJECT_DIR)

    print_banner()

    timings = {}
    dataset_info = {}
    metrics_df = pd.DataFrame()
    morph_df = pd.DataFrame()

    try:
        # Always verify dataset first
        ds_result = step_dataset(paths, args)
        dataset_info = ds_result["info"]
        timings["Dataset Verification"] = ds_result["time"]

        if args.mode == "dataset":
            return

        # PREPROCESS
        if args.mode in ["full", "preprocess"]:
            t = step_preprocess(paths, dataset_info, args)
            timings["Preprocessing"] = t["time"]
            if args.mode == "preprocess":
                return

        # EXTRACT EOM LABELS
        if args.mode in ["full", "preprocess"]:
            t = step_extract_eom(paths, dataset_info, args)
            timings["EOM Label Extraction"] = t["time"]

        # NNUNET CONVERT
        if args.mode in ["full", "train"]:
            t = step_nnunet_convert(paths, dataset_info, args)
            timings["nnU-Net Conversion"] = t["time"]

        # TRAIN
        if args.mode in ["full", "train"]:
            t = step_train(paths, args)
            timings["nnU-Net Training"] = t["time"]
            if args.mode == "train":
                return

        # INFERENCE
        if args.mode in ["full", "inference"]:
            # Need EOM labels for val if not already done
            eom_val_dir = os.path.join(paths["eom_labels"], "val")
            if not os.path.isdir(eom_val_dir) or len(os.listdir(eom_val_dir)) == 0:
                step_extract_eom(paths, dataset_info, args)

            t = step_inference(paths, dataset_info, args)
            timings["Inference"] = t["time"]
            if args.mode == "inference":
                return

        # EVALUATE
        if args.mode in ["full", "evaluate"]:
            t = step_evaluate(paths, args)
            metrics_df = t["df"]
            timings["Evaluation"] = t["time"]
            if args.mode == "evaluate":
                return

        # MORPHOLOGY
        if args.mode in ["full", "morphology"]:
            t = step_morphology(paths, args)
            morph_df = t["df"]
            timings["Morphology"] = t["time"]
            if args.mode == "morphology":
                return

        # VISUALIZE
        if args.mode in ["full", "visualize"]:
            # Load metrics/morph from CSV if not in memory
            metrics_csv = os.path.join(paths["metrics_output"], "segmentation_metrics.csv")
            morph_csv = os.path.join(paths["morph_output"], "eom_morphology_results.csv")
            if metrics_df.empty and os.path.isfile(metrics_csv):
                metrics_df = pd.read_csv(metrics_csv)
            if morph_df.empty and os.path.isfile(morph_csv):
                morph_df = pd.read_csv(morph_csv)

            t = step_visualize(paths, dataset_info, metrics_df, morph_df)
            timings["Visualization"] = t["time"]

        # SUMMARY
        write_run_summary(paths, dataset_info, metrics_df, morph_df, timings, args)

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Pipeline stopped by user.")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
