"""
Visualization Module
=====================
Publication-quality figures for the conference paper.

Figures:
1. Pipeline methodology diagram
2. MRI preprocessing comparison
3. GT vs predicted segmentation
4. Five EOM segmentation overlay
5. Muscle volume comparison
6. CSA profile chart
7. Muscle thickness comparison

Author: TED EOM Research Group
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import Optional, Dict, List, Tuple


# EOM color palette (consistent across figures)
EOM_COLORS = {
    1: "#FF6B6B",  # SR/LPS - Red
    2: "#4ECDC4",  # IR - Teal
    3: "#45B7D1",  # MR - Blue
    4: "#96CEB4",  # LR - Green
    5: "#DDA0DD",  # SO - Plum
}

EOM_NAMES = {
    1: "SR/LPS",
    2: "IR",
    3: "MR",
    4: "LR",
    5: "SO",
}


def _mid(vol):
    return vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2


def _overlay_mask(ax, img_slice, mask_slice, alpha=0.45):
    """Overlay colored segmentation on grayscale image."""
    ax.imshow(img_slice.T, cmap="gray", origin="lower")
    colored = np.zeros((*mask_slice.T.shape, 4))
    m = mask_slice.T
    for idx, hex_c in EOM_COLORS.items():
        if idx in m:
            r, g, b = int(hex_c[1:3], 16)/255, int(hex_c[3:5], 16)/255, int(hex_c[5:7], 16)/255
            colored[m == idx] = [r, g, b, alpha]
    ax.imshow(colored, origin="lower")


def fig1_pipeline(output_path: str):
    """Methodology pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.set_xlim(0, 16); ax.set_ylim(0, 3.5); ax.axis("off")

    steps = [
        ("TOM500\nMRI", "#3498DB"), ("Pre-\nprocessing", "#2ECC71"),
        ("EOM Label\nExtraction", "#E67E22"), ("nnU-Net\nTraining", "#E74C3C"),
        ("Automatic\nSegmentation", "#9B59B6"), ("Evaluation\nMetrics", "#F39C12"),
        ("Morphological\nAnalysis", "#1ABC9C"), ("Results", "#34495E"),
    ]

    bw, bh = 1.5, 1.8
    y_c = 1.75
    for i, (label, color) in enumerate(steps):
        x = 0.3 + i * (bw + 0.3)
        rect = mpatches.FancyBboxPatch((x, y_c - bh/2), bw, bh,
            boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + bw/2, y_c, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + bw + 0.08, y_c), xytext=(x + bw + 0.02, y_c),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    ax.set_title("Proposed Methodology: TED EOM MRI Segmentation & Morphological Analysis",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 1: {output_path}")


def fig2_preprocessing(orig_path: str, prep_path: str, output_path: str):
    """MRI before vs after preprocessing."""
    orig = nib.load(orig_path).get_fdata()
    prep = nib.load(prep_path).get_fdata()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("MRI Preprocessing: Before vs After", fontsize=14, fontweight="bold")

    views = ["Axial", "Coronal", "Sagittal"]
    for col, vname in enumerate(views):
        mo = _mid(orig)
        sl_orig = [orig[mo[0],:,:], orig[:,mo[1],:], orig[:,:,mo[2]]]
        axes[0, col].imshow(sl_orig[col].T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original - {vname}", fontsize=10)
        axes[0, col].axis("off")

        mp = _mid(prep)
        sl_prep = [prep[mp[0],:,:], prep[:,mp[1],:], prep[:,:,mp[2]]]
        axes[1, col].imshow(sl_prep[col].T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Preprocessed - {vname}", fontsize=10)
        axes[1, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 2: {output_path}")


def fig3_gt_vs_pred(img_path: str, gt_path: str, pred_path: str, output_path: str):
    """Ground truth vs predicted segmentation."""
    img = nib.load(img_path).get_fdata()
    gt = nib.load(gt_path).get_fdata().astype(int)
    pred = nib.load(pred_path).get_fdata().astype(int)

    mid = _mid(img)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Ground Truth vs Predicted EOM Segmentation", fontsize=13, fontweight="bold")

    views = ["Axial", "Coronal", "Sagittal"]
    img_sl = [img[mid[0],:,:], img[:,mid[1],:], img[:,:,mid[2]]]
    gt_sl = [gt[mid[0],:,:], gt[:,mid[1],:], gt[:,:,mid[2]]]
    pred_sl = [pred[mid[0],:,:], pred[:,mid[1],:], pred[:,:,mid[2]]]

    for col in range(3):
        _overlay_mask(axes[0, col], img_sl[col], gt_sl[col])
        axes[0, col].set_title(f"Ground Truth - {views[col]}", fontsize=10)
        axes[0, col].axis("off")

        _overlay_mask(axes[1, col], img_sl[col], pred_sl[col])
        axes[1, col].set_title(f"Predicted - {views[col]}", fontsize=10)
        axes[1, col].axis("off")

    patches = [mpatches.Patch(color=c, label=EOM_NAMES.get(i, f"L{i}"), alpha=0.7)
               for i, c in EOM_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 3: {output_path}")


def fig4_eom_overlay(img_path: str, mask_path: str, output_path: str):
    """Five EOM segmentation overlay on MRI."""
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata().astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Extraocular Muscle Segmentation Overlay", fontsize=13, fontweight="bold")

    mid = _mid(img)
    views = ["Axial", "Coronal", "Sagittal"]
    img_sl = [img[mid[0],:,:], img[:,mid[1],:], img[:,:,mid[2]]]
    mask_sl = [mask[mid[0],:,:], mask[:,mid[1],:], mask[:,:,mid[2]]]

    for i, (ax, vname) in enumerate(zip(axes, views)):
        _overlay_mask(ax, img_sl[i], mask_sl[i])
        ax.set_title(vname, fontsize=11)
        ax.axis("off")

    patches = [mpatches.Patch(color=c, label=EOM_NAMES.get(idx, f"L{idx}"), alpha=0.7)
               for idx, c in EOM_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 4: {output_path}")


def fig5_volume_comparison(morph_df: pd.DataFrame, output_path: str):
    """Muscle volume comparison bar chart."""
    if morph_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    muscles = morph_df.groupby("muscle")["volume_cm3"].agg(["mean", "std"]).reset_index()
    colors = [EOM_COLORS.get(i+1, "#888") for i in range(len(muscles))]

    bars = ax.bar(range(len(muscles)), muscles["mean"], yerr=muscles["std"],
                  color=colors[:len(muscles)], alpha=0.85, edgecolor="white",
                  capsize=6, linewidth=1.5)

    ax.set_xticks(range(len(muscles)))
    ax.set_xticklabels(muscles["muscle"], fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Volume (cm$^3$)", fontsize=11)
    ax.set_title("EOM Volume Comparison (mean +/- SD)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, muscles["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 5: {output_path}")


def fig6_csa_chart(morph_df: pd.DataFrame, output_path: str):
    """Cross-sectional area chart."""
    if morph_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Sectional Area Analysis", fontsize=13, fontweight="bold")

    muscles = morph_df.groupby("muscle")

    # Mean CSA
    agg = muscles["mean_csa_mm2"].agg(["mean", "std"]).reset_index()
    colors = [EOM_COLORS.get(i+1, "#888") for i in range(len(agg))]
    axes[0].bar(range(len(agg)), agg["mean"], yerr=agg["std"],
                color=colors[:len(agg)], alpha=0.85, edgecolor="white", capsize=5)
    axes[0].set_xticks(range(len(agg)))
    axes[0].set_xticklabels(agg["muscle"], fontsize=9, rotation=20, ha="right")
    axes[0].set_ylabel("Area (mm$^2$)", fontsize=10)
    axes[0].set_title("Mean CSA", fontsize=11, fontweight="bold")

    # Max CSA
    agg2 = muscles["max_csa_mm2"].agg(["mean", "std"]).reset_index()
    axes[1].bar(range(len(agg2)), agg2["mean"], yerr=agg2["std"],
                color=colors[:len(agg2)], alpha=0.85, edgecolor="white", capsize=5)
    axes[1].set_xticks(range(len(agg2)))
    axes[1].set_xticklabels(agg2["muscle"], fontsize=9, rotation=20, ha="right")
    axes[1].set_ylabel("Area (mm$^2$)", fontsize=10)
    axes[1].set_title("Maximum CSA", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 6: {output_path}")


def fig7_thickness(morph_df: pd.DataFrame, output_path: str):
    """Muscle thickness comparison."""
    if morph_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EOM Thickness Analysis", fontsize=13, fontweight="bold")

    muscles = morph_df.groupby("muscle")

    for ax, col, title in [(axes[0], "mean_thickness_mm", "Mean Thickness"),
                           (axes[1], "max_thickness_mm", "Maximum Thickness")]:
        agg = muscles[col].agg(["mean", "std"]).reset_index()
        colors = [EOM_COLORS.get(i+1, "#888") for i in range(len(agg))]
        ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"],
               color=colors[:len(agg)], alpha=0.85, edgecolor="white", capsize=5)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["muscle"], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Thickness (mm)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Fig 7: {output_path}")


def fig_metrics_boxplot(metrics_df: pd.DataFrame, output_path: str):
    """Boxplot of Dice/IoU per muscle."""
    if metrics_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Segmentation Metrics per EOM", fontsize=13, fontweight="bold")

    for ax, metric, title in [(axes[0], "dice", "Dice Similarity Coefficient"),
                              (axes[1], "iou", "Intersection over Union")]:
        muscles = sorted(metrics_df["muscle"].unique())
        data = [metrics_df[metrics_df["muscle"] == m][metric].values for m in muscles]
        bp = ax.boxplot(data, labels=muscles, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            color = EOM_COLORS.get(i+1, "#888")
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(muscles, fontsize=9, rotation=20, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] Metrics boxplot: {output_path}")


def generate_case_figures(img_dir: str, gt_dir: str, pred_dir: str,
                          fig_dir: str, max_cases: int = 5):
    """Generate per-case segmentation overlay figures."""
    print("\n  Generating per-case segmentation figures ...")
    seg_fig_dir = os.path.join(fig_dir, "segmentation")
    os.makedirs(seg_fig_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(pred_dir)
                        if f.endswith(('.nii', '.nii.gz'))])[:max_cases]

    from src.dataset_manager import extract_case_id

    for pred_f in pred_files:
        case_id = extract_case_id(pred_f)

        # Find matching image and GT
        img_path = gt_path = None
        for ext in [".nii.gz", ".nii"]:
            for prefix in [case_id, f"{case_id}_0000"]:
                p = os.path.join(img_dir, prefix + ext)
                if os.path.isfile(p):
                    img_path = p
                    break
            p = os.path.join(gt_dir, case_id + ext)
            if os.path.isfile(p):
                gt_path = p

        pred_path = os.path.join(pred_dir, pred_f)

        if img_path and gt_path:
            fig3_gt_vs_pred(img_path, gt_path, pred_path,
                           os.path.join(seg_fig_dir, f"{case_id}_gt_vs_pred.png"))

        if img_path:
            fig4_eom_overlay(img_path, pred_path,
                            os.path.join(seg_fig_dir, f"{case_id}_overlay.png"))


def generate_all_figures(project_dir: str,
                         dataset_info: Dict,
                         metrics_df: Optional[pd.DataFrame] = None,
                         morph_df: Optional[pd.DataFrame] = None):
    """Generate all publication figures."""
    print("\n" + "=" * 60)
    print("  Generating Publication Figures")
    print("=" * 60)

    fig_dir = os.path.join(project_dir, "results", "figures")
    structure = dataset_info.get("structure", {})

    # Fig 1: Pipeline
    fig1_pipeline(os.path.join(fig_dir, "fig1_pipeline.png"))

    # Fig 2: Preprocessing (use first available case)
    val_img_dir = structure.get("val_image_dir")
    prep_dir = os.path.join(project_dir, "data", "preprocessed", "val", "images")
    if val_img_dir and os.path.isdir(prep_dir):
        imgs = sorted(os.listdir(val_img_dir))
        preps = sorted(os.listdir(prep_dir)) if os.path.isdir(prep_dir) else []
        if imgs and preps:
            fig2_preprocessing(
                os.path.join(val_img_dir, imgs[0]),
                os.path.join(prep_dir, preps[0]),
                os.path.join(fig_dir, "fig2_preprocessing.png")
            )

    # Fig 3, 4: Per-case segmentation figures
    pred_dir = os.path.join(project_dir, "results", "segmentation")
    eom_lbl_dir = os.path.join(project_dir, "data", "eom_labels", "val")
    if os.path.isdir(pred_dir) and val_img_dir:
        generate_case_figures(val_img_dir, eom_lbl_dir, pred_dir, fig_dir, max_cases=5)

    # Fig 5: Volume
    if morph_df is not None and not morph_df.empty:
        fig5_volume_comparison(morph_df, os.path.join(fig_dir, "fig5_volume.png"))

    # Fig 6: CSA
    if morph_df is not None and not morph_df.empty:
        fig6_csa_chart(morph_df, os.path.join(fig_dir, "fig6_csa.png"))

    # Fig 7: Thickness
    if morph_df is not None and not morph_df.empty:
        fig7_thickness(morph_df, os.path.join(fig_dir, "fig7_thickness.png"))

    # Metrics boxplot
    if metrics_df is not None and not metrics_df.empty:
        fig_metrics_boxplot(metrics_df, os.path.join(fig_dir, "fig8_metrics_boxplot.png"))

    print(f"  * All figures generated -> {fig_dir}\n")
