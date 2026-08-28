# TED EOM MRI Segmentation and Morphological Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2.2%2B-brightgreen.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![Dataset](https://img.shields.io/badge/Dataset-TOM500-orange.svg)](https://doi.org/10.6084/m9.figshare.27133389)

Automated Deep Learning-Based Segmentation and Quantitative Morphological Analysis of Extraocular Muscles in MRI Images of Patients with Thyroid Eye Disease.

---

## Overview

This repository provides an automated end-to-end deep learning framework designed for:
1. **Extraocular Muscle (EOM) Segmentation** from 3D Orbital MRI scans.
2. **Quantitative Morphological Analysis** extracting volume, cross-sectional area (CSA), and thickness across individual muscle structures.

Thyroid Eye Disease (TED), also known as Graves' ophthalmopathy, causes significant enlargement and inflammation of the extraocular muscles. This framework leverages **nnU-Net v2** on the **TOM500 dataset** to perform robust 3D MRI segmentation and quantitative assessment of muscle pathology.

---

## Objectives

* **Objective 1:** Automated 3D multi-class segmentation of extraocular muscles in orbital MRI scans.
* **Objective 2:** Automated quantitative morphological extraction to derive biomarker measurements (volume, maximum cross-sectional area, and maximum thickness).

---

## EOM Structures

The pipeline extracts and analyzes 5 distinct anatomical Extraocular Muscle structures:

* **SR/LPS Complex** (Superior Rectus / Levator Palpebrae Superioris Complex)
* **Inferior Rectus** (IR)
* **Medial Rectus** (MR)
* **Lateral Rectus** (LR)
* **Superior Oblique** (SO)

---

## Pipeline

```
TOM500 MRI
    │
    ▼
Preprocessing (Original Spacing & Intensity Scaling)
    │
    ▼
Label Extraction (Multi-organ to 5-Class EOM Label Mapping)
    │
    ▼
nnU-Net Dataset Conversion (Dataset501_TEDEOM)
    │
    ▼
nnU-Net Training (3D Full-Resolution Architecture)
    │
    ▼
Validation Inference (Auto-tiling & Sliding Window Prediction)
    │
    ▼
Segmentation Evaluation (Dice, IoU, Precision, Recall, HD95)
    │
    ▼
Morphological Analysis (Volume, Max CSA, Max Thickness)
    │
    ▼
Visualization (Metrics Distributions, Boxplots & Summaries)
```

---

## Project Structure

```
TED EOM MRI Segmentation/
├── data/                      # Local dataset storage directory (git-ignored)
│   └── TOM500/                # Place official TOM500 dataset here
├── nnUNet/                    # nnU-Net environment directory (git-ignored)
│   ├── nnUNet_raw/
│   ├── nnUNet_preprocessed/
│   └── nnUNet_results/
├── results/                   # Local execution outputs (git-ignored)
│   ├── segmentation/          # Predicted NIfTI masks
│   ├── metrics/               # Evaluation CSV outputs
│   ├── morphology/            # Morphological metrics CSV outputs
│   └── figures/               # Generated figures and charts
├── src/                       # Python Source Code
│   ├── __init__.py
│   ├── dataset_manager.py     # Dataset structure verification & setup
│   ├── evaluation.py          # Dice, IoU, Precision, Recall, HD95 calculations
│   ├── inference.py           # nnU-Net validation inference wrapper
│   ├── label_extraction.py    # EOM mask extraction from TOM500 labels
│   ├── main.py                # Command-line interface & pipeline orchestrator
│   ├── morphology.py         # 3D morphological feature extraction
│   ├── nnunet_setup.py        # nnU-Net environment & dataset conversion
│   ├── preprocessing.py       # MRI image preprocessing utilities
│   ├── train.py               # Planning & training orchestrator
│   └── visualization.py       # Visualization & figure generation
├── .gitignore                 # Tracked file filter configuration
├── CITATION.cff               # Repository metadata citation file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependency specifications
```

---

## Dataset

> **Note:** The TOM500 dataset is **NOT** included in this repository.

The dataset must be obtained directly from its official repository source:
* **Official Dataset DOI:** [https://doi.org/10.6084/m9.figshare.27133389](https://doi.org/10.6084/m9.figshare.27133389)

### Local Dataset Setup

After downloading the TOM500 dataset, place it locally inside the `data/TOM500/` directory:

```
data/
└── TOM500/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BARANIKESHAVA/TED-EOM-MRI-Segmentation.git
   cd TED-EOM-MRI-Segmentation
   ```

2. **Create and activate a Python Virtual Environment:**
   * **On Linux/macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   * **On Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Execute pipeline stages using the primary orchestrator entrypoint `src/main.py`:

* **Dataset Verification:**
  ```bash
  python src/main.py --mode dataset
  ```

* **Preprocessing & EOM Label Extraction:**
  ```bash
  python src/main.py --mode preprocess
  ```

* **Model Training (nnU-Net):**
  ```bash
  python src/main.py --mode train
  ```

* **Validation Inference:**
  ```bash
  python src/main.py --mode inference
  ```

* **Segmentation Evaluation:**
  ```bash
  python src/main.py --mode evaluate
  ```

* **Morphological Analysis:**
  ```bash
  python src/main.py --mode morphology
  ```

* **Complete End-to-End Pipeline Execution:**
  ```bash
  python src/main.py --mode full
  ```

---

## Results

Actual experimental results and prediction masks are generated locally upon running the pipeline with the TOM500 dataset and are **not** stored in this GitHub repository.

Executing the full pipeline populates the following local outputs:
* `results/run_summary.txt`: Comprehensive execution log and aggregate performance summary.
* `results/metrics/`: Evaluation tables (`segmentation_metrics.csv`).
* `results/morphology/`: Morphological measurement tables (`eom_morphology_results.csv`).
* `results/figures/`: Publication-ready visualization charts.

---

## Reproducibility

To ensure complete scientific reproducibility, the pipeline specifies fixed environmental configurations:

* **Python Version:** `3.10+`
* **nnU-Net Version:** `nnunetv2 >= 2.2.0`
* **Key Dependencies:** `torch >= 2.0.0`, `SimpleITK >= 2.2.0`, `nibabel >= 5.1.0`, `scikit-image >= 0.20.0`, `pandas >= 2.0.0`, `matplotlib >= 3.7.0`
* **Dataset Split:** Official TOM500 split (400 Training cases, 100 Validation cases)
* **Preprocessing:** Original voxel spacing retention (`None`), 5-class EOM multi-class integer remapping
* **Model Configuration:** nnU-Net `3d_fullres` (Dataset ID `501`)
* **Evaluation Metrics:** 3D Dice Similarity Coefficient (DSC), Intersection over Union (IoU), Precision, Recall, Hausdorff Distance 95th Percentile (HD95)

---

## Evaluation Metrics

The segmentation framework is quantitatively evaluated across 5 key metrics:

1. **Dice Similarity Coefficient (DSC):** Measures spatial overlap relative volume.
2. **Intersection over Union (IoU / Jaccard Index):** Quantifies volume intersection accuracy.
3. **Precision:** Ratio of true positive predicted muscle voxels.
4. **Recall (Sensitivity):** Portion of true ground-truth muscle voxels correctly segmented.
5. **HD95 (95th Percentile Hausdorff Distance):** Measures maximum 3D boundary distance error in millimeters (mm).

---

## Morphological Features

Automated 3D morphological analysis extracts clinically significant muscle parameters:

1. **Volume ($\text{cm}^3$):** Total 3D physical volume computed by voxel count multiplied by voxel spacing volume.
2. **Cross-Sectional Area ($\text{mm}^2$):** Slice-wise muscle area perpendicular to the orbital axis; maximum CSA ($\text{CSA}_{\max}$) is reported.
3. **Thickness ($\text{mm}$):** Maximum caliper distance / diameter per muscle cross-section.

---

## Research Integrity

This repository strictly complies with data privacy standards and research integrity guidelines:
* **No Patient Data:** No raw patient MRI images, ground-truth masks, or predicted NIfTI files are hosted or tracked in this repository.
* **Open Source Code:** All scripts, configurations, and evaluation metrics code are fully open for transparency and peer review.
* **Reproduction:** All quantitative metrics must be generated locally after obtaining access to the official TOM500 dataset.
