# Scientific Reproducibility Guide

This guide details the complete hardware, environment, data acquisition, and step-by-step CLI execution workflow required to reproduce all findings reported in the HELPix-R study.

---

## 1. Hardware & System Specifications
- **Operating System**: Windows 10/11 or Ubuntu 22.04 LTS
- **GPU Requirement**: NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., RTX 4050, RTX 3070, T4, A100). Automated Mixed Precision (AMP FP16) is enabled by default to maintain peak VRAM $< 6\text{ GB}$.
- **System Memory**: $\ge 16\text{ GB}$ RAM (lazy image reading ensures RAM stays $< 8\text{ GB}$).
- **Disk Storage**: $\approx 100\text{ GB}$ for raw image archives and derived artifacts.

---

## 2. Software Environment Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.14.6)
- CUDA Toolkit 12.0+ (tested with CUDA 12.6)

### Option A: Standard Pip Installation
```bash
git clone https://github.com/CSrajput-ux/Helpix-reasearh-.git
cd HELPix-R
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Option B: Conda Environment
```bash
conda env create -f environment.yml
conda activate helpix-r
```

---

## 3. Dataset Acquisition and Directory Hierarchy
Due to licensing and ethical distribution restrictions, raw image datasets must be obtained from their official sources:

1. **ISIC 2024 SLICE-3D**: Download from the ISIC Archive (https://www.isic-archive.com).
2. **HAM10000**: Download from Harvard Dataverse (https://doi.org/10.7910/DVN/DBW86T).
3. **PAD-UFES-20**: Download from Mendeley Data (https://doi.org/10.17632/zr7vgbcyr2.1).

Place the downloaded images in the following structure:
```
HELPix-R/
└── data/
    ├── raw/
    │   ├── ISIC_2024/
    │   └── HAM10000/
    └── external_test/
        └── PAD-UFES-20/
```

---

## 4. End-to-End Reproduction Pipeline

All scripts can be executed via the command line.

### Step 1: Audit Dataset Characteristics
Verify raw image counts and generate Table 1:
```bash
python scripts/audit_dataset.py
```
*Expected Output*: `results/tables/Table_1_dataset_characteristics.csv`

### Step 2: Cryptographic SHA-256 Deduplication
Execute 64 KB chunked SHA-256 deduplication:
```bash
python scripts/deduplicate.py
```
*Expected Output*: `results/tables/deduplication_statistics.csv` (7,697 exact duplicates removed).

### Step 3: Leakage-Controlled Group Splitting
Partition the development cohort into Train (70%), Validation (15%), and Internal Test (15%) using patient/lesion grouping:
```bash
python scripts/create_splits.py --seed 42
```
*Expected Output*: `results/tables/final_split_statistics.csv` (Zero group overlap verified).

### Step 4: Model Training
Train HELPix-R using Focal Loss, AdamW, and ReduceLROnPlateau:
```bash
python scripts/train.py --config configs/training_config.yaml --model helpix_r
```
*Expected Output*: Saved model checkpoint `results/helpix_r_best.pt`.

To train baseline models:
```bash
python scripts/train.py --model resnet50
python scripts/train.py --model efficientnet_b0
python scripts/train.py --model mobilenetv3
python scripts/train.py --model convnext_tiny
```

### Step 5: Validation-Based Probability Calibration
Fit Temperature Scaling parameter $T$ exclusively on the validation cohort:
```bash
python scripts/calibrate.py --config configs/calibration_config.yaml
```
*Expected Output*: `results/tables/Table_calibration_comparison.csv` ($T = 1.1799$).

### Step 6: Internal Test Evaluation
Evaluate diagnostic performance on the 55,130-sample internal test set:
```bash
python scripts/evaluate.py
```
*Expected Output*: `results/metrics/final_metrics.csv` and `results/tables/baseline_comparison.csv`.

### Step 7: External Smartphone-Domain Testing
Evaluate out-of-distribution performance on PAD-UFES-20 ($N = 654$):
```bash
python scripts/external_test.py
```
*Expected Output*: `results/tables/domain_shift_analysis.csv`.

### Step 8: Bootstrap Confidence Intervals
Compute 95% non-parametric bootstrap confidence intervals (1,000 resamples):
```bash
python scripts/bootstrap_ci.py --n_bootstrap 1000 --seed 42
```
*Expected Output*: Internal AUROC 95% CI: $[0.9666, 0.9866]$.

### Step 9: Grad-CAM Explainability & Perturbation Test
Evaluate attribution sensitivity by masking the top-40% salient regions:
```bash
python scripts/generate_gradcam.py
```
*Expected Output*: `results/tables/gradcam_deletion_results.csv` (Mean drop: 0.0723, Median drop: 0.0487).

### Step 10: Generate Publication Figures
Render manuscript figures:
```bash
python scripts/generate_figures.py
```
*Expected Output*: Publication PNG files in `results/figures/`.

---

## 5. Random Seeds & Determinism Considerations
All scripts explicitly enforce `seed = 42` across Python `random`, `numpy.random`, and `torch.manual_seed`.
`torch.backends.cudnn.deterministic = True` is enabled. Note that minor floating-point variations may arise across different GPU hardware architectures due to non-deterministic atomic CUDA operations.
