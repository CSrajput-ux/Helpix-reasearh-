# HELPix-R: Reliability-Oriented Binary Skin-Lesion Malignancy Classification

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-[CONFIRMATION_PENDING]-lightgrey.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-[PENDING_RELEASE]-yellow.svg)](docs/BMC_CODE_AVAILABILITY_PREPUBLICATION.md)

This is a reproducibility-oriented scientific research repository containing the code and derived artifacts required to reproduce the reported analysis for:

> **Reliability-Oriented Binary Skin-Lesion Malignancy Classification Using HELPix-R: A Leakage-Controlled Deep-Learning Framework With Calibration, Explainability, and External Testing**  
> *Target Journal*: BMC Medical Imaging  
> *Authors*: Chhotu<sup>1</sup>, Dr. Rishi Gupta<sup>2</sup>  
> <sup>1</sup>Manipal University Jaipur; <sup>2</sup>[AFFILIATION TO BE CONFIRMED]

---

## Overview
Automated classification of pigmented skin lesions from dermoscopic and clinical images holds significant potential for clinical decision support. However, clinical adoption requires not only strong discrimination, but also strict data-leakage controls, calibrated probability estimates, predictive uncertainty quantification, and characterization under real-world domain shifts.

**HELPix-R** addresses these challenges through a unified deep-learning framework incorporating:
1. **Multi-scale intermediate feature extraction** from an EfficientNet-B0 backbone with per-scale Squeeze-and-Excitation (SE) channel attention.
2. **Leakage-controlled group-aware data splitting** that strictly prevents patient- and lesion-level overlap.
3. **Post-hoc probability calibration** via validation-fitted Temperature Scaling.
4. **Predictive uncertainty quantification** via Monte Carlo Dropout (20 forward passes).
5. **Attribution sensitivity analysis** via Grad-CAM and quantitative 40% top-saliency deletion perturbation.
6. **External testing** on smartphone-acquired clinical photography (PAD-UFES-20) to quantify real-world domain transfer.

---

## Research Question
Can an integrated multi-scale attention framework provide robust discrimination on class-imbalanced dermoscopic datasets while maintaining probability calibration, quantifiable uncertainty, and transparent performance boundaries under out-of-distribution optical shift?

---

## Key Contributions
- **Rigorous Leakage Elimination**: Cryptographic SHA-256 deduplication removed 7,697 exact duplicate images across datasets. A local audit of BCN20000 identified 7,690 byte-identical images overlapping with HAM10000, prompting its complete exclusion from the development corpus.
- **Hierarchical Group Splitting**: Group-aware partitioning strictly prevented train/validation/test leakage across 30,495 unique patient/lesion groups.
- **Calibrated Decision Thresholding**: Temperature scaling ($T = 1.1799$) reduced the internal Expected Calibration Error (ECE) to 0.0054.
- **External Shift Characterization**: Documented the quantitative performance drop when transferring dermoscopy-trained representations to smartphone clinical photography.

---

## Pipeline Architecture
```
Raw Image Ingestion (ISIC 2024, HAM10000, PAD-UFES-20, BCN20000 Audit)
                           │
             SHA-256 Exact Deduplication (7,697 duplicates removed)
                           │
       Group-Aware Splitting (70% Train, 15% Val, 15% Internal Test)
                           │
              Data Augmentation + WeightedRandomSampler
                           │
       HELPix-R Architecture (EfficientNet-B0 + 5-Scale SE Attention)
                           │
           Focal Loss (α = 1.0, γ = 2.0) + AdamW Optimization
                           │
             Temperature Scaling Calibration (Fitted on Val)
                           │
     ┌─────────────────────┴────────────────────────┐
     ▼                                              ▼
Internal Dermoscopic Testing (N = 55,130)   External Smartphone Testing (N = 654)
AUROC: 0.9778 | F1: 0.4433 | ECE: 0.0054     AUROC: 0.6451 | F1: 0.5882 | ECE: 0.1650
```

---

## Datasets
The benchmark incorporates four major dermatology repositories:

| Dataset | Modality | Raw Records | Retained Images | Unique Groups | Role in Study |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ISIC 2024 SLICE-3D** | Dermoscopy (Polarized / Non-polarized) | 401,059 | 401,059 | 28,383 | Development (Train / Val / Internal Test) |
| **HAM10000** | Multi-center Dermoscopy | 10,015 | 10,013 | 7,470 | Development (Train / Val / Internal Test) |
| **BCN20000** | Dermoscopy (Clinic Barcelona) | 10,015 | 0 | Overlaps HAM | Excluded (7,690 local duplicates of HAM10000) |
| **PAD-UFES-20** | Smartphone Clinical Photography | 2,298 | 654 | 526 | External Test (Out-of-Distribution Shift) |
| **Total Development** | Dermoscopic Corpus | 411,074 | **411,072** | **30,495** | Leakage-Controlled Development |
| **Total Benchmark** | Multi-Source Cohort | 423,387 | **411,726** | **36,379** | Complete Benchmark Cohort |

> **Local Audit Note regarding BCN20000**: The finding that 7,690 BCN20000 images were byte-identical to HAM10000 images is strictly a finding from the local repository audit. It does not make a general claim regarding complete external BCN20000 archives.

---

## Dataset Curation
- **SHA-256 Exact Deduplication**: Analyzed across all image files in 64 KB memory-safe chunks. Removed 2 intra-HAM10000 duplicates, 5 intra-PAD-UFES-20 duplicates, and 7,690 local BCN20000/HAM10000 overlaps.
- **Group Key Hierarchy**: Priority set to `lesion_id` $\rightarrow$ `patient_id` $\rightarrow$ `image_id`.
- **Zero Leakage**: All splits assert mutually disjoint group sets ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Optional Quality Gate**: The repository includes an optional photographic quality evaluation module (`src/data/quality_gate.py`), but it was **not** part of the reported final training pipeline.

---

## Model Architecture
- **Backbone**: Torchvision EfficientNet-B0 extracting intermediate features at 5 distinct receptive scales:
  - C1: $112 \times 112$, 16 channels
  - C2: $56 \times 56$, 24 channels
  - C3: $28 \times 28$, 40 channels
  - C4: $14 \times 14$, 112 channels
  - C5: $7 \times 7$, 320 channels
- **Attention**: Independent Squeeze-and-Excitation (SE) blocks on each scale (reduction ratio = 16).
- **Fusion**: Adaptive Average Pooling to $1 \times 1$ per scale, concatenated to form a 512-dimensional unified representation.
- **Classifier**: $\text{Dropout}(0.3) \rightarrow \text{Linear}(512 \rightarrow 256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256 \rightarrow 2)$.

---

## Training
- **Loss**: Focal Loss ($\alpha = 1.0, \gamma = 2.0$) to counter severe negative-class imbalance.
- **Optimization**: AdamW ($\text{lr} = 10^{-4}, \text{weight decay} = 10^{-4}$), ReduceLROnPlateau ($\text{factor} = 0.5, \text{patience} = 2$), early stopping ($\text{patience} = 3$).
- **Hardware Acceleration**: Automated Mixed Precision (AMP FP16) on GPU.

---

## Evaluation & Calibration
- **Calibration Protocol**: Temperature scaling parameter $T = 1.1799$ was optimized strictly on the validation cohort using L-BFGS and CrossEntropyLoss. Calibration was never fitted on test cohorts.
- **Uncertainty**: 20 stochastic forward passes via MC-Dropout during evaluation.
- **Statistical Significance**: 1,000-iteration non-parametric bootstrap estimation for 95% confidence intervals.

---

## External Testing & Domain Shift
Testing on the PAD-UFES-20 smartphone dataset ($N = 654$) was conducted without retraining or adaptation:
- Internal AUROC: 0.9778 vs External AUROC: 0.6451 ($\Delta\text{AUROC} = -0.3327$)
- Internal ECE: 0.0054 vs External ECE: 0.1650 ($\Delta\text{ECE} = +0.1596$)

> **Scientific Reporting Standard**:
> Under external smartphone-domain testing on PAD-UFES-20, model discrimination decreased by 0.3327 absolute AUROC points ($\Delta\text{AUROC} = -0.3327$), accompanied by an increase in ECE of 0.1596 absolute points ($\Delta\text{ECE} = +0.1596$).

---

## Explainability
Grad-CAM was implemented on the penultimate feature layer (`features[-2]`). Quantitative 40% top-saliency perturbation deletion tests produced:
- Mean class probability reduction: **0.0723**
- Median class probability reduction: **0.0487**

> **Attribution Note**: The perturbation analysis evaluates attribution sensitivity and does not establish clinical reasoning or causal biological validity.

---

## Results Summary

### Internal Test Set (N = 55,130) vs External Test Set (N = 654)
| Metric | Internal Test (Dermoscopic) | External Test (PAD-UFES-20 Smartphone) | Shift Interpretation |
| :--- | :---: | :---: | :--- |
| **AUROC** | **0.9778** [0.9666, 0.9866] | **0.6451** | $\Delta\text{AUROC} = -0.3327$ |
| **Sensitivity** | 0.7865 | 0.4808 | $\Delta\text{Sensitivity} = -0.3057$ |
| **Specificity** | 0.9910 | 0.7311 | $\Delta\text{Specificity} = -0.2599$ |
| **F1-Score** | 0.4433 | 0.5882 | Modality shift in class balance |
| **ECE (Post-Cal)**| **0.0054** | **0.1650** | $\Delta\text{ECE} = +0.1596$ |
| **Brier Score** | 0.0149 | 0.5641 | Increased calibration divergence |

### Baseline Comparison (Internal Test Cohort)
| Architecture | AUROC | F1-Score | Note |
| :--- | :---: | :---: | :--- |
| ResNet50 | 0.9771 | 0.2650 | Standard residual baseline |
| EfficientNet-B0 | 0.9757 | 0.2412 | Backbone alone without multi-scale attention |
| **MobileNetV3-Large** | **0.9805** | 0.3284 | **Highest reported AUROC** |
| ConvNeXt-Tiny | 0.9751 | 0.3658 | Modernized convnet baseline |
| **HELPix-R (Proposed)** | 0.9778 | **0.4433** | **Highest reported F1-score** |

*HELPix-R achieved the highest F1-score among the evaluated models, with sensitivity of 0.7865 and specificity of 0.9910 on the internal test set.*

---

## Repository Structure
```
HELPix-R/
├── configs/               # YAML configuration files (dataset, training, evaluation, calibration)
├── src/                   # Core Python modular library
│   ├── data/              # Lazy dataset, transforms, deduplication, group-aware splitting, quality gate
│   ├── models/            # HELPix-R, SE attention, backbones, baselines
│   ├── training/          # Focal loss, AMP training loop
│   ├── calibration/       # Temperature scaling
│   ├── evaluation/        # Diagnostic metrics, bootstrap CI
│   ├── uncertainty/       # MC-Dropout
│   ├── explainability/    # Grad-CAM, perturbation deletion
│   └── utils/             # Seeds, YAML config loader
├── scripts/               # Executable command-line reproduction scripts
├── notebooks/             # Educational demonstration Jupyter notebooks (01 to 07)
├── results/               # Audited scientific result artifacts
│   ├── tables/            # Tables 1, calibration, baselines, splits, deduplication, domain shift
│   ├── figures/           # Figures 1 to 5 (PNG format)
│   └── metrics/           # Final metrics CSV
├── docs/                  # Dataset card, Model card, Reproducibility, Discrepancies, BMC text
└── tests/                 # Comprehensive unit test suite
```

---

## Installation
```bash
git clone https://github.com/CSrajput-ux/Helpix-reasearh-.git
cd HELPix-R
pip install -r requirements.txt
```

---

## Reproduction
To reproduce the full evaluation suite from preserved artifacts:
```bash
# 1. Audit dataset characteristics
python scripts/audit_dataset.py

# 2. Review deduplication audit
python scripts/deduplicate.py

# 3. View group-aware split summary
python scripts/create_splits.py

# 4. View diagnostic metrics and baseline comparisons
python scripts/evaluate.py

# 5. View calibration pre/post comparison
python scripts/calibrate.py

# 6. View external domain shift analysis
python scripts/external_test.py

# 7. View bootstrap confidence intervals
python scripts/bootstrap_ci.py

# 8. View Grad-CAM perturbation deletion results
python scripts/generate_gradcam.py

# 9. Generate figures
python scripts/generate_figures.py
```

---

## Software Verification vs. Empirical Reproduction

### Software Verification
- **Automated Test Suite**: 18/18 unit and integration tests passed (`pytest tests/ -v`).
- This verifies that data loading, group-aware partitioning, neural network components (SE attention, EfficientNet backbone, HELPix-R head), training loops, focal loss, temperature scaling, and metric algorithms execute properly and without programmatic errors.
- **Important Note**: Automated test passage verifies software correctness, mathematical mechanics, and interface stability; it does **not** imply that the entire 411,072-image training run was re-executed from scratch during the test suite.

### Empirical Result Artifacts
- The reported scientific metrics, tables, and figures in the manuscript are preserved as audited research artifacts in `results/tables/`, `results/metrics/`, and `results/figures/`.
- Every empirical finding is mapped to its original execution script, configuration, and data sources as documented in [RESULTS_MAP.md](docs/RESULTS_MAP.md) and [IMPLEMENTATION_PROVENANCE.md](docs/IMPLEMENTATION_PROVENANCE.md).

---

## Model Checkpoints & Archival Strategy
- The trained HELPix-R model checkpoint (`helpix_r_best.pt`, 16.9 MB) and baseline checkpoints are not tracked directly in Git to prevent repository bloat and maintain lightweight cloning.
- **Release Strategy**: The final model weights (`helpix_r_best.pt`) will be archived alongside the release code on Zenodo under a persistent DOI upon publication. Users wishing to evaluate the pre-trained model without retraining can download the checkpoint from Zenodo and place it in `results/` or `models/`.
- **Retraining Option**: The complete pipeline can be retrained from scratch using `scripts/train.py --config configs/training_config.yaml --model helpix_r`.

---

## Data Access
Raw image datasets are subject to third-party distribution terms and are not redistributed directly with this repository. Please consult [DATASET_CARD.md](docs/DATASET_CARD.md) and [BMC_DATA_AVAILABILITY.md](docs/BMC_DATA_AVAILABILITY.md) for direct acquisition links.

---

## Code Availability
Please see [BMC_CODE_AVAILABILITY_PREPUBLICATION.md](docs/BMC_CODE_AVAILABILITY_PREPUBLICATION.md) and [BMC_CODE_AVAILABILITY_FINAL.md](docs/BMC_CODE_AVAILABILITY_FINAL.md) for official manuscript paragraphs.

---

## Limitations
Please review [LIMITATIONS.md](docs/LIMITATIONS.md) for scientific boundary conditions, demographic considerations, and non-diagnostic disclaimers.

---

## Citation
Please see [CITATION.cff](CITATION.cff) for citation metadata format.

---

## License
The software license for this research codebase is documented in [LICENSE](LICENSE).

**The repository license applies only to the original software/code contained in this repository and does not grant rights to redistribute or reuse third-party datasets.**
Third-party datasets used in this study (ISIC 2024, HAM10000, PAD-UFES-20, BCN20000) remain subject to their respective original licenses and access terms.

