# Experimental Register and Benchmark Documentation

This document logs all experiments conducted in the HELPix-R research framework, detailing experimental objectives, datasets, architectures, execution status, and corresponding evidence artifacts.

---

## Complete Experiment Registry

| Experiment ID | Dataset / Cohort | Research Purpose | Model Architecture | Training Status | Evaluation Status | Primary Result Artifact | Paper Figure / Table |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **EXP-01** | ISIC 2024 + HAM10000 | Baseline Benchmark | ResNet50 | Completed | Completed | `results/tables/baseline_comparison.csv` | Table 2 (Baseline comparison) |
| **EXP-02** | ISIC 2024 + HAM10000 | Baseline Benchmark | EfficientNet-B0 | Completed | Completed | `results/tables/baseline_comparison.csv` | Table 2 (Baseline comparison) |
| **EXP-03** | ISIC 2024 + HAM10000 | Baseline Benchmark | MobileNetV3-Large | Completed | Completed | `results/tables/baseline_comparison.csv` | Table 2 (Highest AUROC: 0.9805) |
| **EXP-04** | ISIC 2024 + HAM10000 | Baseline Benchmark | ConvNeXt-Tiny | Completed | Completed | `results/tables/baseline_comparison.csv` | Table 2 (Baseline comparison) |
| **EXP-05** | ISIC 2024 + HAM10000 | Proposed Framework | HELPix-R | Completed | Completed | `results/metrics/final_metrics.csv` | Table 2 (Highest F1: 0.4433), Fig 1, Fig 2 |
| **EXP-06** | Validation Split ($N = 52,523$) | Post-hoc Calibration | HELPix-R + Temp Scaling | Completed | Completed | `results/tables/Table_calibration_comparison.csv` | Table 3, Fig 3 ($T = 1.1799$) |
| **EXP-07** | Internal Test ($N = 55,130$) | Diagnostic Evaluation | Calibrated HELPix-R | Completed | Completed | `results/metrics/final_metrics.csv` | Table 3, Fig 2 (AUROC: 0.9778) |
| **EXP-08** | Internal Test ($N = 55,130$) | Statistical Stability | Calibrated HELPix-R | Completed | Completed | `results/metrics/final_metrics.csv` | 95% Bootstrap CI: [0.9666, 0.9866] |
| **EXP-09** | Internal Test ($N = 55,130$) | Predictive Uncertainty | HELPix-R (MC-Dropout) | Completed | Completed | `results/tables/final_metrics.csv` | 20 Stochastic Forward Passes |
| **EXP-10** | PAD-UFES-20 ($N = 654$) | External Domain Shift | Calibrated HELPix-R | Untrained | Completed | `results/tables/domain_shift_analysis.csv` | Table 3, Fig 2, Fig 4 (AUROC: 0.6451) |
| **EXP-11** | Internal Test Subsets | Attribution Attribution | HELPix-R + Grad-CAM | — | Completed | `results/figures/Figure_5_GradCAM_Heatmaps.png` | Figure 5 (Attribution Maps) |
| **EXP-12** | Internal Test Subsets | Attribution Sensitivity | HELPix-R + 40% Deletion | — | Completed | `results/tables/gradcam_deletion_results.csv` | Table 4, Fig 5 (Mean Drop: 0.0723) |

---

## Detailed Experimental Findings

### 1. Architectural Comparison
- **MobileNetV3-Large**: Demonstrated the highest discrimination capability in terms of raw AUROC (0.9805).
- **HELPix-R**: Achieved the highest F1-score (0.4433) among evaluated models, maintaining a sensitivity of 0.7865 and specificity of 0.9910 under severe negative-class dominance.

### 2. Post-Hoc Calibration
- Fitting $T = 1.1799$ on the validation cohort reduced the internal test Expected Calibration Error from 0.0076 to 0.0054 (Brier score: 0.0152 to 0.0149).
- On the external smartphone cohort, calibration improved ECE from 0.1782 to 0.1650 (Brier score: 0.5824 to 0.5641).

### 3. External Testing on PAD-UFES-20
Under external smartphone-domain testing on PAD-UFES-20 without retraining, model discrimination decreased by 0.3327 absolute AUROC points ($\Delta\text{AUROC} = -0.3327$), accompanied by an increase in ECE of 0.1596 absolute points ($\Delta\text{ECE} = +0.1596$).
