# Results and Artifact Mapping Matrix

This document provides a 1-to-1 mapping between every table, metric, and figure presented in the *BMC Medical Imaging* manuscript and its underlying data artifact and generation script.

---

## 1. Tables to Source Artifacts

| Manuscript Table | Description | Underlying Data Artifact | Generating Script |
| :--- | :--- | :--- | :--- |
| **Table 1** | Multi-Cohort Dataset Characteristics | `results/tables/Table_1_dataset_characteristics.csv` | `scripts/audit_dataset.py` |
| **Table 2** | Baseline Model vs HELPix-R Performance | `results/tables/baseline_comparison.csv` | `scripts/evaluate.py` |
| **Table 3** | Pre- vs Post-Calibration (ECE & Brier) | `results/tables/Table_calibration_comparison.csv` | `scripts/calibrate.py` |
| **Table 4** | Final Diagnostic Metrics (Internal & External) | `results/metrics/final_metrics.csv` | `scripts/evaluate.py` |
| **Split Table** | Group-Aware Data Partition Summary | `results/tables/final_split_statistics.csv` | `scripts/create_splits.py` |
| **Dedup Table** | Cryptographic SHA-256 Deduplication Audit | `results/tables/deduplication_statistics.csv` | `scripts/deduplicate.py` |
| **Domain Shift**| Absolute Shift on PAD-UFES-20 | `results/tables/domain_shift_analysis.csv` | `scripts/external_test.py` |
| **Saliency Deletion**| Quantitative 40% Deletion Attribution | `results/tables/gradcam_deletion_results.csv` | `scripts/generate_gradcam.py` |

---

## 2. Figures to Visual Artifacts

| Manuscript Figure | Description | Rendered Visual Artifact | Generating Script |
| :--- | :--- | :--- | :--- |
| **Figure 1** | HELPix-R Architecture & Feature Fusion Diagram | `results/figures/Figure_1_HELPix_Architecture.png` | `scripts/generate_figures.py` |
| **Figure 2** | ROC Curves: Internal vs External Test Set | `results/figures/Figure_2_ROC_Internal_vs_External.png` | `scripts/generate_figures.py` |
| **Figure 3** | Calibration Reliability Diagram (Internal Test) | `results/figures/Figure_3_Calibration_Internal.png` | `scripts/generate_figures.py` |
| **Figure 4** | Calibration Reliability Diagram (PAD-UFES-20) | `results/figures/Figure_4_Calibration_External.png` | `scripts/generate_figures.py` |
| **Figure 5** | Grad-CAM Attribution Heatmaps & Deletion Sensitivity | `results/figures/Figure_5_GradCAM_Heatmaps.png` | `scripts/generate_figures.py` |

---

## 3. Metric Verification Checksum
All numerical metrics cited in the text match the records in `final_metrics.csv`:
- **Internal Test AUROC**: `0.977839...` $\approx 0.9778$
- **Internal Test Sensitivity**: `0.786476...` $\approx 0.7865$
- **Internal Test Specificity**: `0.990975...` $\approx 0.9910$
- **Internal Test F1**: `0.443329...` $\approx 0.4433$
- **Internal Test ECE**: `0.005418...` $\approx 0.0054$
- **Internal Test Brier**: `0.014897...` $\approx 0.0149$
- **External Test AUROC**: `0.645058...` $\approx 0.6451$
- **External Test ECE**: `0.165008...` $\approx 0.1650$
