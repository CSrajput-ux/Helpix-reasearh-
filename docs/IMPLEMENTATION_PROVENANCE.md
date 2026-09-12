# Implementation Provenance and Verification Record

This document provides a strict, traceable mapping between the original research implementation scripts, functions, checkpoints, and result artifacts, and the modular publication repository components.

---

## Provenance Matrix

| Research Component | Original Script & Function/Class | Publication Repository Component | Verified Equivalent? | Evidence Artifact |
| :--- | :--- | :--- | :---: | :--- |
| **Exact Hash Deduplication** | `run_helpix_pipeline.py`<br>`sha256_of_file()` (L128–140)<br>`deduplicate_dataset()` (L306–366) | `src/data/deduplication.py`<br>`scripts/deduplicate.py` | **YES** | `results/tables/deduplication_statistics.csv`<br>(7,697 duplicates removed, 0 phash) |
| **Group-Aware Splitting** | `run_helpix_pipeline.py`<br>`perform_grouped_split()` (L369–428)<br>`get_group()` (L372–378) | `src/data/splitting.py`<br>`scripts/create_splits.py` | **YES** | `results/tables/final_split_statistics.csv`<br>(Train: 303,419 / Val: 52,523 / Internal Test: 55,130 / External Test: 654) |
| **Dataset & Augmentation** | `run_helpix_pipeline.py`<br>`SkinLesionDataset` (L433–452)<br>`get_dataloaders()` (L454–495) | `src/data/dataset.py`<br>`src/data/transforms.py` | **YES** | ImageNet norm, RandomHorizontalFlip, RandomVerticalFlip, Rotation(20), ColorJitter, WeightedRandomSampler |
| **Optional Quality Gate** | *Not in final pipeline* | `src/data/quality_gate.py` | **OPTIONAL** | Documented as optional research utility not included in final training corpus |
| **HELPix-R Architecture** | `run_helpix_pipeline.py`<br>`SEBlock` (L514–532)<br>`TVEfficientNetFeatureExtractor` (L534–551)<br>`HELPixR` (L553–585) | `src/models/helpix_r.py`<br>`src/models/se_attention.py`<br>`src/models/backbones.py` | **YES** | `d:\Helpix_data\HELPix-Research\models\helpix_r_best.pt`<br>16,935,131 bytes |
| **Baseline Architectures** | `run_helpix_pipeline.py`<br>`create_baseline_model()` (L974–996) | `src/models/baselines.py` | **YES** | `d:\Helpix_data\HELPix-Research\models\`<br>`resnet50_best.pt`, `efficientnet_b0_best.pt`, `mobilenetv3_best.pt`, `convnext_tiny_best.pt` |
| **Loss & Training Engine** | `run_helpix_pipeline.py`<br>`FocalLoss` (L500–512)<br>`train_epoch` (L610–641)<br>`fit_model` (L665–714) | `src/training/losses.py`<br>`src/training/trainer.py`<br>`scripts/train.py` | **YES** | AdamW ($\text{lr}=10^{-4}, \text{wd}=10^{-4}$), Focal Loss ($\gamma=2.0, \alpha=1.0$), ReduceLROnPlateau, AMP FP16 |
| **Temperature Scaling Calibration** | `run_helpix_pipeline.py`<br>`TemperatureScaler` (L719–727)<br>`fit_temperature()` (L729–753) | `src/calibration/temperature_scaling.py`<br>`scripts/calibrate.py` | **YES** | `results/tables/Table_calibration_comparison.csv`<br>Fitted $T = 1.1799$ on validation set |
| **Internal Test Evaluation** | `run_helpix_pipeline.py`<br>`calculate_metrics()` (L778–806)<br>`eval_epoch()` (L644–663) | `src/evaluation/metrics.py`<br>`scripts/evaluate.py` | **YES** | `results/metrics/final_metrics.csv`<br>AUROC: 0.9778, Sens: 0.7865, Spec: 0.9910, F1: 0.4433, ECE: 0.0054, Brier: 0.0149 |
| **Non-Parametric Bootstrap CI** | `run_helpix_pipeline.py`<br>`bootstrap_auroc()` (L808–828) | `src/evaluation/bootstrap.py`<br>`scripts/bootstrap_ci.py` | **YES** | 1,000 resamples, seed=42; 95% CI: [0.9666, 0.9866] |
| **External Testing (PAD-UFES-20)** | `run_helpix_pipeline.py`<br>`get_eval_data(loaders['ext'])` (L1060–1076) | `scripts/external_test.py` | **YES** | `results/metrics/final_metrics.csv`<br>AUROC: 0.6451, ECE: 0.1650, Sens: 0.4808, Spec: 0.7311, F1: 0.5882, Brier: 0.5641 |
| **Domain Shift Analysis** | `run_helpix_pipeline.py`<br>`domain_shift` dict (L1068–1071) | `src/evaluation/metrics.py` | **YES** | `results/tables/domain_shift_analysis.csv`<br>$\Delta\text{AUROC} = -0.3327$, $\Delta\text{ECE} = +0.1596$ |
| **Uncertainty Estimation (MC-Dropout)** | `run_helpix_pipeline.py`<br>`mc_dropout_predict()` (L586–605) | `src/uncertainty/mc_dropout.py` | **YES** | 20 forward passes with stochastic dropout enabled, BatchNorm in eval mode |
| **Grad-CAM & Perturbation Deletion** | `run_helpix_pipeline.py` (L833–881)<br>`generate_paper_needs.py` (L113–263) | `src/explainability/gradcam.py`<br>`src/explainability/perturbation.py`<br>`scripts/generate_gradcam.py`<br>`scripts/generate_figures.py` | **YES** | `results/tables/gradcam_deletion_results.csv`<br>Top-40% salient region masked: Mean drop: 0.0723, Median drop: 0.0487 |
| **Figure Generation** | `generate_paper_needs.py`<br>`run_helpix_pipeline.py` (L886–927) | `scripts/generate_figures.py` | **YES** | Figures 1 to 5 in `results/figures/` (Figure 5 title corrected to mathematical clarity) |

---

## Verification Statement

All core algorithmic implementations in `src/` either directly mirror or faithfully wrap the audited research routines in `run_helpix_pipeline.py` and `generate_paper_needs.py`. No mathematical operations, network topologies, loss formulations, or evaluation criteria have been modified.
