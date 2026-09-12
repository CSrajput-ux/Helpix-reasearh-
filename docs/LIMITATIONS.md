# Scientific Limitations and Boundary Conditions

This document outlines the methodological, photographic, algorithmic, and translational limitations of the HELPix-R study.

---

## 1. Modality and Domain Shift
When transferring from dermoscopic images (ISIC 2024 SLICE-3D and HAM10000) to smartphone clinical photography (PAD-UFES-20) without fine-tuning:
- Model discrimination decreased by 0.3327 absolute AUROC points ($\Delta\text{AUROC} = -0.3327$, from 0.9778 to 0.6451).
- Expected Calibration Error increased by 0.1596 absolute points ($\Delta\text{ECE} = +0.1596$, from 0.0054 to 0.1650).
- False positive control (specificity) dropped from 0.9910 to 0.7311.

**Implication**: Models trained exclusively on polarized and non-polarized dermoscopy cannot be deployed directly into primary care smartphone triaging without domain adaptation or dedicated clinical photography training data.

---

## 2. Demographic and Phenotypic Representation
Public skin lesion archives predominantly reflect light-skinned patient cohorts (Fitzpatrick skin phototypes I–III from European, North American, and Australian centers). Darker skin types (Fitzpatrick IV–VI) are substantially under-represented in the training corpus. Consequently, the generalizability of HELPix-R across diverse global populations remains an open research challenge.

---

## 3. Explainability and Attribution Sensitivity
- Grad-CAM heatmaps highlight feature activation regions that correlate with model decision boundaries.
- Quantitative perturbation testing (40% top-saliency deletion) confirmed attribution sensitivity (mean probability reduction: 0.0723).
- **Critical Caveat**: Feature saliency does **NOT** equate to biological, histopathological, or clinical reasoning. Saliency maps must not be interpreted as causal clinical explanations.

---

## 4. Quality Filtering Scope
While an automated image quality gate was implemented as an optional modular utility (`src/data/quality_gate.py`), **it was not part of the reported final training pipeline**. All reported model results reflect the full, unpruned development splits.

---

## 5. Retrospective Study Design
All evaluations in this study were conducted on retrospective benchmark datasets. Prospective clinical trials in real-world clinical workflows with consecutive patient presentations are required before any automated diagnostic tool can be safely integrated into medical practice.

---

## 6. Non-Diagnostic Disclaimer
HELPix-R is a research software framework for methodological analysis. It is **not** an approved diagnostic medical device and must **not** be used for clinical decision-making or patient diagnosis.
