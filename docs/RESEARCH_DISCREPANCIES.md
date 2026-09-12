# Research Integrity and Transparency Record: Clarifications and Discrepancies

This document transparently records methodological nuances, legacy drafting discrepancies, and precision corrections resolved in the publication repository.

---

## 1. Architectural Metric Ranking Clarification
- **Observation**: In some preliminary project notes, HELPix-R was colloquially described as "outperforming" all baselines.
- **Audited Empirical Finding**:
  - `MobileNetV3-Large` achieved the highest reported AUROC (**0.9805**).
  - `HELPix-R` achieved an AUROC of **0.9778** and the highest reported F1-score (**0.4433**), providing a balance of sensitivity (0.7865) and specificity (0.9910) under severe class imbalance.
- **Resolution**: All publication documents explicitly report MobileNetV3-Large as having the highest AUROC, and HELPix-R as achieving the highest F1-score. Qualitative claims of "balanced sensitivity/specificity" have been removed.

---

## 2. Figure 5 Title Labeling Correction
- **Observation**: In legacy visualization code (`generate_paper_needs.py`, line 249), the deletion title was formatted as:
  ```python
  axes[i, 3].set_title(f"Drop: -{drop:.3f} ({(drop/max(orig_p,1e-4))*100:.1f}%)")
  ```
  Because `drop = orig_p - masked_p` was already positive (e.g., $0.959$), prefixing a minus sign displayed `"Drop: -0.959"`, which could be misread as an increase rather than a drop.
- **Resolution**: In the publication figure generator (`scripts/generate_figures.py`), this has been corrected to mathematically unambiguous labeling:
  ```python
  axes[i, 3].set_title(f"Confidence reduction: {drop:.3f} ({(drop/max(orig_p,1e-4))*100:.1f}%)")
  ```
  or signed differential $\Delta P = P_{\text{masked}} - P_{\text{orig}}$.

---

## 3. Dataset Naming Precision
- **Observation**: Preliminary scratch files occasionally used colloquial or legacy shorthand such as `SLIC-30` or `ISIC 2019`.
- **Resolution**: Standardized across all documents to the formal title: **ISIC 2024 SLICE-3D** (referencing the 401,059 crop sequence).

---

## 4. BCN20000 Scope of Audit
- **Observation**: Local deduplication found 7,690 BCN20000 images to be byte-identical to HAM10000 images by SHA-256.
- **Resolution**: This finding is explicitly described as a finding of the local repository audit. The text explicitly refrains from claiming that all global BCN20000 archives are duplicates.

---

## 5. Quality Gate Usage
- **Observation**: A quality filter utility evaluating Laplacian blur and brightness exists in the codebase.
- **Resolution**: Explicitly documented that this quality gate was **not** applied to the reported final training pipeline. All reported results reflect the complete, unpruned development cohort.
