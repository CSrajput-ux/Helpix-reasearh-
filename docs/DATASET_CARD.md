# Dataset Card: HELPix-R Skin Lesion Benchmark Cohorts

## 1. Overview and Scope
The HELPix-R research framework integrates multi-source dermoscopic and clinical photograph datasets to benchmark reliability, calibration, and domain transfer in binary skin lesion malignancy classification.

| Dataset | Modality | Raw Records | Retained Images | Unique Groups | Malignant | Non-Malignant | Role in Study |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ISIC 2024 SLICE-3D** | Dermoscopy (Polarized / Non-polarized) | 401,059 | 401,059 | 28,383 | 393 | 400,666 | Development (Train / Val / Internal Test) |
| **HAM10000** | Dermoscopy (Multi-center Vienna & Graz) | 10,015 | 10,013 | 7,470 | 1,627 | 8,386 | Development (Train / Val / Internal Test) |
| **BCN20000 (Audit)** | Dermoscopy (Clinic Barcelona) | 10,015 | 0 | Overlaps HAM | Overlaps HAM | Overlaps HAM | **Excluded** (7,690 exact duplicates of HAM10000 in local audit) |
| **PAD-UFES-20** | Smartphone Clinical Photography | 2,298 | 654 | 526 | 416 | 238 | **External Test** (Out-of-Distribution Shift) |
| **Total Development** | Dermoscopic | 411,074 | **411,072** | **35,853** | 2,020 | 409,052 | 70% Train / 15% Val / 15% Internal Test |
| **Total Benchmark** | Harmonized Multi-Modal | 423,387 | **411,726** | **36,379** | 2,436 | 409,290 | Complete Evaluated Cohort |

---

## 2. Dataset Descriptions and Sources

### ISIC 2024 SLICE-3D
- **Source**: International Skin Imaging Collaboration (ISIC) 2024 Challenge / SLICE-3D 3D-TBP dataset.
- **Acquisition**: Total-body photography crop sequences capturing high-resolution dermoscopic and close-up views.
- **Labels**: Biopsy-verified histopathology ground truth and expert consensus for benign non-malignant controls.
- **Access / License**: Publicly accessible via the ISIC Archive under CC0 / Creative Commons terms.

### HAM10000 ("Human Against Machine with 10000 training images")
- **Source**: Medical University of Vienna and Cliff Rosendahl clinical practice (Queensland, Australia).
- **Acquisition**: Multisource dermatoscopic images collected over 20 years using diverse dermatoscopes.
- **Preprocessing & Deduplication**: Cryptographic SHA-256 analysis identified 2 exact duplicate image pairs within HAM10000, which were removed, retaining 10,013 images.
- **Access / License**: Harvard Dataverse under Creative Commons Attribution-NonCommercial (CC BY-NC 4.0).

### BCN20000 Local Repository Audit and Exclusion Rationale
- **Source**: Hospital Clínic de Barcelona, Spain.
- **Audit Findings**: The local repository environment contained 7,704 image files corresponding to 10,015 metadata rows. Cryptographic SHA-256 hashing identified 7,690 unique hashes, all of which were byte-identical duplicates of images present in HAM10000.
- **Scientific Decision**: BCN20000 was completely excluded from the training and development corpus to prevent duplicate bias and over-representation.
- **Important Boundary Condition**: This exclusion is strictly bounded to the local repository audit and does not constitute a global claim regarding the complete external BCN20000 archive.

### PAD-UFES-20 (External Test Set)
- **Source**: Federal University of Espírito Santo (UFES), Dermatological Outpatient Service, Brazil.
- **Modality**: Smartphone clinical photography acquired via various consumer smartphones under ambient lighting, capturing real-world primary-care presentations.
- **Audit Findings**: The local research repository contained 659 images from the 2,298-record archive. Cryptographic hashing identified 5 duplicate files, yielding 654 unique images.
- **Role**: Held strictly as an out-of-distribution external test cohort without retraining or fine-tuning.
- **Access / License**: Mendeley Data under CC BY 4.0.

---

## 3. Class Harmonization
All diagnostic labels are mapped to a binary classification task:
- **Class 1 (Malignant)**: Melanoma (`MEL`), Basal Cell Carcinoma (`BCC`), Squamous Cell Carcinoma (`SCC`).
- **Class 0 (Non-Malignant)**: Nevus (`NV`), Benign Keratosis (`BKL`), Actinic Keratosis / Bowen's (`AK`), Dermatofibroma (`DF`), Vascular Lesion (`VASC`).

---

## 4. Leakage Prevention Protocol
Data splitting was executed via group-aware `GroupShuffleSplit`:
- **Group Key Priority**: `lesion_id` $\rightarrow$ `patient_id` $\rightarrow$ `image_id` fallback.
- **Guarantee**: Zero patient or lesion overlap exists across Train (303,419 images), Validation (52,523 images), and Internal Test (55,130 images).

---

## 5. No Data Redistribution Notice
Raw image files are the proprietary property of their respective originating consortia and are governed by third-party licenses. Users must download the datasets directly from official sources following the instructions in `REPRODUCIBILITY.md`.
