# Data Curation and Leakage-Controlled Engineering Protocol

## 1. Objectives of Curation
Modern machine learning models for dermatological imaging are highly susceptible to:
1. **Identical Byte Duplication**: Cross-dataset ingestion of identical images leading to artificial inflation of performance.
2. **Patient and Lesion Leakage**: Multiple images of the same lesion or patient partitioned across both training and evaluation subsets, leading to memorization rather than generalization.
3. **Artifact and Modality Bias**: Dermoscopic versus clinical photographic distribution shifts.

---

## 2. Cryptographic Deduplication Protocol

### SHA-256 Exact Hash Computation
Every image file across the raw directories was hashed using cryptographic SHA-256 in 64 KB memory-safe chunks:
```python
def sha256_of_file(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
```

### Exact Duplicate Removal Audit
| Verification Level | Candidate Scope | Method | Duplicates Removed | Impact / Action |
| :--- | :--- | :--- | :---: | :--- |
| **Intra-HAM10000** | HAM10000 archive | SHA-256 | **2** | Removed redundant duplicate images (retaining 10,013) |
| **Intra-PAD-UFES-20** | PAD-UFES-20 local files | SHA-256 | **5** | Removed duplicate files (retaining 654) |
| **Cross-Dataset: BCN vs HAM** | Local BCN20000 vs HAM10000 | SHA-256 | **7,690** | BCN20000 excluded from development corpus |
| **Perceptual Hash (pHash)** | Candidate near-duplicates | Hamming dist $\le 4$ | **0** | No visually distinct near-duplicates warranted removal |
| **Total Duplicates Removed** | Entire Corpus | Cryptographic SHA-256 | **7,697** | Fully audited and documented |

> **Important Scientific Scope Note regarding BCN20000**:
> The finding that 7,690 BCN20000 images were byte-identical to HAM10000 images is strictly a finding from the local repository audit. It does not make a general or global claim regarding complete external BCN20000 repositories distributed elsewhere.

---

## 3. Group-Aware Data Splitting Protocol

### Group Key Priority
To strictly eliminate data leakage across evaluation boundaries, samples were grouped prior to splitting according to the following hierarchical priority:
$$\text{Group Key} = \begin{cases} \text{dataset} \parallel \text{lesion\_id} & \text{if } \text{lesion\_id} \text{ is valid} \\ \text{dataset} \parallel \text{patient\_id} & \text{if } \text{patient\_id} \text{ is valid} \\ \text{dataset} \parallel \text{image\_id} & \text{fallback} \end{cases}$$

### Target and Realized Partition Statistics
Target ratio on the development cohort: **70% Train, 15% Validation, 15% Internal Test**.
The external test cohort (**PAD-UFES-20**) was held completely disjoint and untouched during training and calibration.

| Partition | Images (N) | % of Development | Unique Patient/Lesion Groups | Zero Leakage Assertion |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 303,419 | 73.81% | 21,346 | $\text{Train} \cap \text{Val} = \emptyset$ (Passed) |
| **Validation** | 52,523 | 12.78% | 4,574 | $\text{Train} \cap \text{Test} = \emptyset$ (Passed) |
| **Internal Test** | 55,130 | 13.41% | 4,575 | $\text{Val} \cap \text{Test} = \emptyset$ (Passed) |
| **External Test (PAD-UFES-20)** | 654 | — | 526 | Disjoint External Modality |
| **Total Development** | **411,072** | 100.0% | **30,495** | Leakage-Controlled |

---

## 4. Optional Quality Gate
The codebase provides an optional image quality assessment module (`src/data/quality_gate.py`) evaluating:
- Laplacian blur variance
- Grayscale mean brightness
- Spatial resolution

**Scientific Clarification**: The quality gate was NOT applied to filter the reported training, validation, or test cohorts. All reported metrics reflect the complete unpruned, leakage-controlled splits.
