# Model Card: HELPix-R

## 1. Model Details
- **Model Name**: HELPix-R (High-Efficiency Lesion Pixel-Reliability Framework)
- **Model Type**: Deep Convolutional Neural Network with Multi-Scale Feature Fusion and Channel Attention
- **Backbone**: EfficientNet-B0 (torchvision pretrained weights)
- **Primary Task**: Binary skin-lesion malignancy classification (Malignant vs. Non-Malignant)
- **Input Modality**: Single RGB dermatoscopic or clinical image ($224 \times 224 \times 3$)
- **Output**: Calibrated malignancy risk probability $P(\text{Malignant}) \in [0, 1]$, binary decision, and MC-Dropout predictive uncertainty
- **Weight Checkpoint Distribution**: The pre-trained PyTorch weights (`helpix_r_best.pt`, 16.9 MB) are not tracked directly in the Git repository to prevent repository bloat. The weights will be archived in the official release package on Zenodo under a persistent DOI upon manuscript publication. Users wishing to evaluate the trained model without retraining can download the checkpoint from Zenodo and place it in `results/` or `models/`. Alternatively, users can retrain the model from scratch using `scripts/train.py --config configs/training_config.yaml --model helpix_r`.

---

## 2. Model Architecture
```
Input Image (224 × 224 × 3)
      │
EfficientNet-B0 Backbone
      ├── Stage 1 (C1): 112 × 112,  16 channels ──► SE Attention ──► Global AvgPool ──► (16-D)
      ├── Stage 2 (C2):  56 ×  56,  24 channels ──► SE Attention ──► Global AvgPool ──► (24-D)
      ├── Stage 3 (C3):  28 ×  28,  40 channels ──► SE Attention ──► Global AvgPool ──► (40-D)
      ├── Stage 4 (C4):  14 ×  14, 112 channels ──► SE Attention ──► Global AvgPool ──► (112-D)
      └── Stage 5 (C5):   7 ×   7, 320 channels ──► SE Attention ──► Global AvgPool ──► (320-D)
                                                                                          │
                                                                   Channel Concatenation (512-D)
                                                                                          │
                                                                                    Dropout (p = 0.3)
                                                                                          │
                                                                                 Linear (512 ──► 256)
                                                                                          │
                                                                                        ReLU
                                                                                          │
                                                                                    Dropout (p = 0.3)
                                                                                          │
                                                                                 Linear (256 ──► 2)
                                                                                          │
                                                                         Temperature Scaling (T = 1.1799)
                                                                                          │
                                                                             Calibrated Probability
```

---

## 3. Training & Optimization
- **Loss Function**: Focal Loss ($\alpha = 1.0, \gamma = 2.0$) to mitigate severe class imbalance.
- **Optimizer**: AdamW ($\text{learning rate} = 10^{-4}, \text{weight decay} = 10^{-4}$).
- **Scheduler**: ReduceLROnPlateau ($\text{factor} = 0.5, \text{patience} = 2$).
- **Batch Size**: 32 with Automated Mixed Precision (AMP FP16).
- **Sampling Strategy**: WeightedRandomSampler with inverse-class frequency weighting.
- **Data Augmentations**: Random horizontal/vertical flip, random rotation ($20^\circ$), color jitter (brightness 0.2, contrast 0.2, saturation 0.1).

---

## 4. Evaluated Performance Summary

### Internal Test Set (N = 55,130 dermoscopic images)
- **AUROC**: 0.9778 (95% Bootstrap CI: 0.9666–0.9866)
- **Sensitivity**: 0.7865
- **Specificity**: 0.9910
- **F1-Score**: 0.4433
- **Expected Calibration Error (ECE)**: 0.0054 (post-calibration, improved from 0.0076 pre-calibration)
- **Brier Score**: 0.0149 (post-calibration, improved from 0.0152 pre-calibration)

> **Scientific Clarification on Model Rankings**:
> Among the evaluated models on the internal test set, **MobileNetV3-Large achieved the highest AUROC (0.9805)**. HELPix-R achieved the highest F1-score (0.4433) among the evaluated models, with sensitivity of 0.7865 and specificity of 0.9910 on the internal test set.

### Baseline Comparison (Internal Test Cohort)
| Architecture | AUROC | F1-Score |
| :--- | :---: | :---: |
| ResNet50 | 0.9771 | 0.2650 |
| EfficientNet-B0 | 0.9757 | 0.2412 |
| **MobileNetV3-Large** | **0.9805** | 0.3284 |
| ConvNeXt-Tiny | 0.9751 | 0.3658 |
| **HELPix-R (Proposed)** | 0.9778 | **0.4433** |

---

## 5. External Testing & Domain Shift (PAD-UFES-20, N = 654)
- **AUROC**: 0.6451
- **Sensitivity**: 0.4808
- **Specificity**: 0.7311
- **F1-Score**: 0.5882
- **ECE**: 0.1650 (post-calibration, improved from 0.1782 pre-calibration)
- **Brier Score**: 0.5641
- **Domain Shift Characterization**:
  Under external smartphone-domain testing on PAD-UFES-20, model discrimination decreased by 0.3327 absolute AUROC points ($\Delta\text{AUROC} = -0.3327$), accompanied by an increase in ECE of 0.1596 absolute points ($\Delta\text{ECE} = +0.1596$).

---

## 6. Uncertainty & Explainability
- **Predictive Uncertainty**: 20-sample Monte Carlo Dropout during inference (BatchNorm locked in eval mode).
- **Attribution Sensitivity**: Grad-CAM on penultimate feature layer (`features[-2]`). Quantitative 40% top-saliency masking yielded a mean probability reduction of 0.0723 (median: 0.0487).
- **Scope Note**: Perturbation evaluates attribution sensitivity and does not establish clinical reasoning or causal biological validity.

---

## 7. Clinical Disclaimer & Non-Diagnostic Notice
> **IMPORTANT CLINICAL DISCLAIMER**:
> This model is a scientific research prototype developed exclusively for retrospective methodological benchmarking and domain-transfer analysis. **It is NOT a clinically certified medical device and is NOT approved for autonomous diagnosis, triaging, or therapeutic decision-making in clinical healthcare settings.**
