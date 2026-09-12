# Contributing to HELPix-R

We welcome scientific discussions, bug reports, and methodological extensions to the HELPix-R framework.

## How to Contribute

1. **Reporting Issues**:
   - For bug reports or discrepancy investigations, open an issue detailing the execution environment, random seed, hardware configuration, and full traceback.
2. **Submitting Pull Requests**:
   - Ensure all changes preserve backward compatibility and unit tests (`pytest tests/`).
   - If proposing architectural variants or new baseline models, log full evaluation metrics (AUROC, Sensitivity, Specificity, F1, ECE, Brier score).
3. **Reproducibility Verification**:
   - Include random seeds (`seed = 42`) in all experimental configurations.
