# Pre-Submission and Publication Release Checklist

This checklist must be verified before submitting the manuscript to *BMC Medical Imaging* and before making the GitHub repository publicly accessible.

---

## 1. Scientific & Ethical Integrity
- [ ] **No Fabricated Claims**: Ensure no text claims "state-of-the-art" or "clinical-grade" without rigorous external clinical trial validation.
- [ ] **AUROC Clarification**: Confirm that `MobileNetV3-Large` is acknowledged as the highest AUROC baseline (0.9805), and HELPix-R is highlighted for the highest F1-score (0.4433).
- [ ] **Phrasing Check**: Confirm that the phrase "balanced sensitivity/specificity" has been removed everywhere.
- [ ] **External Testing Terminology**: Ensure `external testing` is used consistently, never "external validation".
- [ ] **No Patient Identifiers**: Confirm zero patient health information (PHI), clinical names, or identifying metadata exist in the repository.
- [ ] **No Raw Image Redistribution**: Verify that raw images from ISIC, HAM10000, PAD-UFES-20, or BCN20000 are not committed to Git.
- [ ] **No Credentials / Absolute Paths**: Verify that no API keys, private passwords, tokens, or hardcoded machine paths (`D:\...`) exist in committed code.

---

## 2. Licensing & Governance
- [ ] **Software License Confirmed**: Authors must select and confirm the open-source software license (e.g., MIT or Apache 2.0) to replace `[SOFTWARE LICENSE TO BE CONFIRMED]`.
- [ ] **Dataset Licenses Distinct**: Confirm that original dataset licenses (CC0, CC BY-NC 4.0, CC BY 4.0) are documented separately from the code license.
- [ ] **CITATION.cff Review**: Confirm author affiliations and details before final tagging.

---

## 3. Code & Environment Verification
- [ ] **Unit Tests Passed**: Run `pytest tests/` and verify that all tests pass.
- [ ] **CLI Scripts Tested**: Confirm that each script in `scripts/` executes with `--help` or dry-run configuration.
- [ ] **Environment Export**: Verify `requirements.txt` and `environment.yml` can build cleanly in a fresh Python environment.

---

## 4. Release Archiving & Paper Linking
- [ ] **Create Public GitHub Repository**: Push the finalized codebase to GitHub.
- [ ] **Create Release Tag**: Create a semantic release tag (e.g., `v1.0.0`) on GitHub.
- [ ] **Link Zenodo Archive**: Connect Zenodo to the GitHub repository to mint a permanent DOI for the release.
- [ ] **Update Manuscript Text**:
  - Insert GitHub repository URL into `docs/BMC_CODE_AVAILABILITY_FINAL.md`.
  - Insert Zenodo DOI into `docs/BMC_CODE_AVAILABILITY_FINAL.md`.
  - Paste the finalized text into the manuscript's "Code Availability" section.
  - Paste `docs/BMC_DATA_AVAILABILITY.md` into the manuscript's "Data Availability" section.
