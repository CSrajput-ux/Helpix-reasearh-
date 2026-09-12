import os
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from src.utils.reproducibility import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_seed_consistency():
    set_seed(42)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(42)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert r1 == r2
    assert np.allclose(np1, np2)
    assert torch.allclose(t1, t2)


def test_result_artifacts_exist():
    required_tables = [
        "Table_1_dataset_characteristics.csv",
        "baseline_comparison.csv",
        "Table_calibration_comparison.csv",
        "final_split_statistics.csv",
        "deduplication_statistics.csv",
        "domain_shift_analysis.csv",
        "gradcam_deletion_results.csv",
    ]
    for tbl in required_tables:
        p = PROJECT_ROOT / "results" / "tables" / tbl
        assert p.exists(), f"Missing required table artifact: {p}"

    metrics_p = PROJECT_ROOT / "results" / "metrics" / "final_metrics.csv"
    assert metrics_p.exists(), f"Missing final_metrics.csv: {metrics_p}"

    required_figures = [
        "Figure_1_HELPix_Architecture.png",
        "Figure_2_ROC_Internal_vs_External.png",
        "Figure_3_Calibration_Internal.png",
        "Figure_4_Calibration_External.png",
        "Figure_5_GradCAM_Heatmaps.png",
    ]
    for fig in required_figures:
        p = PROJECT_ROOT / "results" / "figures" / fig
        assert p.exists(), f"Missing required figure artifact: {p}"


def test_no_hardcoded_absolute_paths_in_src():
    src_dir = PROJECT_ROOT / "src"
    prohibited = ["d:\\helpix", "c:\\users", "/home/"]
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for term in prohibited:
            assert term not in content, f"Hardcoded machine path '{term}' found in {py_file}"


def test_verified_reported_values():
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "final_metrics.csv"
    df = pd.read_csv(metrics_path)
    
    int_row = df[df['Cohort'] == 'Internal Test'].iloc[0]
    assert np.isclose(int_row['AUROC'], 0.9778, atol=1e-3)
    assert np.isclose(int_row['Sensitivity'], 0.7865, atol=1e-3)
    assert np.isclose(int_row['Specificity'], 0.9910, atol=1e-3)
    assert np.isclose(int_row['F1'], 0.4433, atol=1e-3)
    assert np.isclose(int_row['ECE'], 0.0054, atol=1e-3)
    assert np.isclose(int_row['Brier'], 0.0149, atol=1e-3)
    assert int_row['N'] == 55130

    ext_row = df[df['Cohort'].str.contains('External')].iloc[0]
    assert np.isclose(ext_row['AUROC'], 0.6451, atol=1e-3)
    assert np.isclose(ext_row['ECE'], 0.1650, atol=1e-3)
    assert ext_row['N'] == 654
