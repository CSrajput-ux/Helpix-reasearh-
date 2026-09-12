import pytest
import pandas as pd
import numpy as np
import torch
from src.data.transforms import get_training_transforms, get_evaluation_transforms
from src.data.dataset import SkinLesionDataset, LABEL_TO_IDX
from src.data.splitting import get_group_key, perform_grouped_split
from src.data.quality_gate import evaluate_image_quality


def test_label_mapping():
    assert LABEL_TO_IDX['non_malignant'] == 0
    assert LABEL_TO_IDX['malignant'] == 1


def test_transforms():
    train_tf = get_training_transforms(224)
    eval_tf = get_evaluation_transforms(224)
    assert train_tf is not None
    assert eval_tf is not None


def test_group_key():
    r1 = pd.Series({'dataset': 'HAM', 'lesion_id': 'L123', 'patient_id': 'P456', 'image_id': 'IMG789'})
    assert get_group_key(r1) == 'HAM_L123'

    r2 = pd.Series({'dataset': 'HAM', 'lesion_id': 'NA', 'patient_id': 'P456', 'image_id': 'IMG789'})
    assert get_group_key(r2) == 'HAM_P456'

    r3 = pd.Series({'dataset': 'HAM', 'lesion_id': 'NA', 'patient_id': 'NA', 'image_id': 'IMG789'})
    assert get_group_key(r3) == 'HAM_IMG789'


def test_perform_grouped_split():
    # Construct synthetic dataset with multiple images per patient
    records = []
    for pid in range(100):
        for img in range(3):
            records.append({
                'dataset': 'ISIC',
                'lesion_id': f'lesion_{pid}',
                'patient_id': f'patient_{pid}',
                'image_id': f'img_{pid}_{img}',
                'task1_binary': 'malignant' if pid % 5 == 0 else 'non_malignant',
                'image_path': f'/dummy/path/{pid}_{img}.jpg'
            })
    # Add external samples
    for ext_id in range(20):
        records.append({
            'dataset': 'PAD-UFES-20',
            'lesion_id': f'ext_lesion_{ext_id}',
            'patient_id': f'ext_patient_{ext_id}',
            'image_id': f'ext_img_{ext_id}',
            'task1_binary': 'malignant' if ext_id % 2 == 0 else 'non_malignant',
            'image_path': f'/dummy/ext/{ext_id}.jpg'
        })
    df = pd.DataFrame(records)

    train_df, val_df, test_df, ext_df = perform_grouped_split(df, seed=42)

    # Check that lengths are non-zero
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(ext_df) == 20

    # Check zero-leakage
    train_groups = set(train_df['group_key'])
    val_groups = set(val_df['group_key'])
    test_groups = set(test_df['group_key'])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
