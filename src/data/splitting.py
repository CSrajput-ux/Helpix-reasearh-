from typing import Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_group_key(row: pd.Series) -> str:
    """
    Determine patient/lesion grouping key to prevent data leakage:
    Priority: lesion_id -> patient_id -> image_id fallback.
    """
    dataset = row['dataset']
    lesion = row.get('lesion_id', 'NA')
    patient = row.get('patient_id', 'NA')
    image = row.get('image_id', 'NA')

    if pd.notna(lesion) and str(lesion).strip() not in ('NA', 'nan', ''):
        return f"{dataset}_{lesion}"
    if pd.notna(patient) and str(patient).strip() not in ('NA', 'nan', ''):
        return f"{dataset}_{patient}"
    return f"{dataset}_{image}"


def perform_grouped_split(
    df: pd.DataFrame,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset using group-aware GroupShuffleSplit to strictly prevent patient/lesion leakage.
    Target: 70% Train | 15% Validation | 15% Internal Test.
    External test set (PAD-UFES-20) is kept completely separate.
    """
    df_copy = df.copy()
    df_copy['group_key'] = df_copy.apply(get_group_key, axis=1)

    dev_df = df_copy[
        (df_copy['dataset'] != 'PAD-UFES-20') &
        (df_copy['task1_binary'] != 'NA') &
        (df_copy['image_path'] != 'NA')
    ].copy()

    external_df = df_copy[
        (df_copy['dataset'] == 'PAD-UFES-20') &
        (df_copy['task1_binary'] != 'NA') &
        (df_copy['image_path'] != 'NA')
    ].copy()

    if len(dev_df) == 0:
        raise ValueError("No development images found for splitting.")

    # 70% Train | 30% Temporary (Validation + Internal Test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(dev_df, groups=dev_df['group_key']))
    train_df = dev_df.iloc[train_idx].reset_index(drop=True)
    temp_df  = dev_df.iloc[temp_idx].reset_index(drop=True)

    # 15% Validation | 15% Internal Test (50/50 split of the 30% temp set)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['group_key']))
    val_df           = temp_df.iloc[val_idx].reset_index(drop=True)
    internal_test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Zero-leakage verification assertions
    train_groups = set(train_df['group_key'])
    val_groups   = set(val_df['group_key'])
    test_groups  = set(internal_test_df['group_key'])

    assert train_groups.isdisjoint(val_groups), "LEAKAGE DETECTED: Train and Validation sets share groups!"
    assert train_groups.isdisjoint(test_groups), "LEAKAGE DETECTED: Train and Internal Test sets share groups!"
    assert val_groups.isdisjoint(test_groups), "LEAKAGE DETECTED: Validation and Internal Test sets share groups!"

    return train_df, val_df, internal_test_df, external_df
