from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .transforms import get_training_transforms, get_evaluation_transforms

CLASS_NAMES = ['non_malignant', 'malignant']
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class SkinLesionDataset(Dataset):
    """
    Memory-efficient lazy-loading skin lesion dataset.
    Images are opened and converted on demand, preventing high system RAM consumption.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        raw_label = row['task1_binary']
        label = LABEL_TO_IDX[raw_label] if isinstance(raw_label, str) else int(raw_label)
        return image, label


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[Dict[str, DataLoader], SkinLesionDataset]:
    """
    Create PyTorch DataLoaders for train, val, internal test, and external test cohorts.
    Applies WeightedRandomSampler on the training split to handle class imbalance.
    """
    train_tf = get_training_transforms(img_size)
    eval_tf = get_evaluation_transforms(img_size)

    train_ds = SkinLesionDataset(train_df, train_tf)
    val_ds   = SkinLesionDataset(val_df,   eval_tf)
    test_ds  = SkinLesionDataset(test_df,  eval_tf)
    ext_ds   = SkinLesionDataset(ext_df,   eval_tf)

    # Class-imbalance correction via WeightedRandomSampler
    counts = train_df['task1_binary'].value_counts()
    weights_map = {c: 1.0 / counts[c] for c in counts.index}
    sample_weights = train_df['task1_binary'].map(weights_map).values.astype('float64')
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory),
        'val':   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
        'test':  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
        'ext':   DataLoader(ext_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
    }
    return loaders, test_ds
