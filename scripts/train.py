#!/usr/bin/env python3
"""
scripts/train.py
================
Executes training for HELPix-R or baseline models using Focal Loss, AdamW, and ReduceLROnPlateau.
Supports automated mixed precision (AMP) for memory safety.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import pandas as pd
from src.utils.config import load_yaml_config
from src.utils.reproducibility import set_seed
from src.models.helpix_r import HELPixR
from src.models.baselines import create_baseline_model
from src.data.dataset import get_dataloaders
from src.data.splitting import perform_grouped_split
from src.training.trainer import fit_model


def main():
    parser = argparse.ArgumentParser(description="Train HELPix-R or baseline architectures.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
                        help="Path to training config YAML")
    parser.add_argument("--model", type=str, default="helpix_r",
                        choices=["helpix_r", "resnet50", "efficientnet_b0", "mobilenetv3", "convnext_tiny"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override initial learning rate")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to audited master metadata CSV")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save or load trained model checkpoint")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = cfg.get("hardware", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device} | Architecture: {args.model}")

    ckpt_path = args.checkpoint or str(PROJECT_ROOT / "results" / f"{args.model}_best.pt")

    if args.metadata is None:
        print("No metadata CSV provided. To run full training, specify --metadata <path_to_master_metadata_audited.csv>.")
        print(f"Verified checkpoints are catalogued in the models directory.")
        return

    df = pd.read_csv(args.metadata)
    train_df, val_df, test_df, ext_df = perform_grouped_split(df, seed=seed)
    loaders, _ = get_dataloaders(train_df, val_df, test_df, ext_df,
                                 img_size=cfg["image"]["size"],
                                 batch_size=cfg["optimization"]["batch_size"],
                                 num_workers=cfg["hardware"]["num_workers"])

    if args.model == "helpix_r":
        model = HELPixR(num_classes=2, dropout_p=cfg["model"]["dropout_rate"])
    else:
        model = create_baseline_model(args.model, num_classes=2)

    epochs = args.epochs or cfg["optimization"]["epochs"]
    lr = args.lr or cfg["optimization"]["learning_rate"]
    patience = cfg["optimization"]["early_stopping_patience"]

    print(f"Starting training: {epochs} epochs, lr={lr}, patience={patience}")
    fit_model(model, loaders['train'], loaders['val'], epochs=epochs, lr=lr, patience=patience, ckpt_path=ckpt_path, device=device)
    print(f"Training completed. Best checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
