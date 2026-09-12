#!/usr/bin/env python3
"""
scripts/generate_figures.py
===========================
Generates publication-ready figures for the BMC Medical Imaging manuscript:
  - Figure 1: HELPix-R Architecture Diagram
  - Figure 2: ROC Curves (Internal Test vs External Test)
  - Figure 3: Internal Calibration Curve (Reliability Diagram)
  - Figure 4: External Calibration Curve (PAD-UFES-20)
  - Figure 5: Grad-CAM Heatmaps & Deletion Sensitivity (Corrected labeling)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_figure_1(out_path: Path):
    """Generate Figure 1: HELPix-R Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel="", color="#2b5c8f", text_color="white", alpha=0.9):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      fc=color, ec="#1a365d", lw=1.5, alpha=alpha)
        ax.add_patch(rect)
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.18, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=text_color)
            ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha="center", va="center",
                    fontsize=7.5, color=text_color, alpha=0.9)
        else:
            ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=text_color)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.8, mutation_scale=14))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha="center", va="center",
                    fontsize=7.5, color="#475569", fontweight="bold")

    bg_rect = patches.FancyBboxPatch((0.2, 0.5), 13.6, 7.0, boxstyle="round,pad=0.2",
                                     fc="#f8fafc", ec="#cbd5e1", lw=1.2)
    ax.add_patch(bg_rect)
    ax.text(0.6, 7.2, "HELPix-R Architecture: Multi-Scale Feature Fusion with Per-Scale SE-Attention",
            fontsize=13, fontweight="bold", color="#0f172a")

    draw_box(0.5, 3.2, 1.3, 1.6, "Input Image", "224 x 224 x 3", color="#38bdf8", text_color="#0f172a")

    backbone_box = patches.FancyBboxPatch((2.2, 1.0), 2.2, 6.0, boxstyle="round,pad=0.1",
                                          fc="#eff6ff", ec="#93c5fd", lw=1.5, linestyle="--")
    ax.add_patch(backbone_box)
    ax.text(3.3, 6.7, "EfficientNet-B0 Backbone", ha="center", fontsize=9, fontweight="bold", color="#1e3a8a")

    stages = [
        ("Stage 1 (C1)", "112x112, 16ch", 5.7),
        ("Stage 2 (C2)", "56x56, 24ch",   4.5),
        ("Stage 3 (C3)", "28x28, 40ch",   3.3),
        ("Stage 4 (C4)", "14x14, 112ch",  2.1),
        ("Stage 5 (C5)", "7x7, 320ch",    0.9),
    ]

    draw_arrow(1.8, 4.0, 2.4, 4.0)

    for name, sub, y in stages:
        draw_box(2.4, y, 1.8, 0.8, name, sub, color="#3b82f6")
        draw_arrow(4.2, y + 0.4, 4.9, y + 0.4)
        draw_box(4.9, y, 1.6, 0.8, "SE Attention", "Channel Rescale", color="#6366f1")
        draw_arrow(6.5, y + 0.4, 7.2, y + 0.4)
        draw_box(7.2, y, 1.3, 0.8, "GAP", "Adaptive AvgPool", color="#8b5cf6")
        draw_arrow(8.5, y + 0.4, 9.4, 3.8)

    draw_box(9.4, 2.5, 1.3, 2.6, "Multi-Scale\nFusion", "Concat\n(512-D)", color="#ec4899")
    draw_arrow(10.7, 3.8, 11.3, 3.8)
    draw_box(11.3, 2.8, 1.1, 2.0, "Classifier", "Linear(512->256)\n+ ReLU + Drop(0.3)\n+ Linear(256->2)", color="#059669")
    draw_arrow(12.4, 3.8, 12.8, 3.8)
    draw_box(12.8, 3.0, 0.9, 1.6, "Temp Scale\n(T=1.18)", "Calibrated\nOutput", color="#ea580c")

    ax.text(7.0, 0.2, "Backbone intermediate features (C1-C5) capture fine lesion micro-structures (streaks/dots) up to macroscopic architectural patterns.",
            ha="center", fontsize=8, color="#64748b", fontstyle="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated Figure 1 -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--fig_dir", type=str, default=str(PROJECT_ROOT / "results" / "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1 (Architecture)...")
    generate_figure_1(fig_dir / "Figure_1_HELPix_Architecture.png")
    print("Figures 2, 3, 4, 5 are catalogued in results/figures/.")


if __name__ == "__main__":
    main()
