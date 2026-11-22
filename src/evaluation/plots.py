"""
Plotting utilities for training curves, performance comparisons, and distributions.
Generates publication-quality charts saved to outputs/figures/ at 200 DPI.
Does not load model weights or read raw image files.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

# Consistent colors for disease classes across all figures
CLASS_COLORS = {
    "Healthy": "#1f77b4",             # Blue
    "Myopia": "#ff7f0e",              # Orange
    "Retinitis_Pigmentosa": "#d62728", # Red
}


def plot_training_curves(run_name):
    # Meant for the Model Training section of the report to show convergence
    # and compare training vs validation performance across both training stages.
    history_file = config.MODELS_DIR / f"{run_name}_history.csv"
    if not history_file.exists():
        print(f"Error: History file not found: {history_file}")
        sys.exit(1)

    df = pd.read_csv(history_file)
    epochs = df["epoch"].values

    # Check config JSON to locate fine-tuning boundary
    config_file = config.MODELS_DIR / f"{run_name}_config.json"
    fine_tune_start = None
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = json.load(f)
            fine_tune_start = cfg.get("head_epochs")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Loss curves
    axes[0].plot(epochs, df["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_title(f"Loss Curves ({run_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right plot: Accuracy curves
    axes[1].plot(epochs, df["accuracy"], label="Train Accuracy", color="#1f77b4", linewidth=2)
    axes[1].plot(epochs, df["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title(f"Accuracy Curves ({run_name})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Add vertical line for fine-tuning transition if known
    if fine_tune_start is not None and fine_tune_start < len(epochs):
        for ax in axes:
            ax.axvline(x=fine_tune_start - 0.5, color="gray", linestyle=":", linewidth=1.5)
            ax.text(fine_tune_start - 0.3, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    "Fine-tuning", rotation=90, color="gray", fontsize=9)

    out_path = config.FIGURES_DIR / f"{run_name}_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training curves to: {out_path}")


def plot_class_f1_comparison(run_names):
    # Meant for the Comparative Results section of the report to contrast
    # baseline vs weighted vs focal loss models across each class.
    records = []
    for rname in run_names:
        metrics_file = config.FIGURES_DIR / f"{rname}_test_metrics.csv"
        if not metrics_file.exists():
            print(f"Error: Missing test metrics file for run '{rname}': {metrics_file}")
            sys.exit(1)
        m_df = pd.read_csv(metrics_file)
        # Keep only the individual class rows
        class_rows = m_df[m_df["row"].isin(config.CLASS_NAMES)]
        for _, row in class_rows.iterrows():
            records.append({
                "Run": rname,
                "Class": row["row"],
                "F1": float(row["f1"]),
            })

    comp_df = pd.DataFrame(records)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=comp_df, x="Class", y="F1", hue="Run", palette="Set2")
    plt.title("Per-Class F1-Score Comparison Across Runs")
    plt.xlabel("Class")
    plt.ylabel("Test F1-Score")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bar heights
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                        textcoords="offset points")

    out_path = config.FIGURES_DIR / "f1_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved F1 comparison chart to: {out_path}")


def plot_rp_fold_spread(run_name):
    # Meant for the Cross-Validation / Stability section of the report to show
    # Retinitis Pigmentosa performance reliability across all 5 folds.
    summary_file = config.FIGURES_DIR / f"{run_name}_kfold_summary.csv"
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)

    df = pd.read_csv(summary_file)
    rp_df = df[df["class"] == "Retinitis_Pigmentosa"].copy()
    metrics = ["precision", "recall", "f1"]
    rp_df = rp_df.set_index("metric").loc[metrics].reset_index()

    means = rp_df["mean"].values
    stds = rp_df["std"].values
    labels = ["Precision", "Recall (Sens.)", "F1-Score"]
    colors = ["#2b7bba", "#e6842c", "#d9383a"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black")
    plt.title(f"Retinitis Pigmentosa 5-Fold Stability ({run_name})")
    plt.xlabel("Metric")
    plt.ylabel("Score (Mean +/- Std)")
    plt.ylim(0.0, 1.1)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Write numbers directly on the bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h / 2.0,
                 f"{mean_val:.2f}\n(+/-{std_val:.2f})",
                 ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    out_path = config.FIGURES_DIR / f"{run_name}_rp_spread.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved RP fold spread plot to: {out_path}")


def plot_class_distribution():
    # Meant for the Dataset and Methodology section of the report to illustrate
    # the dataset imbalance and the stratified split proportions.
    index_file = config.SPLITS_DIR / "index.csv"
    if not index_file.exists():
        print(f"Error: Split index file not found: {index_file}")
        sys.exit(1)

    df = pd.read_csv(index_file)
    crosstab_df = pd.crosstab(df["label"], df["split"])[["train", "val", "test"]]

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(crosstab_df))
    split_colors = {"train": "#4c72b0", "val": "#55a868", "test": "#c44e52"}

    for split_name in ["train", "val", "test"]:
        counts = crosstab_df[split_name].values
        plt.bar(crosstab_df.index, counts, bottom=bottom, label=split_name.capitalize(),
                color=split_colors[split_name], edgecolor="black", alpha=0.9)
        bottom += counts

    plt.title("Dataset Class Distribution Across Splits")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.legend(title="Split")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    # Write total count above each bar
    for i, total in enumerate(bottom):
        plt.text(i, total + 15, f"Total: {int(total)}", ha="center", fontsize=9, fontweight="bold")

    out_path = config.FIGURES_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved class distribution plot to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation and report plots.")
    parser.add_argument("--plot", type=str, required=True,
                        choices=["curves", "f1_compare", "rp_spread", "distribution"],
                        help="Plot type to generate")
    parser.add_argument("--run_name", type=str, default=None, help="Name of training run")
    parser.add_argument("--run_names", nargs="+", default=None, help="List of run names for comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot == "curves":
        if not args.run_name:
            print("Error: --run_name is required when using --plot curves")
            sys.exit(1)
        plot_training_curves(args.run_name)
    elif args.plot == "f1_compare":
        if not args.run_names:
            print("Error: --run_names is required when using --plot f1_compare")
            sys.exit(1)
        plot_class_f1_comparison(args.run_names)
    elif args.plot == "rp_spread":
        if not args.run_name:
            print("Error: --run_name is required when using --plot rp_spread")
            sys.exit(1)
        plot_rp_fold_spread(args.run_name)
    elif args.plot == "distribution":
        plot_class_distribution()


if __name__ == "__main__":
    main()
