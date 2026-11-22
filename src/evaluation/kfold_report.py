"""
Summarize 5-fold cross-validation performance across folds.
Reads prediction CSV files outputs/predictions/<run_name>_fold0.csv to fold4.csv,
computes mean, standard deviation, minimum, and maximum for each metric,
and saves the summary in long format to outputs/figures/<run_name>_kfold_summary.csv.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5-fold cross-validation summary report.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of training run (e.g. kfold_weighted)")
    return parser.parse_args()


def main():
    args = parse_args()
    n_folds = config.N_FOLDS

    # Step 1: Check all fold prediction files exist
    fold_files = []
    for fold in range(n_folds):
        fpath = config.PREDICTIONS_DIR / f"{args.run_name}_fold{fold}.csv"
        if not fpath.exists():
            print(f"Error: Missing required fold prediction file: {fpath}")
            sys.exit(1)
        fold_files.append(fpath)

    # Step 2: Compute metrics for each fold
    # We store metrics per fold in lists to calculate fold-to-fold statistics
    fold_metrics = {c: {"precision": [], "recall": [], "f1": []} for c in config.CLASS_NAMES}
    macro_metrics = {"precision": [], "recall": [], "f1": []}

    for fold, fpath in enumerate(fold_files):
        df = pd.read_csv(fpath)
        y_true = df["true_id"].values
        y_pred = df["pred_id"].values

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        for i, c in enumerate(config.CLASS_NAMES):
            fold_metrics[c]["precision"].append(p[i])
            fold_metrics[c]["recall"].append(r[i])
            fold_metrics[c]["f1"].append(f[i])

        macro_metrics["precision"].append(float(np.mean(p)))
        macro_metrics["recall"].append(float(np.mean(r)))
        macro_metrics["f1"].append(float(np.mean(f)))

    # Step 3: Compute summary statistics (mean, std, min, max) across 5 folds
    summary_rows = []
    target_groups = [(c, fold_metrics[c]) for c in config.CLASS_NAMES] + [("macro", macro_metrics)]

    for group_name, metrics_dict in target_groups:
        for metric_name in ["precision", "recall", "f1"]:
            values = metrics_dict[metric_name]
            summary_rows.append({
                "class": group_name,
                "metric": metric_name,
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            })

    # Step 4: Save summary CSV matching contract point 7
    summary_df = pd.DataFrame(summary_rows)
    output_path = config.FIGURES_DIR / f"{args.run_name}_kfold_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"\n--- 5-Fold Cross-Validation Summary for '{args.run_name}' ---")
    print(summary_df.to_string(index=False))
    print(f"\nSaved K-fold summary report to: {output_path}")

    # Step 5: Highlight Retinitis_Pigmentosa recall across folds
    # Why fold spread matters for small classes:
    # With only ~135 Retinitis Pigmentosa cases, each validation fold has only ~27 RP images.
    # A single misclassified image causes a ~3.7% swing in recall. Looking at min, max, and std
    # reveals whether performance is genuinely reliable or highly sensitive to the test split.
    rp_recall_stats = summary_df[
        (summary_df["class"] == "Retinitis_Pigmentosa") & (summary_df["metric"] == "recall")
    ].iloc[0]

    print(
        f"\nRP recall across {n_folds} folds: mean {rp_recall_stats['mean']:.2f}, "
        f"std {rp_recall_stats['std']:.2f}, range {rp_recall_stats['min']:.2f} to {rp_recall_stats['max']:.2f}"
    )


if __name__ == "__main__":
    main()
