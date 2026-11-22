"""
Evaluate model predictions on a validation or test set split.
Reads prediction CSV files from outputs/predictions/, computes classification metrics,
saves a metrics CSV report, and generates raw and normalized confusion matrix plots.
Does not load model weights or read raw image files.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate screening model predictions.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of training run (e.g. baseline)")
    parser.add_argument("--tag", type=str, required=True, help="Split tag (e.g. val or test)")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_file = config.PREDICTIONS_DIR / f"{args.run_name}_{args.tag}.csv"

    # Step 1: Verify prediction file exists
    if not pred_file.exists():
        print(f"Error: Prediction file not found: {pred_file}")
        sys.exit(1)

    print(f"Reading predictions from: {pred_file}")
    df = pd.read_csv(pred_file)

    y_true = df["true_id"].values
    y_pred = df["pred_id"].values

    # Step 2: Compute key classification metrics
    # Why accuracy alone can mislead on imbalanced data:
    # If 80% of samples are Healthy, a trivial model predicting 'Healthy' for everything
    # would get 80% accuracy while failing 100% of disease cases. Balanced accuracy
    # computes the unweighted average of recalls across all classes.
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n--- Evaluation Summary ---")
    print(f"Overall Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy:         {bal_acc:.4f}")
    print(f"Macro F1-Score:            {macro_f1:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )
    print(report_text)

    # Step 3: Print Retinitis_Pigmentosa recall separately
    # Missing a true disease case (false negative) is the most critical error in screening.
    rp_id = config.CLASS_TO_ID["Retinitis_Pigmentosa"]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    rp_recall = recalls[rp_id]
    print(f"Retinitis_Pigmentosa Recall (Sensitivity): {rp_recall:.4f}")

    # Step 4: Build and save confusion matrix figures
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # 4a: Raw count confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix ({args.run_name} - {args.tag})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = config.FIGURES_DIR / f"{args.run_name}_{args.tag}_cm.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved raw confusion matrix to: {cm_path}")

    # 4b: Row-normalized confusion matrix (percentages)
    # Row normalization shows class recall directly on the diagonal
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero if class has 0 support
    cm_norm = cm.astype(float) / row_sums

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.title(f"Normalized Confusion Matrix ({args.run_name} - {args.tag})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_norm_path = config.FIGURES_DIR / f"{args.run_name}_{args.tag}_cm_norm.png"
    plt.savefig(cm_norm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved normalized confusion matrix to: {cm_norm_path}")

    # Step 5: Save metrics CSV matching contract point 6
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    total_support = int(np.sum(supports))

    metrics_rows = []
    for i, class_name in enumerate(config.CLASS_NAMES):
        metrics_rows.append({
            "row": class_name,
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1": round(float(f1s[i]), 4),
            "support": int(supports[i]),
            "accuracy": "",
            "balanced_accuracy": "",
        })

    metrics_rows.append({
        "row": "overall",
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
        "support": total_support,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = config.FIGURES_DIR / f"{args.run_name}_{args.tag}_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics table to: {metrics_csv_path}")


if __name__ == "__main__":
    main()
