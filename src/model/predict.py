# What this file does: loads a saved model and runs inference on a chosen split.
# What it reads: splits/index.csv, image files via load_image, a saved .keras model.
# What it writes: outputs/predictions/<run_name>_<tag>.csv (CONTRACT point 5).
# Why it exists: separates prediction from training so anyone can re-run inference
# on any split at any time without re-training.

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from config import SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR, CLASS_NAMES
from src.preprocessing.image_utils import load_image


def run_prediction(run_name, tag, model=None):
    # Loads the model (if not given), selects the right rows from index.csv,
    # predicts in batches of 32, and writes the predictions CSV.
    # tag is "val", "test", or "fold0".."fold4".

    # --- decide the model path based on tag ---
    # Normal tags (val, test) use the main model file.
    # Fold tags (fold0..fold4) use the per-fold model file.
    if tag in ("val", "test"):
        model_path = MODELS_DIR / f"{run_name}.keras"
    elif tag.startswith("fold") and tag[4:].isdigit():
        k = tag[4:]
        model_path = MODELS_DIR / f"{run_name}_fold{k}.keras"
    else:
        raise ValueError(f"Unknown tag '{tag}'. Use val, test or fold0..fold4.")

    # Load model from disk only if the caller did not pass one in.
    # Passing a model in avoids loading it twice during a training run.
    if model is None:
        print(f"[predict] loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path), compile=False)

    # --- select the right rows from index.csv ---
    index_path = SPLITS_DIR / "index.csv"
    df = pd.read_csv(index_path)

    if tag in ("val", "test"):
        # For val and test we use the fixed split column.
        subset = df[df["split"] == tag].copy()
    else:
        # For fold<k> we use the fold column equal to k.
        k = int(tag[4:])
        subset = df[df["fold"] == k].copy()

    subset = subset.reset_index(drop=True)
    print(f"[predict] tag={tag}  images to predict: {len(subset)}")

    # --- load images and predict in batches of 32 ---
    batch_size = 32
    all_probs = []

    for start in range(0, len(subset), batch_size):
        batch_rows = subset.iloc[start : start + batch_size]
        # Build a numpy batch of shape (n, 224, 224, 3).
        batch_images = np.stack([
            load_image(row["filepath"]) for _, row in batch_rows.iterrows()
        ])
        probs = model.predict(batch_images, verbose=0)  # shape (n, 3)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)   # shape (total, 3)
    pred_ids = np.argmax(all_probs, axis=1)

    # --- build the output DataFrame (CONTRACT point 5) ---
    out_df = pd.DataFrame({
        "filepath":     subset["filepath"].values,
        "true_id":      subset["label_id"].values,
        "pred_id":      pred_ids,
        "prob_healthy": all_probs[:, 0],
        "prob_myopia":  all_probs[:, 1],
        "prob_rp":      all_probs[:, 2],
    })

    # --- save predictions CSV ---
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{run_name}_{tag}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[predict] saved predictions to {out_path}")

    # --- print accuracy and RP recall ---
    accuracy = (out_df["true_id"] == out_df["pred_id"]).mean()
    rp_id = CLASS_NAMES.index("Retinitis_Pigmentosa")
    rp_rows = out_df[out_df["true_id"] == rp_id]
    rp_correct = (rp_rows["pred_id"] == rp_id).sum()
    rp_total = len(rp_rows)
    print(f"[predict] accuracy : {accuracy:.4f}  ({int(accuracy * len(out_df))}/{len(out_df)} correct)")
    print(f"[predict] RP correct: {rp_correct}/{rp_total}")

    return out_df


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a split.")
    parser.add_argument("--run_name", required=True,
                        help="Name of the run (used to find model and output files).")
    parser.add_argument("--tag", required=True,
                        help="Which split to predict: val, test or fold0..fold4.")
    args = parser.parse_args()

    run_prediction(args.run_name, args.tag)
