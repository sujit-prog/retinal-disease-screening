"""
Build stratified train/val/test splits and 5-fold cross-validation index.
Reads images from data/ for the 3 classes, filters corrupt and duplicate files,
and writes the final split index to splits/index.csv.
"""

import hashlib
from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import config


def compute_file_hash(file_path):
    """Compute MD5 hash of a file by reading in chunks to detect identical files."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("--- STEP 1: Verifying class folders in data directory ---")
    for class_name, folder_rel in config.CLASS_FOLDERS.items():
        folder_path = config.RAW_DIR / folder_rel
        if not folder_path.exists():
            raise FileNotFoundError(
                f"Missing required class folder for {class_name}: {folder_path}"
            )
        print(f"Found folder for {class_name}: {folder_path}")

    print("\n--- STEP 2: Collecting image files ---")
    image_extensions = {".jpg", ".jpeg", ".png"}
    rows = []

    for class_name in config.CLASS_NAMES:
        folder_path = config.RAW_DIR / config.CLASS_FOLDERS[class_name]
        class_id = config.CLASS_TO_ID[class_name]
        
        # Search all image files in the class folder and subfolders
        for file_path in folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                rel_path = file_path.relative_to(config.RAW_DIR).as_posix()
                rows.append({
                    "filepath": rel_path,
                    "label": class_name,
                    "label_id": class_id,
                })

    df = pd.DataFrame(rows)
    print(f"Collected total of {len(df)} candidate image files.")

    print("\n--- STEP 3: Checking image file readability with OpenCV ---")
    valid_rows = []
    skipped_count = 0
    for idx, row in df.iterrows():
        full_img_path = config.RAW_DIR / row["filepath"]
        test_img = cv2.imread(str(full_img_path))
        if test_img is None:
            print(f"Skipping unreadable file: {row['filepath']}")
            skipped_count += 1
        else:
            valid_rows.append(row)

    print(f"Unreadable files skipped: {skipped_count}")
    df = pd.DataFrame(valid_rows)

    print("\n--- STEP 4: Detecting exact duplicates and conflicting labels ---")
    # Why deduplication is critical:
    # If duplicate images exist across train and test sets, the model would memorize them,
    # causing data leakage and overly optimistic test results that fail on new patients.
    counts_before = df["label"].value_counts().to_dict()
    print("Image counts per class before deduplication:", counts_before)

    df["file_hash"] = [
        compute_file_hash(config.RAW_DIR / p) for p in df["filepath"]
    ]

    # Find duplicate hashes
    hash_counts = df["file_hash"].value_counts()
    dup_hashes = hash_counts[hash_counts > 1].index.tolist()
    total_duplicate_copies = df["file_hash"].duplicated(keep=False).sum() - len(dup_hashes)
    print(f"Found {len(dup_hashes)} duplicate hash sets ({total_duplicate_copies} duplicate copies).")

    # Check for cross-class label conflicts
    conflicting_hashes = set()
    for file_hash, group in df.groupby("file_hash"):
        if group["label"].nunique() > 1:
            conflicting_hashes.add(file_hash)
            print(f"Conflicting label duplicate found across classes:")
            for _, r in group.iterrows():
                print(f"  Class: {r['label']:20} Path: {r['filepath']}")

    # Drop all conflicting files because their ground-truth label is ambiguous
    if conflicting_hashes:
        print(f"Dropping all {len(conflicting_hashes)} conflicting duplicate sets.")
        df = df[~df["file_hash"].isin(conflicting_hashes)].copy()

    # For same-class duplicates, keep only the first copy
    df = df.drop_duplicates(subset=["file_hash"], keep="first").copy()
    df = df.drop(columns=["file_hash"]).reset_index(drop=True)

    counts_after = df["label"].value_counts().to_dict()
    print("Image counts per class after deduplication:", counts_after)

    print("\n--- STEP 5: Creating stratified train/val/test splits ---")
    # Why we split first and augment later:
    # Augmentation must happen only on training images after the split. If we augmented
    # before splitting, augmented versions of the same fundus would leak into val/test,
    # invalidating our evaluation.
    train_ratio, val_ratio, test_ratio = config.SPLIT_RATIOS
    rest_ratio = 1.0 - train_ratio
    test_share_of_rest = test_ratio / (val_ratio + test_ratio)

    df["split"] = ""
    train_idx, rest_idx = train_test_split(
        df.index,
        test_size=rest_ratio,
        stratify=df["label_id"],
        random_state=config.RANDOM_SEED,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_share_of_rest,
        stratify=df.loc[rest_idx, "label_id"],
        random_state=config.RANDOM_SEED,
    )

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    print("\n--- STEP 6: Creating 5-fold cross-validation fold assignments ---")
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    df["fold"] = -1
    for fold_num, (_, fold_val_idx) in enumerate(skf.split(df, df["label_id"])):
        df.loc[df.index[fold_val_idx], "fold"] = fold_num

    print("\n--- STEP 7: Running contract sanity assertions ---")
    assert df["filepath"].nunique() == len(df), "Duplicate filepath found in DataFrame"
    assert df["split"].isin(["train", "val", "test"]).all(), "Invalid or missing split values"
    assert df["fold"].isin(list(range(config.N_FOLDS))).all(), "Invalid or missing fold values"
    for s in ["train", "val", "test"]:
        for c in config.CLASS_NAMES:
            count = ((df["split"] == s) & (df["label"] == c)).sum()
            assert count > 0, f"Class {c} is missing from split {s}"
    for f in range(config.N_FOLDS):
        for c in config.CLASS_NAMES:
            count = ((df["fold"] == f) & (df["label"] == c)).sum()
            assert count > 0, f"Class {c} is missing from fold {f}"
    print("All contract assertions passed successfully.")

    print("\n--- STEP 8: Saving splits/index.csv ---")
    final_cols = ["filepath", "label", "label_id", "split", "fold"]
    df = df[final_cols]
    output_csv = config.SPLITS_DIR / "index.csv"
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(df)} rows to: {output_csv}")

    print("\n--- STEP 9: Distribution summary tables ---")
    print("Image counts per class per split:")
    split_table = pd.crosstab(df["label"], df["split"], margins=True)
    print(split_table)

    print("\nImage counts per class per fold:")
    fold_table = pd.crosstab(df["label"], df["fold"], margins=True)
    print(fold_table)

    rp_train = ((df["label"] == "Retinitis_Pigmentosa") & (df["split"] == "train")).sum()
    rp_val = ((df["label"] == "Retinitis_Pigmentosa") & (df["split"] == "val")).sum()
    rp_test = ((df["label"] == "Retinitis_Pigmentosa") & (df["split"] == "test")).sum()
    print(f"\nRetinitis_Pigmentosa count in each split: train={rp_train}, val={rp_val}, test={rp_test}")


if __name__ == "__main__":
    main()
