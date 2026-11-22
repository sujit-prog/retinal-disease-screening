# What this file does: trains the retinal disease classifier in two stages
# (frozen head-only, then partial fine-tuning) and saves the model, history
# and config. It also calls predict.py at the end to generate predictions.
# What it reads: splits/index.csv, image files via load_image, config constants.
# What it writes: outputs/models/<run_name>.keras (or _fold<k>.keras),
#   _history.csv, _config.json, and the predictions CSV via predict.py.
# Why it exists: separates training logic from model architecture so either
# teammate can read or modify training without touching the model definition.

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES, CLASS_FOLDERS, RAW_DIR, SPLITS_DIR,
    MODELS_DIR, FIGURES_DIR, RANDOM_SEED
)
from src.preprocessing.image_utils import load_image
from src.model.build_model import build_model, unfreeze_last_layers
from src.model import losses as loss_module
from src.model.predict import run_prediction


# ---------------------------------------------------------------------------
# STEP 0: Check that the data folders and all CSV filepaths actually exist.
# ---------------------------------------------------------------------------
def check_data_integrity():
    # Verify every class folder is on disk.
    for class_name, folder in CLASS_FOLDERS.items():
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            sys.exit(f"ERROR: data folder missing for class '{class_name}': {folder_path}")

    # Verify every filepath in index.csv resolves to a real file.
    index_path = SPLITS_DIR / "index.csv"
    df = pd.read_csv(index_path)
    missing = []
    for fp in df["filepath"]:
        if not (RAW_DIR / fp).exists():
            missing.append(fp)
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} file(s) listed in splits/index.csv are missing "
            f"from RAW_DIR. First few: {missing[:5]}"
        )
    print(f"[check] all {len(df)} filepaths in index.csv exist on disk.")
    return df


# ---------------------------------------------------------------------------
# STEP 1: Load images into memory.
# ---------------------------------------------------------------------------
def load_images_for_rows(rows):
    # Loads all images listed in a DataFrame slice into a numpy array.
    # We load everything into RAM so tf.data can read from arrays (faster than
    # re-reading disk every epoch).
    images = []
    for fp in rows["filepath"]:
        images.append(load_image(fp))
    images = np.stack(images, axis=0).astype(np.float32)
    mb = images.nbytes / (1024 * 1024)
    print(f"  loaded {len(images)} images  ({mb:.1f} MB)")
    return images


# ---------------------------------------------------------------------------
# STEP 2: Build the tf.data pipeline with augmentation.
# ---------------------------------------------------------------------------
def make_aug_layers_normal():
    # Normal augmentation for non-RP images.
    # Mild flipping, rotation and colour jitter to help generalisation.
    return [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
        tf.keras.layers.RandomContrast(0.1),
    ]

def make_aug_layers_rp():
    # Stronger augmentation for RP images because we have very few of them.
    # More rotation and zoom increase the effective variety.
    return [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
        tf.keras.layers.RandomContrast(0.2),
    ]

def apply_aug(image, label, aug_layers):
    # Apply each augmentation layer with training=True so the layer uses
    # random values. Without training=True some layers behave as identity.
    for layer in aug_layers:
        image = layer(image, training=True)
    # Clip to [0, 1] after augmentation so pixel values stay valid.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def build_train_dataset(images, labels, rp_oversample, batch_size):
    # Augmentation and oversampling are applied only to training images,
    # only after the split, so no augmented copy of a training image can
    # leak into validation or test sets (CONTRACT point 3 and task spec).

    rp_id = CLASS_NAMES.index("Retinitis_Pigmentosa")

    # Separate RP images from the rest so we can apply different augmentation.
    rp_mask = labels == rp_id
    other_mask = ~rp_mask

    other_images = images[other_mask]
    other_labels = labels[other_mask]
    rp_images    = images[rp_mask]
    rp_labels    = labels[rp_mask]

    aug_normal = make_aug_layers_normal()
    aug_rp     = make_aug_layers_rp()

    # Build a dataset for non-RP images with mild augmentation.
    ds_other = tf.data.Dataset.from_tensor_slices((other_images, other_labels))
    ds_other = ds_other.map(
        lambda img, lbl: apply_aug(img, lbl, aug_normal),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Build a dataset for RP images with stronger augmentation.
    # .repeat(rp_oversample) is called BEFORE .map so each repeated copy
    # gets its own independent random augmentation (not the same transform
    # applied multiple times).
    if len(rp_images) > 0:
        ds_rp = tf.data.Dataset.from_tensor_slices((rp_images, rp_labels))
        ds_rp = ds_rp.repeat(rp_oversample)
        ds_rp = ds_rp.map(
            lambda img, lbl: apply_aug(img, lbl, aug_rp),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train = ds_other.concatenate(ds_rp)
    else:
        ds_train = ds_other

    # Total size after oversampling (used for the shuffle buffer).
    n_total = len(other_images) + len(rp_images) * rp_oversample

    # Shuffle the whole combined dataset so RP images are spread across batches.
    ds_train = ds_train.shuffle(buffer_size=n_total, seed=RANDOM_SEED,
                                reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, n_total

def build_val_dataset(images, labels, batch_size):
    # Validation images: no augmentation, just batch and prefetch.
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# STEP 3: Select and optionally sub-sample the data rows.
# ---------------------------------------------------------------------------
def select_rows(df, fold, max_images):
    # Fold mode and normal mode are SEPARATE evaluation protocols.
    # Numbers from fold mode must NEVER be compared with numbers from the
    # fixed test split, because the sets overlap and the validation sets
    # are constructed differently.
    if fold is None:
        # Normal mode: use the fixed 70/15/15 split.
        train_rows = df[df["split"] == "train"].copy()
        val_rows   = df[df["split"] == "val"].copy()
    else:
        # Fold mode: images where fold == k are the evaluation set.
        # All other images form the pool; we take 15 percent of that pool
        # as the validation set for early stopping.
        eval_rows  = df[df["fold"] == fold].copy()
        pool_rows  = df[df["fold"] != fold].copy()
        train_rows, val_rows = train_test_split(
            pool_rows,
            test_size=0.15,
            stratify=pool_rows["label_id"],
            random_state=RANDOM_SEED,
        )
        train_rows = train_rows.copy()
        val_rows   = val_rows.copy()

    # Optional: keep at most max_images total (split roughly evenly across classes).
    if max_images is not None:
        per_class = max(1, max_images // len(CLASS_NAMES))
        # Sample per class and concatenate to avoid pandas groupby column loss.
        train_parts = []
        val_parts = []
        for cls_id in range(len(CLASS_NAMES)):
            t = train_rows[train_rows["label_id"] == cls_id]
            train_parts.append(t.sample(min(len(t), per_class),
                                        random_state=RANDOM_SEED))
            v = val_rows[val_rows["label_id"] == cls_id]
            val_parts.append(v.sample(min(len(v), max(1, per_class // 4)),
                                      random_state=RANDOM_SEED))
        train_rows = pd.concat(train_parts, ignore_index=True)
        val_rows   = pd.concat(val_parts,   ignore_index=True)

    train_rows = train_rows.reset_index(drop=True)
    val_rows   = val_rows.reset_index(drop=True)
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# STEP 4: Build the loss function.
# ---------------------------------------------------------------------------
def build_loss(loss_name, effective_labels):
    # effective_labels is the array of labels for training images, with RP
    # already repeated rp_oversample times, so the class weight is not
    # applied on top of oversampling twice.
    class_weights = None
    if loss_name == "ce":
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_name == "weighted_ce":
        class_weights = loss_module.compute_class_weights(effective_labels)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_name == "focal":
        class_weights = loss_module.compute_class_weights(effective_labels)
        loss_fn = loss_module.sparse_focal_loss(gamma=2.0,
                                                class_weights=class_weights)
    else:
        sys.exit(f"ERROR: unknown --loss '{loss_name}'. Use ce, weighted_ce or focal.")
    return loss_fn, class_weights


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the retinal classifier.")
    parser.add_argument("--run_name",        required=True)
    parser.add_argument("--backbone",        default="resnet50v2")
    parser.add_argument("--loss",            default="ce",
                        choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--head_epochs",     type=int,   default=8)
    parser.add_argument("--finetune_epochs", type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--freeze_fraction", type=float, default=0.8,
                        help="Fraction of backbone layers to keep FROZEN in stage 2.")
    parser.add_argument("--rp_oversample",   type=int,   default=1)
    parser.add_argument("--fold",            type=int,   default=None)
    parser.add_argument("--max_images",      type=int,   default=None)
    args = parser.parse_args()

    # Set the global random seed first so everything is reproducible.
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # --- data integrity check ---
    print("=== checking data integrity ===")
    df = check_data_integrity()

    # --- decide file name prefix ---
    if args.fold is not None:
        prefix = f"{args.run_name}_fold{args.fold}"
        pred_tag = f"fold{args.fold}"
    else:
        prefix = args.run_name
        pred_tag = "val"

    model_path   = MODELS_DIR / f"{prefix}.keras"
    history_path = MODELS_DIR / f"{prefix}_history.csv"
    config_path  = MODELS_DIR / f"{prefix}_config.json"
    best_weights = MODELS_DIR / f"{prefix}_best.weights.h5"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- select rows ---
    print("=== selecting data rows ===")
    train_rows, val_rows = select_rows(df, args.fold, args.max_images)
    print(f"train rows : {len(train_rows)}  "
          f"(Healthy={int((train_rows['label_id']==0).sum())}, "
          f"Myopia={(train_rows['label_id']==1).sum()}, "
          f"RP={(train_rows['label_id']==2).sum()})")
    print(f"val rows   : {len(val_rows)}  "
          f"(Healthy={int((val_rows['label_id']==0).sum())}, "
          f"Myopia={(val_rows['label_id']==1).sum()}, "
          f"RP={(val_rows['label_id']==2).sum()})")

    # --- load images ---
    print("=== loading training images ===")
    train_images = load_images_for_rows(train_rows)
    train_labels = train_rows["label_id"].values.astype(np.int32)

    print("=== loading validation images ===")
    val_images = load_images_for_rows(val_rows)
    val_labels = val_rows["label_id"].values.astype(np.int32)

    # --- build effective label list for class weights ---
    # RP rows are repeated rp_oversample times in the training dataset.
    # We build the same repeated label list so the weight reflects the
    # actual training distribution (avoids double-correcting).
    rp_id = CLASS_NAMES.index("Retinitis_Pigmentosa")
    rp_labels = train_labels[train_labels == rp_id]
    other_labels = train_labels[train_labels != rp_id]
    effective_labels = np.concatenate(
        [other_labels] + [rp_labels] * args.rp_oversample
    )

    # --- build loss ---
    loss_fn, class_weights = build_loss(args.loss, effective_labels)

    # --- build tf.data pipelines ---
    print("=== building tf.data pipelines ===")
    ds_train, n_train_eff = build_train_dataset(
        train_images, train_labels, args.rp_oversample, args.batch_size
    )
    ds_val = build_val_dataset(val_images, val_labels, args.batch_size)
    print(f"effective training samples (after oversampling): {n_train_eff}")

    # --- build model ---
    print("=== building model ===")
    model = build_model(args.backbone)

    # -------------------------------------------------------------------
    # STAGE 1: train only the head, backbone frozen.
    # -------------------------------------------------------------------
    print(f"\n=== STAGE 1: head-only training for {args.head_epochs} epoch(s) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    # class_weight is passed to fit for weighted_ce; None otherwise.
    fit_class_weight = class_weights if args.loss == "weighted_ce" else None

    es1 = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    hist1 = model.fit(
        ds_train,
        epochs=args.head_epochs,
        validation_data=ds_val,
        class_weight=fit_class_weight,
        callbacks=[es1],
        verbose=1,
    )
    stage1_epochs = len(hist1.history["loss"])

    # -------------------------------------------------------------------
    # STAGE 2: unfreeze last (1 - freeze_fraction) of backbone layers and
    # fine-tune at a lower learning rate.
    # unfreeze_fraction is the share to OPEN, so pass (1 - freeze_fraction).
    # -------------------------------------------------------------------
    unfreeze_frac = 1.0 - args.freeze_fraction
    print(f"\n=== STAGE 2: fine-tuning last {unfreeze_frac:.0%} of backbone ===")
    unfreeze_last_layers(model, unfreeze_frac)

    # Must recompile after unfreezing so the optimizer picks up the new trainable weights.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    es2 = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_weights),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )

    hist2 = model.fit(
        ds_train,
        epochs=stage1_epochs + args.finetune_epochs,
        initial_epoch=stage1_epochs,  # epoch numbers continue from stage 1
        validation_data=ds_val,
        class_weight=fit_class_weight,
        callbacks=[es2, rlr, mc],
        verbose=1,
    )

    # -------------------------------------------------------------------
    # Load best weights, recompile with a plain string loss so the saved
    # .keras file has no dependency on our custom focal loss function,
    # then save the final model.
    # -------------------------------------------------------------------
    print("\n=== saving model ===")
    if best_weights.exists():
        model.load_weights(str(best_weights))
        print("[train] loaded best checkpoint weights.")

    # Recompile with the plain string loss so the .keras file is self-contained.
    # Custom loss functions like sparse_focal_loss are Python closures and cannot
    # be serialised reliably; using the string form makes the file loadable anywhere.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.save(str(model_path))
    print(f"[train] model saved to {model_path}")

    # Clean up the temporary weights file.
    if best_weights.exists():
        os.remove(best_weights)
        print(f"[train] deleted temporary weights file {best_weights.name}")

    # -------------------------------------------------------------------
    # Save history CSV (epoch numbers continuous across both stages).
    # -------------------------------------------------------------------
    def build_history_df(h1, h2, stage1_len):
        # Stitch stage-1 and stage-2 histories together into one table.
        rows = []
        for i, (lo, ac, vlo, vac) in enumerate(zip(
            h1["loss"], h1["accuracy"], h1["val_loss"], h1["val_accuracy"]
        )):
            rows.append({"epoch": i, "loss": lo, "accuracy": ac,
                         "val_loss": vlo, "val_accuracy": vac})
        for i, (lo, ac, vlo, vac) in enumerate(zip(
            h2["loss"], h2["accuracy"], h2["val_loss"], h2["val_accuracy"]
        )):
            rows.append({"epoch": stage1_len + i, "loss": lo, "accuracy": ac,
                         "val_loss": vlo, "val_accuracy": vac})
        return pd.DataFrame(rows, columns=["epoch", "loss", "accuracy",
                                           "val_loss", "val_accuracy"])

    history_df = build_history_df(
        hist1.history, hist2.history, stage1_epochs
    )
    history_df.to_csv(history_path, index=False)
    print(f"[train] history saved to {history_path}")

    # -------------------------------------------------------------------
    # Save config JSON.
    # -------------------------------------------------------------------
    train_counts = {
        name: int((train_labels == CLASS_NAMES.index(name)).sum())
        for name in CLASS_NAMES
    }
    val_counts = {
        name: int((val_labels == CLASS_NAMES.index(name)).sum())
        for name in CLASS_NAMES
    }
    config_dict = {
        "run_name":        args.run_name,
        "backbone":        args.backbone,
        "loss":            args.loss,
        "head_epochs":     args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "freeze_fraction": args.freeze_fraction,
        "rp_oversample":   args.rp_oversample,
        "fold":            args.fold,
        "max_images":      args.max_images,
        "class_weights":   {str(k): float(v) for k, v in class_weights.items()}
                           if class_weights else None,
        "train_counts":    train_counts,
        "val_counts":      val_counts,
        "random_seed":     RANDOM_SEED,
        "date":            str(date.today()),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[train] config saved to {config_path}")

    # -------------------------------------------------------------------
    # Run predictions on the evaluation set.
    # -------------------------------------------------------------------
    print(f"\n=== running predictions on '{pred_tag}' set ===")
    run_prediction(args.run_name, pred_tag, model=model)

    # Predictions on the TEST split are made separately after training is
    # finished and all hyperparameter tuning is complete, using predict.py.
    # This avoids peeking at test results while still choosing hyperparameters.
    if args.fold is None:
        print("[train] NOTE: to predict on the test split, run:")
        print(f"  python -m src.model.predict --run_name {args.run_name} --tag test")

    print("\n=== training complete ===")


if __name__ == "__main__":
    main()
