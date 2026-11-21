# What this file does: defines a function that builds a Keras transfer-learning model
# for 3-class retinal disease classification using a chosen pretrained backbone.
# What it reads: nothing from disk (weights are downloaded from the internet on first use).
# What it writes: nothing (the caller saves the model).
# Why it exists: separates the model architecture from training logic so both teammates
# can import build_model without knowing training details.

import sys
import tensorflow as tf
from config import IMAGE_SIZE

# ---------------------------------------------------------------------------
# BACKBONE REGISTRY
# ---------------------------------------------------------------------------
# Each backbone expects a different pixel-value range, so we store both the
# Keras application class and a Rescaling layer config alongside it.
#
# Why each backbone needs its own rescaling:
#   ResNet50V2 and MobileNetV2 were trained expecting values in [-1, 1].
#   Each was trained with a specific preprocessing step, and we must match it
#   so the pretrained features make sense for our images.
#
# Why we use a Rescaling layer instead of a Lambda layer:
#   A Lambda layer serialises by saving the Python bytecode, which can break
#   when the model is loaded in a different Python session or TF version.
#   A Rescaling layer is a named Keras layer that saves and loads cleanly.

BACKBONES = {
    "resnet50v2": {
        "class": tf.keras.applications.ResNet50V2,
        # scale 0..1 to -1..1: new_val = pixel * 2.0 - 1.0
        "rescaling": tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
    },
    "mobilenet_v2": {
        "class": tf.keras.applications.MobileNetV2,
        # same range as ResNet50V2: scale 0..1 to -1..1
        "rescaling": tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),
    },
}


def build_model(backbone_name, dropout=0.3):
    # Builds and returns a Keras model ready for the first training stage.
    # The head (new layers) is trainable. The backbone is frozen so we do not
    # destroy the pretrained ImageNet weights before the head has converged.
    # backbone_name must be one of the keys in BACKBONES.
    # dropout controls the Dropout rate before the final Dense layer.

    if backbone_name not in BACKBONES:
        # Stop clearly if the caller made a typo or used an unsupported backbone.
        sys.exit(
            f"ERROR: '{backbone_name}' is not a supported backbone. "
            f"Choose one of: {list(BACKBONES.keys())}"
        )

    entry = BACKBONES[backbone_name]

    # Input layer: images arrive as (224, 224, 3) float32 in range 0..1
    # (CONTRACT point 3 - the caller always passes normalised images).
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")

    # Rescale to the range the backbone expects (each backbone is different).
    x = entry["rescaling"](inputs)

    # Load pretrained backbone without its top classification head.
    # include_top=False keeps only the convolutional feature extractor.
    # weights="imagenet" downloads pretrained weights on first use.
    # We pass input_shape explicitly so the backbone knows our image size and
    # loads the matching pretrained weights without any warning.
    # We name it "backbone" so unfreeze_last_layers can find it by name.
    backbone = entry["class"](
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="backbone",
    )

    # Freeze all backbone layers so stage-1 training only updates the new head.
    # We call the backbone with training=False so its BatchNormalization layers
    # always use their stored running statistics, not the small-batch statistics.
    # This is important because fine-tuning with training=True on BN can corrupt
    # the learned statistics when the batch size is small.
    backbone.trainable = False
    x = backbone(x, training=False)

    # Global average pooling converts the feature map (H, W, C) to a 1-D vector (C,).
    # It is simpler and less prone to overfitting than a Flatten layer.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dropout randomly zeroes out activations during training to reduce overfitting.
    x = tf.keras.layers.Dropout(dropout)(x)

    # Final Dense layer with softmax gives a probability for each of the 3 classes.
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"retina_{backbone_name}")

    # Print parameter counts so the student can see how many weights are trainable
    # (the head only) versus frozen (the backbone).
    trainable_count = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_trainable_count = sum(
        tf.size(w).numpy() for w in model.non_trainable_weights
    )
    print(f"[build_model] backbone: {backbone_name}")
    print(f"[build_model] trainable params    : {trainable_count:,}")
    print(f"[build_model] non-trainable params: {non_trainable_count:,}")

    return model


def unfreeze_last_layers(model, fraction):
    # Sets the last `fraction` of the backbone layers to trainable.
    # For example fraction=0.2 means the last 20 percent of backbone layers.
    # Every BatchNormalization layer is kept frozen even if it falls in that range,
    # because unfreezing BN on a small dataset tends to destroy the learned statistics.
    #
    # WHY two-stage fine-tuning on a small dataset:
    #   Stage 1 (frozen backbone): trains only the new head. This is fast and stable
    #   because the random head weights do not cause large gradient updates that could
    #   destroy the pretrained features in the backbone.
    #   Stage 2 (top layers unfrozen): once the head is reasonable, we gently adjust
    #   the top backbone layers that learn higher-level features relevant to our images.
    #   Fine-tuning the WHOLE backbone on a small dataset risks overfitting, so we only
    #   touch a small fraction at the end of the network.
    #
    # NOTE: after calling this function the model MUST be recompiled before training,
    # because the optimizer state (momentum, adaptive learning rate) needs to be reset
    # for the newly trainable weights.

    backbone = model.get_layer("backbone")
    all_layers = backbone.layers
    n_total = len(all_layers)

    # Calculate how many layers from the end should become trainable.
    n_unfreeze = max(1, int(n_total * fraction))
    cutoff = n_total - n_unfreeze

    backbone.trainable = True  # Allow individual layers to be changed.

    trainable_count = 0
    for i, layer in enumerate(all_layers):
        if i < cutoff:
            # Layer is in the frozen portion - keep it frozen.
            layer.trainable = False
        else:
            # Layer is in the last fraction. Freeze BN layers regardless.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                trainable_count += 1

    print(f"[unfreeze_last_layers] total backbone layers : {n_total}")
    print(f"[unfreeze_last_layers] fraction requested    : {fraction}")
    print(f"[unfreeze_last_layers] trainable backbone layers (non-BN): {trainable_count}")
    print("[unfreeze_last_layers] REMINDER: recompile the model before continuing training.")

    return model
