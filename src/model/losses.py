# What this file does: provides class-weight computation and a focal loss function
# for training the retinal disease classifier.
# What it reads: nothing from disk.
# What it writes: nothing (functions return values to the caller).
# Why it exists: separates loss/weight logic from model architecture and training code
# so either teammate can import these utilities without touching the other modules.

import numpy as np
import tensorflow as tf


def compute_class_weights(label_ids):
    # Computes a per-class weight dictionary from a list of integer labels.
    # Formula: weight = total_samples / (num_classes * count_of_that_class)
    #
    # WHY a rarer class gets a bigger weight:
    #   If Retinitis_Pigmentosa has only 139 samples and Healthy has 1024,
    #   the model sees far more Healthy images per epoch. Without correction it
    #   learns to predict Healthy most of the time and still gets a low loss.
    #   A bigger weight on RP examples makes each RP mistake count more in the
    #   gradient update, pushing the model to learn RP features too.

    label_ids = np.array(label_ids, dtype=np.int32)
    total = len(label_ids)
    num_classes = 3  # fixed: Healthy, Myopia, Retinitis_Pigmentosa

    weights = {}
    for class_id in range(num_classes):
        count = np.sum(label_ids == class_id)
        # Avoid division by zero in case a class is somehow absent.
        if count == 0:
            weights[class_id] = 1.0
        else:
            weights[class_id] = total / (num_classes * count)

    print("[compute_class_weights] class weights:")
    print(f"  class 0 (Healthy)              : {weights[0]:.4f}")
    print(f"  class 1 (Myopia)               : {weights[1]:.4f}")
    print(f"  class 2 (Retinitis_Pigmentosa) : {weights[2]:.4f}")
    return weights


def sparse_focal_loss(gamma=2.0, class_weights=None):
    # Returns a loss function compatible with Keras model.compile(loss=...).
    # It works with INTEGER labels (sparse) and SOFTMAX outputs (probabilities).
    #
    # FOCAL LOSS FORMULA (in plain words):
    #   For each sample, let p_t be the probability the model gave to the TRUE class.
    #   Standard cross-entropy loss is  -log(p_t).
    #   Focal loss multiplies that by  (1 - p_t)^gamma.
    #   When the model is very confident and correct (p_t close to 1):
    #     (1 - p_t) is close to 0, so the loss shrinks almost to zero.
    #   When the model is wrong or uncertain (p_t close to 0):
    #     (1 - p_t) is close to 1, so the loss stays large.
    #   This means the model focuses its learning on hard or misclassified examples
    #   (like rare RP images) instead of wasting gradient updates on easy ones.
    #   alpha (from class_weights) additionally scales the loss per class.
    #
    #   Full formula:  loss = -alpha * (1 - p_t)^gamma * log(p_t)

    def loss_fn(y_true, y_pred):
        # y_true can arrive as (batch,) or (batch, 1) depending on the data pipeline.
        # Flatten to (batch,) and cast to int32 so tf.one_hot works correctly.
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Convert integer labels to one-hot vectors shape (batch, 3).
        y_true_onehot = tf.one_hot(y_true, depth=3)

        # Clip probabilities away from 0 and 1 to prevent log(0) = -inf.
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # p_t is the probability the model assigned to the CORRECT class.
        # Summing after element-wise multiplication picks out just that one value.
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)  # shape (batch,)

        # Focal modulating factor: easy examples get a near-zero multiplier.
        focal_factor = tf.pow(1.0 - p_t, gamma)

        # Cross-entropy part of the loss.
        cross_entropy = -tf.math.log(p_t)

        # Build alpha per sample from class_weights dict (or 1.0 if no weights).
        if class_weights is not None:
            # Build a tensor [w0, w1, w2] and gather the right weight per sample.
            weight_tensor = tf.constant(
                [class_weights[0], class_weights[1], class_weights[2]],
                dtype=tf.float32,
            )
            alpha = tf.gather(weight_tensor, y_true)  # shape (batch,)
        else:
            alpha = 1.0

        loss_per_sample = alpha * focal_factor * cross_entropy

        # Return the mean loss over the batch, as Keras expects a scalar.
        return tf.reduce_mean(loss_per_sample)

    return loss_fn


# ---------------------------------------------------------------------------
# Quick sanity check: focal loss with gamma=0 should equal sparse
# categorical cross-entropy (the focal factor becomes (1-p)^0 = 1).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("--- losses.py self-test ---")

    # Small made-up batch: 4 samples, 3 classes.
    np.random.seed(42)
    raw = np.random.rand(4, 3).astype(np.float32)
    # Softmax to make valid probability distributions.
    exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
    y_pred = (exp_raw / exp_raw.sum(axis=1, keepdims=True)).astype(np.float32)
    y_true = np.array([0, 2, 1, 0], dtype=np.int32)

    # Focal loss with gamma=0 should equal standard cross-entropy.
    focal_fn = sparse_focal_loss(gamma=0.0, class_weights=None)
    focal_value = focal_fn(y_true, y_pred).numpy()

    keras_ce = tf.keras.losses.SparseCategoricalCrossentropy()
    keras_value = keras_ce(y_true, y_pred).numpy()

    print(f"focal loss (gamma=0) : {focal_value:.6f}")
    print(f"Keras sparse CE      : {keras_value:.6f}")
    diff = abs(focal_value - keras_value)
    print(f"absolute difference  : {diff:.6f}")
    if diff < 1e-4:
        print("PASS: values are almost equal (difference < 1e-4).")
    else:
        print("FAIL: values differ more than expected.")

    print()
    print("--- compute_class_weights test ---")
    # Simulate realistic label distribution: 1024 Healthy, 500 Myopia, 139 RP.
    label_list = [0] * 1024 + [1] * 500 + [2] * 139
    compute_class_weights(label_list)
