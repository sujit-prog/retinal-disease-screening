# Interface Specification Contract

1. config.py (project root) holds: CLASS_NAMES, CLASS_TO_ID, CLASS_FOLDERS, IMAGE_SIZE = (224, 224), RANDOM_SEED = 42, N_FOLDS = 5, SPLIT_RATIOS = (0.70, 0.15, 0.15), and the paths PROJECT_ROOT, DATA_DIR, RAW_DIR, SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR. DATA_DIR defaults to PROJECT_ROOT/data and can be overridden by the environment variable RETINA_DATA_DIR (needed on Google Colab). RAW_DIR = DATA_DIR. All paths use pathlib. No other file hardcodes any of these values. Everyone imports them from config.
2. splits/index.csv has one row per image with columns: filepath (relative to RAW_DIR, for example Myopia/img_001.jpg), label (class name from CLASS_NAMES), label_id (0, 1 or 2), split (train, val or test), fold (0 to 4).
   - split is a stratified 70/15/15 split by label.
   - fold is a stratified 5-fold assignment over ALL images, separate from split, used only for cross-validation.
3. Images given to the model are RGB, 224x224, float32, pixel values scaled to 0..1. The model code converts these values to whatever its backbone needs. Augmentation happens only inside training code, only on training images and only after the split. Validation and test images are never augmented.
4. Model files (run_name is chosen on the command line):
   - Normal run: outputs/models/<run_name>.keras, outputs/models/<run_name>_history.csv, outputs/models/<run_name>_config.json
   - Fold run (fold k): outputs/models/<run_name>_fold<k>.keras, outputs/models/<run_name>_fold<k>_history.csv, outputs/models/<run_name>_fold<k>_config.json
   - History CSV columns: epoch, loss, accuracy, val_loss, val_accuracy (epoch numbers are continuous across training stages).
5. Predictions: outputs/predictions/<run_name>_<tag>.csv where tag is val, test or fold0 to fold4. Columns: filepath, true_id, pred_id, prob_healthy, prob_myopia, prob_rp (the three probabilities are in class id order 0, 1, 2).
6. Metrics file: outputs/figures/<run_name>_<tag>_metrics.csv with columns: row, precision, recall, f1, support, accuracy, balanced_accuracy. There is one row per class (row = class name) with precision, recall, f1 and support filled in, plus one row named overall with the macro averages in precision, recall and f1, the total support, and accuracy and balanced_accuracy filled in.
7. K-fold summary: outputs/figures/<run_name>_kfold_summary.csv in long format with columns: class, metric, mean, std, min, max. metric is precision, recall or f1. class is a class name or macro.
8. Only src/preprocessing/image_utils.py reads image files. Everyone else imports load_image from it. Signature: load_image(relative_path) takes a path relative to RAW_DIR and returns a float32 numpy array of shape (224, 224, 3) in RGB order with values from 0 to 1.
9. RANDOM_SEED from config is used for every split, shuffle and sample.
10. Everything is run from the project root with python -m, for example: python -m src.preprocessing.build_splits
