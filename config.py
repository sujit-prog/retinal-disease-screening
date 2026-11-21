"""
Configuration settings for the retinal disease screening project.
This file defines all shared constants, class definitions, and directory paths.
It does not read or write data. All other modules import configuration from here.
"""

import os
from pathlib import Path

# Project root directory containing this config file
PROJECT_ROOT = Path(__file__).resolve().parent

# Base data directory, defaults to project data/ folder but can be overridden by RETINA_DATA_DIR
DATA_DIR = Path(os.environ["RETINA_DATA_DIR"]) if "RETINA_DATA_DIR" in os.environ else PROJECT_ROOT / "data"

# Raw images directory, identical to DATA_DIR
RAW_DIR = DATA_DIR

# Directory for cross-validation and train/val/test split index files
SPLITS_DIR = PROJECT_ROOT / "splits"

# Directory for saved trained model files
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Directory for model prediction output CSV files
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Directory for output plots, figures, and metrics reports
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Fixed class names used across all project modules
CLASS_NAMES = ["Healthy", "Myopia", "Retinitis_Pigmentosa"]

# Mapping from class name to numerical class ID
CLASS_TO_ID = {
    "Healthy": 0,
    "Myopia": 1,
    "Retinitis_Pigmentosa": 2,
}

# Mapping from our class names to the actual folder names on disk inside the raw data folder
# Note: real folder names on disk can differ from our class names (for example spaces or subfolders)
CLASS_FOLDERS = {
    "Healthy": "Original Dataset/Healthy",
    "Myopia": "Original Dataset/Myopia",
    "Retinitis_Pigmentosa": "Original Dataset/Retinitis Pigmentosa",
}

# Target input image size (height, width) for model backbones
IMAGE_SIZE = (224, 224)

# Global random seed used for all splits, shuffles, and sampling for reproducibility
RANDOM_SEED = 42

# Number of folds for stratified cross-validation
N_FOLDS = 5

# Proportions for train, validation, and test dataset splits
SPLIT_RATIOS = (0.70, 0.15, 0.15)
