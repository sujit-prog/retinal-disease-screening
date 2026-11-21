# Retinal Disease Screening Assist

This project is a B.Tech final-year minor project: a screening-assist classifier for retinal fundus images with 3 classes.
It is designed strictly as an educational screening-assist student project and is not a clinical or medical diagnostic tool.
The system uses transfer learning with deep convolutional neural networks in TensorFlow/Keras to screen retinal images.

## Setup

1. Ensure Python 3.10, 3.11, or 3.12 is installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- Place the dataset inside the `data/` directory at the project root.
- The dataset used is the ORIGINAL (non-augmented) Eye Disease Image Dataset.
- The three classes used and their corresponding folders inside `data/` are:
  - `Healthy` -> `Original Dataset/Healthy`
  - `Myopia` -> `Original Dataset/Myopia`
  - `Retinitis_Pigmentosa` -> `Original Dataset/Retinitis Pigmentosa`
- Note: The `data/` directory is ignored by Git and will never be committed to the repository.

## How to Run

Run all commands from the project root in the following order:

```bash
# 1. Generate stratified dataset splits and 5-fold cross-validation index
python -m src.preprocessing.build_splits

# 2. Exploratory Data Analysis notebook
# Open and run all cells in notebooks/01_eda.ipynb

# 3. Train baseline model (ResNet50V2 with Cross-Entropy loss)
python -m src.model.train --run_name baseline --backbone resnet50v2 --loss ce --head_epochs 8 --finetune_epochs 20

# 4. Generate predictions on test split
python -m src.model.predict --run_name baseline --tag test

# 5. Evaluate predictions and compute metrics
python -m src.evaluation.evaluate --run_name baseline --tag test

# 6. Train 5-fold cross-validation models
python -m src.model.train --run_name kfold_weighted --backbone resnet50v2 --loss weighted_ce --fold 0
# Repeat for folds 1 to 4:
python -m src.model.train --run_name kfold_weighted --backbone resnet50v2 --loss weighted_ce --fold 1
python -m src.model.train --run_name kfold_weighted --backbone resnet50v2 --loss weighted_ce --fold 2
python -m src.model.train --run_name kfold_weighted --backbone resnet50v2 --loss weighted_ce --fold 3
python -m src.model.train --run_name kfold_weighted --backbone resnet50v2 --loss weighted_ce --fold 4

# 7. Generate K-fold summary report across folds
python -m src.evaluation.kfold_report --run_name kfold_weighted

# 8. Generate evaluation plots and figures
python -m src.evaluation.plots --plot curves --run_name baseline
python -m src.evaluation.plots --plot f1_compare --run_names baseline weighted focal
python -m src.evaluation.plots --plot rp_spread --run_name kfold_weighted
python -m src.evaluation.plots --plot distribution
```

## Module Ownership

| Module Area | Components | Owner |
| :--- | :--- | :--- |
| Preprocessing | `src/preprocessing/`, dataset splitting, image utilities | Sujit |
| Evaluation | `src/evaluation/`, metrics computation, visual plots | Sujit |
| Documentation | `docs/`, `README.md`, interface specifications | Sujit |
| Model Development | `src/model/`, architectures, loss functions, training, prediction | Narayan |
| Notebooks | `notebooks/01_eda.ipynb`, exploratory data analysis | Narayan |

## Google Colab Usage

When running training runs on Google Colab, mount Google Drive and set the `RETINA_DATA_DIR` environment variable to point to the dataset folder on Google Drive before running any script:
```python
import os
os.environ["RETINA_DATA_DIR"] = "/content/drive/MyDrive/path_to_data"
```
