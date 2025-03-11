# Higgs Boson Classification Project

This repository contains code for classifying particle collision events to identify those that produce the Higgs Boson particle. The project uses ensemble boosting methods with physics-inspired feature engineering to achieve high classification performance.

## Project Overview

The goal of this project is to classify events from the Large Hadron Collider (LHC) at CERN into two classes:
- Events that produce the exotic Higgs Boson particle (signal)
- Events that do not produce the Higgs Boson (background)

Each event is a simulated particle collision represented by 28 features describing the trajectories of decay particles. These simulations are based on realistic data from the ATLAS detector.

## Dataset

The dataset is derived from the HIGGS data set published by Baldi, Sadowski, and Whiteson in their paper "Searching for Exotic Particles in High-Energy Physics with Deep Learning." The original dataset can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS).

## Features

- 28 original features representing properties of particle collisions
- Physics-inspired engineered features including:
  - Radial distances for pairs of features
  - Angles for feature pairs
  - Various transformations based on particle physics domain knowledge

## Models

The project implements and compares three ensemble boosting methods:
1. **XGBoost**: Extreme Gradient Boosting
2. **LightGBM**: Light Gradient Boosting Machine
3. **Gradient Boosting**: Scikit-learn's GradientBoostingClassifier
4. **Weighted Ensemble**: A combination of the above models


## Installation

Download the dataset from the [competition page](https://www.kaggle.com/c/higgs-boson) or the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS) and place the files in the `data/` directory.

## Usage

### Generate Submission

```bash
python scripts/generate_submission.py
```

This script:
- Loads trained models
- Creates a weighted ensemble
- Generates predictions for the test set
- Creates a submission file in the required format

## Model Performance

| Model | AUROC |
|-------|-------|
| XGBoost | 0.87342 |
| LightGBM | 0.88915 |
| Gradient Boosting | 0.86703 |
| **Ensemble (weighted average)** | **0.89271** |

## Hyperparameters

### XGBoost
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8

### LightGBM
- n_estimators: 300
- learning_rate: 0.05
- max_depth: 10
- num_leaves: 64
- subsample: 0.8
- colsample_bytree: 0.8

### Gradient Boosting
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 6
- subsample: 0.8

### Ensemble Weights
- XGBoost: 0.4
- LightGBM: 0.4
- Gradient Boosting: 0.2

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project uses data from the HIGGS dataset published with Baldi, Sadowski, and Whiteson, "Searching for Exotic Particles in High-Energy Physics with Deep Learning."