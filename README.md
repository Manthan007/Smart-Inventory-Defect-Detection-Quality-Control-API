# Smart Inventory Defect Detection & Quality Control API

An end-to-end Deep Learning project for automated steel surface defect segmentation using a custom U-Net architecture built with PyTorch.

The project follows a modular MLOps-inspired architecture consisting of data ingestion, preprocessing, model building, training, evaluation, and experiment reproducibility using DVC.

---

## Problem Statement

Manual inspection of steel surfaces is time-consuming, inconsistent, and prone to human error.

This project automates the quality inspection process by performing semantic segmentation on steel sheet images to identify four different defect types.

Dataset used:
- Severstal Steel Defect Detection (Kaggle)

---

## Features

- Modular project architecture
- YAML-based configuration management
- Custom PyTorch Dataset
- Albumentations image augmentation pipeline
- Custom U-Net implementation
- BCE + Dice Loss
- Dice Score & IoU evaluation metrics
- Mixed Precision Training (AMP)
- Automatic checkpoint saving
- Per-class segmentation evaluation
- Prediction visualization
- Threshold analysis
- DVC integration for experiment reproducibility

---

## Project Structure

```
Smart-Inventory-Defect-Detection-Quality-Control-API/

│
├── artifacts/
│   ├── data_ingestion/
│   ├── model_training/
│   └── model_evaluation/
│
├── config/
│   └── config.yaml
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_loss_and_metrics.ipynb
│   ├── 06_model_training.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   └── SIDD/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
│
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- YAML
- DVC

---

## Model

Architecture:
- Custom U-Net

Input Size:
- 3 × 256 × 1600

Output:
- 4 segmentation masks

Loss Function:
- BCEWithLogitsLoss
- Dice Loss

Optimizer:
- AdamW

Learning Rate Scheduler:
- Cosine Annealing LR

Training:
- Mixed Precision Training (AMP)

---

## Evaluation Metrics

The model is evaluated using

- Validation Loss
- Dice Score
- IoU Score
- Per-Class Dice Score
- Per-Class IoU Score
- Threshold Evaluation
- Prediction Visualization

Example evaluation:

| Metric | Score |
|---------|--------|
| Validation Loss | 0.8859 |
| Dice Score | 0.3375 |
| IoU Score | 0.2997 |

Per-class Dice

| Defect | Dice |
|---------|------|
| Defect 1 | 0.1681 |
| Defect 2 | 0.2265 |
| Defect 3 | 0.4891 |
| Defect 4 | 0.4661 |

---

## Training Pipeline

The project follows a modular pipeline:

```
Data Ingestion
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preparation
        │
        ▼
Model Building
        │
        ▼
Loss & Metrics
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
```

---

## DVC Pipeline

This project uses DVC to version datasets and model artifacts.

Typical workflow:

```bash
dvc repro
```

```bash
dvc status
```

```bash
dvc dag
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Create environment

```bash
conda create -n SIDD python=3.11
```

```bash
conda activate SIDD
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

Training

```bash
python src/SIDD/pipeline/stage_02_model_training.py
```

Evaluation

```bash
python src/SIDD/pipeline/stage_03_model_evaluation.py
```

---

## Future Improvements

- DeepLabV3+
- Attention U-Net
- MLflow experiment tracking
- FastAPI deployment
- Docker support
- CI/CD pipeline
- ONNX/TorchScript export
- Model inference API

---

## Author

**Manthan Sharma**

Bennett University