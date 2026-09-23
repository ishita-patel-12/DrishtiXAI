import json
import os

notebooks = {
    "01_data_prep_aptos.ipynb": [
        ("# 01 - APTOS 2019 Data Preparation & Preprocessing", "markdown"),
        ("import os\nimport cv2\nimport numpy as np\nimport pandas as pd\n\nprint('Preparing APTOS 2019 dataset manifests...')", "code"),
        ("# Split data into Train (70%), Validation (15%), Test (15%)\n# Apply cropping and 512x512 resizing", "markdown")
    ],
    "02_train_classifier.ipynb": [
        ("# 02 - Train EfficientNet-B0 DR Classifier", "markdown"),
        ("import torch\nimport torch.nn as nn\nfrom app.pipeline.classifier import DRClassifier\n\nmodel = DRClassifier(num_classes=5)\nprint('EfficientNet-B0 classifier initialized.')", "code"),
        ("# CrossEntropyLoss with class weights, AdamW optimizer, CosineAnnealingLR", "markdown")
    ],
    "03_calibrate_temperature.ipynb": [
        ("# 03 - Temperature Scaling Confidence Calibration", "markdown"),
        ("from app.pipeline.calibration import TemperatureCalibrator\n\ncalibrator = TemperatureCalibrator()\nprint(f'Temperature parameter T = {calibrator.temperature}')", "code")
    ],
    "04_gradcam_eval.ipynb": [
        ("# 04 - Grad-CAM Visual Explanation Evaluation", "markdown"),
        ("from app.pipeline.gradcam import GradCAMGenerator\nprint('Grad-CAM evaluation pipeline initialized.')", "code")
    ],
    "05_data_prep_idrid.ipynb": [
        ("# 05 - IDRiD Lesion Segmentation Data Prep", "markdown"),
        ("print('IDRiD Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates mask prep.')", "code")
    ],
    "06_train_lesion_unet.ipynb": [
        ("# 06 - Train U-Net Lesion Segmentation Model", "markdown"),
        ("print('U-Net multi-class lesion segmentation training.')", "code")
    ],
    "07_export_onnx.ipynb": [
        ("# 07 - Export PyTorch Models to ONNX", "markdown"),
        ("from ml.scripts.export_onnx import export_classifier_to_onnx\nprint('ONNX model exporter ready.')", "code")
    ],
    "08_validation_report.ipynb": [
        ("# 08 - Clinical Validation & Benchmark Report", "markdown"),
        ("import matplotlib.pyplot as plt\nfrom sklearn.metrics import roc_curve, auc, confusion_matrix\nprint('Generating validation metrics: AUROC, Sensitivity, Specificity, ECE.')", "code")
    ]
}

os.makedirs("ml/notebooks", exist_ok=True)

for name, cells in notebooks.items():
    nb_cells = []
    for content, cell_type in cells:
        nb_cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "source": [content] if cell_type == "markdown" else [line + "\n" for line in content.split("\n")],
            "outputs": [] if cell_type == "code" else None,
            "execution_count": None if cell_type == "code" else None
        })
    nb_dict = {
        "cells": nb_cells,
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    file_path = os.path.join("ml/notebooks", name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"Created notebook: {file_path}")
