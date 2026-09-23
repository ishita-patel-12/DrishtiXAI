# DrishtiXAI System Architecture

## Overview

```
FRONTEND (HTML/CSS/JS)
  |
  v (REST API)
FASTAPI BACKEND
  |-- Image Quality Gate (OpenCV Blur/FOV)
  |-- Fundus Preprocessing
  |-- EfficientNet-B0 DR Classifier
  |-- Temperature Calibration (T=1.24)
  |-- Grad-CAM Heatmap Engine
  |-- U-Net Lesion Segmentation Engine
  |-- Evidence Alignment Engine
  |-- Decision Policy Engine
  |-- PostgreSQL Database (SQLAlchemy ORM)
  |-- ReportLab PDF Generator
```
