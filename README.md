# DrishtiXAI — Explainable AI System for Diabetic Retinopathy Screening

**Smart India Hackathon Problem Statement**: SIH26038  
**Project**: DrishtiXAI  
**Domain**: AI in Healthcare / Computer Vision / Explainable AI (XAI)

---

## Overview

DrishtiXAI is an end-to-end, Explainable AI system for Diabetic Retinopathy (DR) screening using retinal fundus imagery. It combines real-time image quality validation, 5-class severity classification, temperature-scaled confidence calibration, Grad-CAM visual attention heatmaps, multi-class lesion segmentation, evidence alignment scoring, deterministic decision policy, PostgreSQL database persistence, PDF report generation, and MATLAB/Simulink workflow simulation.

> **Medical Safety Disclaimer**: DrishtiXAI is an AI-assisted screening prototype. Final clinical assessment remains with a qualified healthcare professional.

---

## Architecture Pipeline

```
                 DRISHTIXAI FRONTEND (HTML/CSS/JS)
                                |
                              REST
                                |
                                v
                          FASTAPI BACKEND
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
     QUALITY GATE          DR CLASSIFIER        LESION MODEL
     (OpenCV Blur/FOV)    (EfficientNet-B0)       (U-Net)
          |                     |                     |
          |                     v                     |
          |                CALIBRATION                |
          |                     |                     |
          |                     v                     |
          |                 GRAD-CAM                  |
          |                     |                     |
          +---------------------+---------------------+
                                |
                                v
                         EVIDENCE ENGINE
                                |
                                v
                         DECISION ENGINE
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
           ROUTINE            REFER           RECAPTURE
              |                 |
              +-----------------+
                                |
                                v
                       POSTGRESQL DATABASE
                                |
                   +------------+------------+
                   |                         |
                   v                         v
                REPORTS                   REVIEWS
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) PostgreSQL database server
- (Optional) MATLAB / Simulink for discrete event simulation

### Quick Start (Local Prototype Server)

1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r backend/requirements.txt
   ```

3. **Export Model to ONNX** (Optional):
   ```powershell
   $env:PYTHONPATH="backend"
   python ml/scripts/export_onnx.py
   ```

4. **Launch FastAPI Backend**:
   ```powershell
   $env:PYTHONPATH="backend"
   python backend/app/main.py
   ```
   Open your browser to: `http://127.0.0.1:8000`

5. **Run Test Suite**:
   ```powershell
   $env:PYTHONPATH="backend"
   pytest tests/ -v
   ```

---

## Repository Structure

```
drishtixai/
├── GEMINI.md
├── AGENTS.md
├── PROJECT_STATUS.md
├── README.md
├── docker-compose.yml
│
├── frontend/
│   └── drishtixai.html          # Existing visual interface with real API wiring
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entrypoint
│   │   ├── config.py            # Centralized settings & thresholds
│   │   ├── deps.py              # Database dependencies
│   │   ├── routers/             # API routers (analyze, cases, reports, review, system)
│   │   ├── pipeline/            # Quality, preprocess, classifier, calibration, gradcam, lesions, evidence, decision
│   │   ├── models/              # SQLAlchemy ORM & Pydantic schemas
│   │   ├── services/            # Storage, inference orchestration, report service
│   │   └── report/              # ReportLab PDF generator
│   ├── weights/                 # ONNX & PyTorch weight artifacts
│   ├── outputs/                 # Saved Grad-CAM, lesion overlays, PDF reports
│   └── requirements.txt
│
├── ml/
│   ├── notebooks/               # 01-08 Jupyter notebooks for prep, training, calibration, validation
│   ├── scripts/                 # ONNX export script
│   └── datasets/                # Setup instructions & manifests
│
├── matlab/
│   ├── import_onnx_model.m      # MATLAB ONNX Deep Learning Toolbox importer
│   ├── gradcam_validate.m       # MATLAB Grad-CAM validation
│   ├── validation_metrics.m     # Confusion matrix & ROC metrics
│   └── simulink/
│       ├── screening_workflow.slx.md
│       └── run_simulink_simulation.m
│
└── tests/                       # Complete Pytest test suite
```
