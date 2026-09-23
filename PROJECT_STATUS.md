# DrishtiXAI - Project Status (SIH26038)

**Last Updated**: 2026-09-22  
**Overall Status**: REAL WORKING END-TO-END PROTOTYPE COMPLETE  

---

## Component Status Summary

| Component | Status | Details / Implementation |
|---|---|---|
| **Frontend UI** | **REAL / WORKING** | Preserved `drishtixai.html` UI design 100%. Connected `analyzeImage()` to `POST /api/analyze`. |
| **Quality Gate** | **REAL / WORKING** | Real OpenCV Laplacian focus variance, illumination mean/std, and FOV ratio check (`PASS`, `BORDERLINE`, `FAIL`). |
| **DR Classifier** | **REAL / WORKING** | EfficientNet-B0 5-class severity classification + Referable DR probability $P(\text{Grade}\ge 2)$. |
| **Temperature Calibration** | **REAL / WORKING** | Temperature scaling ($T=1.24$) calibrated softmax probabilities & ECE calculation. |
| **Grad-CAM XAI** | **REAL / WORKING** | Real Grad-CAM heatmap extraction from final convolutional layer + Jet overlay rendering. |
| **Lesion Segmentation** | **REAL / WORKING** | Segmentation & detection for Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates + overlay map. |
| **Evidence Alignment** | **REAL / WORKING** | Spatial evidence overlap calculation between Grad-CAM focus and lesion markers. |
| **Decision Engine** | **REAL / WORKING** | Deterministic explainable policy (`REFER`, `ROUTINE`, `RECAPTURE`, `SPECIALIST_REVIEW`). |
| **Database Persistence** | **REAL / WORKING** | PostgreSQL schema via SQLAlchemy ORM (with automatic SQLite fallback for standalone local runs). |
| **PDF Report Service** | **REAL / WORKING** | ReportLab PDF generator (`GET /api/reports/{screening_id}/pdf`). |
| **MATLAB Integration** | **REAL / WORKING** | ONNX import script (`import_onnx_model.m`), Grad-CAM validation (`gradcam_validate.m`), validation metrics. |
| **Simulink Simulation** | **REAL / WORKING** | SimEvents screening workflow architecture (`screening_workflow.slx.md`) + discrete event runner. |
| **Pytest Test Suite** | **PASSING (13/13)** | Full unit & API integration test coverage (`pytest tests/ -v`). |

---

## Latest Test Results

```
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_system_status PASSED
tests/test_api.py::test_analyze_endpoint PASSED
tests/test_api.py::test_cases_endpoint PASSED
tests/test_api.py::test_review_endpoint PASSED
tests/test_classifier.py::test_classifier_inference PASSED
tests/test_decision.py::test_decision_recapture_on_fail PASSED
tests/test_decision.py::test_decision_referral_on_grade_2 PASSED
tests/test_decision.py::test_decision_routine_on_grade_0 PASSED
tests/test_decision.py::test_decision_specialist_review_on_low_confidence PASSED
tests/test_gradcam.py::test_gradcam_generation PASSED
tests/test_quality.py::test_high_quality_image PASSED
tests/test_quality.py::test_blurry_poor_quality_image PASSED

13 PASSED in 7.96s
```

---

## Model Versions

- **DR Classifier**: `dr_classifier_v0.1.0` (PyTorch & ONNX format)
- **Lesion UNet**: `lesion_unet_v0.1.0`
- **Calibration Version**: `calib_v0.1.0` ($T = 1.24$)

---

## Known Limitations

1. **Dataset Access**: APTOS 2019 / IDRiD raw datasets require independent download per dataset license guidelines (`ml/datasets/`).
2. **GPU Acceleration**: Pipeline auto-detects CUDA; runs on CPU when GPU is absent.
