"""
Computer Vision support layer for the Diabetic Retinopathy screening pipeline.

This package is Person 2's contribution to a larger, multi-person Explainable
AI project. It intentionally does NOT contain any DR classification model,
Grad-CAM logic, or application/UI code. It exposes three simple functions
that Person 3 (classification pipeline owner) can import directly:

    from modules.quality import assess_quality, enhance_image
    from modules.lesions import analyze_lesions

See modules/quality.py and modules/lesions.py for details, and the top-level
README.md for the integration contract.
"""
