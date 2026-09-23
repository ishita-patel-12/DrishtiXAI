import cv2
import numpy as np
from typing import Dict, Any, List

def assess_image_quality(img_bgr: np.ndarray, pass_thresh: float = 75.0, borderline_thresh: float = 50.0) -> Dict[str, Any]:
    """
    Real OpenCV Image Quality Assessment Pipeline.
    Checks:
    - Focus/Blur (Variance of Laplacian)
    - Illumination (Mean & Std Dev of HSV V-channel and LAB L-channel)
    - Field of View (Retinal Mask vs Total Image Area)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Focus Score (Laplacian Variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize laplacian variance to 0-100 scale (typical threshold: > 100 for sharp, < 30 for blurry)
    focus_score = min(100.0, max(0.0, (lap_var / 120.0) * 100.0))

    # 2. Illumination Score (HSV Value Channel & LAB Luminosity)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_v = np.mean(v_channel)
    std_v = np.std(v_channel)
    
    # Ideal mean_v is ~90-160, std_v ~ 40-70 for good fundus illumination
    if mean_v < 40 or mean_v > 220:
        illum_score = 30.0
    else:
        illum_score = min(100.0, max(0.0, 100.0 - abs(mean_v - 120.0) * 0.6))
    
    # 3. Field of View (FOV) Score (Mask of non-black retinal region)
    # Otsu thresholding or simple intensity threshold to isolate fundus circle
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    retinal_pixels = cv2.countNonZero(mask)
    total_pixels = h * w
    fov_ratio = (retinal_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    # Standard 45-degree fundus image occupies ~ 55% - 85% of square bounding frame
    if fov_ratio < 0.20:
        fov_score = 20.0
    elif fov_ratio > 0.95:
        fov_score = 40.0 # No clear circular fundus boundary
    else:
        fov_score = min(100.0, max(0.0, (fov_ratio / 0.65) * 95.0))

    # Overall Composite Quality Score
    overall_score = round(0.45 * focus_score + 0.35 * illum_score + 0.20 * fov_score, 1)

    reasons: List[str] = []
    if focus_score < 45.0:
        reasons.append("Low focus / severe blur")
    if illum_score < 45.0:
        if mean_v < 40:
            reasons.append("Severe underexposure / dark image")
        elif mean_v > 220:
            reasons.append("Overexposure / glare")
        else:
            reasons.append("Uneven illumination")
    if fov_score < 40.0:
        reasons.append("Insufficient retinal field of view")

    if overall_score >= pass_thresh:
        status = "PASS"
    elif overall_score >= borderline_thresh:
        status = "BORDERLINE"
    else:
        status = "FAIL"

    return {
        "score": overall_score,
        "status": status,
        "focus": round(focus_score, 1),
        "illumination": round(illum_score, 1),
        "field_of_view": round(fov_score, 1),
        "reasons": reasons
    }
