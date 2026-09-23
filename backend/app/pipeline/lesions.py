import cv2
import numpy as np
import os
from typing import Dict, Any, Tuple

def detect_and_segment_lesions(img_bgr: np.ndarray, save_overlay_path: str) -> Dict[str, Any]:
    """
    Real Retinal Lesion Detection and Segmentation Engine.
    Detects:
    - Microaneurysms (red small punctate spots)
    - Hemorrhages (larger red blotches/flames)
    - Hard Exudates (bright yellowish lipid deposits)
    - Soft Exudates / Cotton Wool Spots (pale white fluffy lesions)
    """
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    
    # Dark background mask to restrict detection strictly inside retinal circle
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fundus_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fundus_mask_eroded = cv2.erode(fundus_mask, kernel_erode)

    # Convert to LAB & HSV for color-specific lesion extraction
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Hard Exudates (High Luminosity + High Yellow b-channel)
    exudate_mask = (l_chan > 180) & (b_chan > 145) & (fundus_mask_eroded > 0)
    hard_exudates_detected = bool(np.sum(exudate_mask) > 40)
    
    # 2. Red Lesions (Microaneurysms & Hemorrhages via Green Channel Inversion)
    g_chan = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g_chan)
    
    # Morphological Top-Hat to highlight small dark spots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top_hat = cv2.morphologyEx(255 - g_enhanced, cv2.MORPH_TOPHAT, kernel_small)
    
    ma_mask = (top_hat > 45) & (fundus_mask_eroded > 0)
    microaneurysms_detected = bool(np.sum(ma_mask) > 15)

    # Hemorrhages (Larger dark blobs)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat_large = cv2.morphologyEx(255 - g_enhanced, cv2.MORPH_TOPHAT, kernel_large)
    he_mask = (top_hat_large > 60) & (~ma_mask) & (fundus_mask_eroded > 0)
    hemorrhages_detected = bool(np.sum(he_mask) > 50)

    # 3. Soft Exudates (Pale fluffy white regions)
    soft_exudate_mask = (l_chan > 195) & (b_chan <= 140) & (fundus_mask_eroded > 0)
    soft_exudates_detected = bool(np.sum(soft_exudate_mask) > 80)

    # Draw Lesion Contours on Overlay
    # Microaneurysms = Red
    if microaneurysms_detected:
        contours, _ = cv2.findContours(np.uint8(ma_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (95, 90, 255), 2)
        
    # Hemorrhages = Orange/Red
    if hemorrhages_detected:
        contours, _ = cv2.findContours(np.uint8(he_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (90, 159, 255), 2)

    # Hard Exudates = Yellow
    if hard_exudates_detected:
        contours, _ = cv2.findContours(np.uint8(exudate_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (107, 227, 242), 2)

    # Soft Exudates = Light Blue
    if soft_exudates_detected:
        contours, _ = cv2.findContours(np.uint8(soft_exudate_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 167, 91), 2)

    os.makedirs(os.path.dirname(save_overlay_path), exist_ok=True)
    cv2.imwrite(save_overlay_path, overlay)

    return {
        "microaneurysms": microaneurysms_detected,
        "hemorrhages": hemorrhages_detected,
        "hard_exudates": hard_exudates_detected,
        "soft_exudates": soft_exudates_detected,
        "overlay_path": save_overlay_path
    }
