import cv2
import numpy as np
from typing import Tuple

def crop_retinal_field(img_bgr: np.ndarray, tol: int = 15) -> np.ndarray:
    """
    Crops black borders around the circular fundus region.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return img_bgr
    
    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)
    
    ymin, ymax = np.where(row_sums > 0)[0][[0, -1]]
    xmin, xmax = np.where(col_sums > 0)[0][[0, -1]]
    
    return img_bgr[ymin:ymax+1, xmin:xmax+1]

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization on the LAB L-channel.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def preprocess_fundus(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    apply_enhancement: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic fundus image preprocessing pipeline.
    Returns:
      (processed_bgr, normalized_rgb_float32)
    """
    # 1. Crop retinal field
    cropped = crop_retinal_field(img_bgr)
    
    # 2. Resize maintaining square bounding box
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Optional CLAHE enhancement if borderline quality
    if apply_enhancement:
        processed = apply_clahe(resized)
    else:
        processed = resized
        
    # 4. Prepare normalized float tensor (RGB, [0, 1])
    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_tensor = (img_rgb - mean) / std
    
    return processed, norm_tensor
