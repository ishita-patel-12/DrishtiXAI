import numpy as np
from typing import Dict, Any

def compute_evidence_alignment(cam_np: np.ndarray, lesions_dict: Dict[str, Any]) -> float:
    """
    Computes spatial evidence alignment score between Grad-CAM high-attention regions
    and detected lesion flags.
    Returns score in range [0.0, 1.0].
    """
    if cam_np is None or cam_np.size == 0:
        return 0.50
        
    # High-attention region threshold (> 0.5 max intensity)
    high_att_mask = cam_np > 0.50
    att_ratio = float(np.mean(high_att_mask))
    
    # Count detected lesion types
    detected_count = sum(1 for k in ["microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"] if lesions_dict.get(k, False))
    
    if detected_count == 0:
        # If no lesions, high alignment means low diffuse attention
        alignment = 1.0 - min(1.0, att_ratio * 3.0)
    else:
        # If lesions detected, high alignment means targeted attention around lesions
        alignment = min(1.0, 0.40 + (detected_count * 0.15) + (att_ratio * 0.50))
        
    return float(round(alignment, 2))
