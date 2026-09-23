import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, Any, Tuple

GRADES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

class DRClassifier(nn.Module):
    """
    EfficientNet-B0 architecture for 5-class Diabetic Retinopathy severity classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(DRClassifier, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_target_layer(self):
        """Returns the final convolutional feature layer for Grad-CAM."""
        return self.backbone.features[-1]


class DRInferencePipeline:
    def __init__(self, weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DRClassifier(num_classes=5, pretrained=True)
        
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()

    def predict_logits(self, norm_tensor: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Input: norm_tensor of shape (H, W, C)
        Returns: (raw_logits_numpy, input_torch_tensor, logits_torch_tensor)
        """
        # Convert (H, W, C) to (1, C, H, W)
        tensor_chw = np.transpose(norm_tensor, (2, 0, 1))
        input_tensor = torch.from_numpy(tensor_chw).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        with torch.enable_grad(): # Keep grad for Grad-CAM
            logits = self.model(input_tensor)
            
        raw_logits = logits.detach().cpu().numpy()[0]
        return raw_logits, input_tensor, logits

    def process_predictions(self, logits_np: np.ndarray, temp: float = 1.0) -> Dict[str, Any]:
        """
        Applies temperature scaling to raw logits and computes calibrated class probabilities.
        """
        scaled_logits = logits_np / max(0.1, temp)
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        # Referable DR = P(Grade 2) + P(Grade 3) + P(Grade 4)
        referable_prob = float(probs[2] + probs[3] + probs[4])
        
        prob_dict = {
            "No DR": float(probs[0]),
            "Mild": float(probs[1]),
            "Moderate": float(probs[2]),
            "Severe": float(probs[3]),
            "Proliferative": float(probs[4])
        }
        
        return {
            "grade": pred_class,
            "label": GRADES[pred_class],
            "confidence": round(confidence, 3),
            "referable_probability": round(referable_prob, 3),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "raw_logits": logits_np.tolist()
        }
