import cv2
import numpy as np
import torch
import os
from typing import Tuple

class GradCAMGenerator:
    """
    Generates genuine Grad-CAM heatmaps from target convolutional layers of PyTorch models.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.get_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, logits: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Computes Grad-CAM heatmap normalized to [0, 1].
        """
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture gradients
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive contributions
        cam = torch.clamp(cam, min=0)
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if np.max(cam_np) > 0:
            cam_np = cam_np / np.max(cam_np)

        return cam_np

    def save_gradcam_artifact(
        self,
        img_bgr: np.ndarray,
        cam_np: np.ndarray,
        save_path: str
    ) -> str:
        """
        Resizes heatmap to original image dimensions, creates Jet colormap overlay, and saves to file.
        """
        h, w = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to 8-bit heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Create 50/50 overlay with original image
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)
        return save_path
