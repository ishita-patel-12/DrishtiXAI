import os
import pytest
import numpy as np
import cv2
from app.pipeline.classifier import DRInferencePipeline
from app.pipeline.gradcam import GradCAMGenerator

def test_gradcam_generation(tmp_path):
    pipeline = DRInferencePipeline()
    gradcam_gen = GradCAMGenerator(pipeline.model)

    dummy_norm_tensor = np.random.randn(512, 512, 3).astype(np.float32)
    raw_logits, input_tensor, logits_tensor = pipeline.predict_logits(dummy_norm_tensor)

    cam_np = gradcam_gen.generate_heatmap(input_tensor, logits_tensor, target_class=2)
    assert cam_np.ndim == 2
    assert cam_np.shape[0] > 0 and cam_np.shape[1] > 0
    assert np.max(cam_np) <= 1.0
    assert np.min(cam_np) >= 0.0

    dummy_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    save_file = str(tmp_path / "gradcam_test.jpg")
    out_path = gradcam_gen.save_gradcam_artifact(dummy_bgr, cam_np, save_file)
    assert os.path.exists(out_path)
