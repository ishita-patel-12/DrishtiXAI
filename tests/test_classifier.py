import pytest
import numpy as np
from app.pipeline.classifier import DRInferencePipeline, GRADES

def test_classifier_inference():
    pipeline = DRInferencePipeline()
    dummy_norm_tensor = np.random.randn(512, 512, 3).astype(np.float32)

    raw_logits, input_tensor, logits_tensor = pipeline.predict_logits(dummy_norm_tensor)
    assert raw_logits.shape == (5,)

    pred = pipeline.process_predictions(raw_logits, temp=1.24)
    assert 0 <= pred["grade"] <= 4
    assert pred["label"] in GRADES
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["referable_probability"] <= 1.0
    assert len(pred["probabilities"]) == 5
