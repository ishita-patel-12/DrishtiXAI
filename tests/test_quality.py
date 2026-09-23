import pytest
import numpy as np
import cv2
from app.pipeline.quality import assess_image_quality

def test_high_quality_image():
    # Create a synthetic sharp image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 100
    cv2.circle(img, (256, 256), 200, (30, 80, 180), -1)
    # Add high-contrast features for sharp focus
    cv2.line(img, (100, 100), (400, 400), (255, 255, 255), 5)
    cv2.circle(img, (200, 200), 20, (0, 0, 0), -1)

    result = assess_image_quality(img)
    assert "score" in result
    assert "status" in result
    assert result["status"] in ["PASS", "BORDERLINE"]
    assert isinstance(result["focus"], float)

def test_blurry_poor_quality_image():
    # Create a severely blurred uniform image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 30
    blurred = cv2.GaussianBlur(img, (51, 51), 0)

    result = assess_image_quality(blurred)
    assert result["status"] == "FAIL"
    assert len(result["reasons"]) > 0
