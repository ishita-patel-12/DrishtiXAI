"""
tests/test_quality.py

Basic software-behavior tests for modules/quality.py.

These tests verify that the functions behave correctly and return the
expected schema/types. They do NOT establish clinical validity - the
quality thresholds are prototype engineering values (see quality.py).

Run with:
    python -m pytest tests/test_quality.py -v
or:
    python tests/test_quality.py
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.quality import assess_quality, enhance_image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEMO_CASES_DIR = os.path.join(PROJECT_ROOT, "demo_cases")


def _load(case_id):
    path = os.path.join(DEMO_CASES_DIR, case_id, "image.jpg")
    image = cv2.imread(path)
    if image is None:
        pytest.skip(
            f"{path} not found - run generate_demo_assets.py before tests."
        )
    return image


def _make_sharp_synthetic_image():
    """A synthetic image with high-frequency noise -> should score well on
    focus (variance of Laplacian is high for noisy/sharp content)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    return img


def _make_blurry_image():
    """Take a sharp image and blur it heavily -> low focus score."""
    sharp = _make_sharp_synthetic_image()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=20)
    return blurred


def _make_dark_image():
    return np.full((300, 300, 3), 5, dtype=np.uint8)


# ---------------------------------------------------------------------------
# assess_quality
# ---------------------------------------------------------------------------

def test_valid_normal_image_has_expected_schema():
    image = _load("normal")
    result = assess_quality(image)

    expected_keys = {"score", "status", "focus", "illumination", "field_of_view", "reason"}
    assert expected_keys.issubset(result.keys())
    assert result["status"] in ("PASS", "BORDERLINE", "REJECT")
    for key in ("score", "focus", "illumination", "field_of_view"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of [0,1] range: {result[key]}"


def test_blurry_image_scores_low_focus_and_is_flagged():
    blurry = _make_blurry_image()
    result = assess_quality(blurry)
    assert 0.0 <= result["focus"] <= 1.0
    # A heavily blurred image should not score a high focus value.
    assert result["focus"] < 0.5
    assert result["status"] in ("BORDERLINE", "REJECT")


def test_very_dark_image_scores_low_illumination():
    dark = _make_dark_image()
    result = assess_quality(dark)
    assert 0.0 <= result["illumination"] <= 1.0
    assert result["illumination"] < 0.5
    assert result["status"] in ("BORDERLINE", "REJECT")


def test_poor_quality_demo_case_is_rejected_or_borderline():
    poor = _load("poor_quality")
    result = assess_quality(poor)
    assert result["status"] in ("REJECT", "BORDERLINE")
    if result["status"] == "REJECT":
        assert result["reason"] is not None
        assert isinstance(result["reason"], str)


def test_missing_image_raises_value_error():
    with pytest.raises(ValueError):
        assess_quality(None)


def test_empty_array_raises_value_error():
    with pytest.raises(ValueError):
        assess_quality(np.array([]))


def test_non_ndarray_input_raises_value_error():
    with pytest.raises(ValueError):
        assess_quality("not an image")


def test_score_matches_documented_weights():
    image = _load("normal")
    result = assess_quality(image)
    expected_score = (
        0.4 * result["focus"] +
        0.3 * result["illumination"] +
        0.3 * result["field_of_view"]
    )
    assert abs(result["score"] - round(expected_score, 4)) < 1e-6


# ---------------------------------------------------------------------------
# enhance_image
# ---------------------------------------------------------------------------

def test_enhance_image_returns_same_shape_and_dtype():
    image = _load("normal")
    enhanced = enhance_image(image)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == image.dtype


def test_enhance_image_does_not_modify_original():
    image = _load("normal")
    original_copy = image.copy()
    _ = enhance_image(image)
    assert np.array_equal(image, original_copy), "enhance_image mutated its input"


def test_enhance_image_raises_on_none():
    with pytest.raises(ValueError):
        enhance_image(None)


def test_enhance_image_handles_grayscale_input():
    gray = cv2.cvtColor(_load("normal"), cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    assert enhanced.shape == gray.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
