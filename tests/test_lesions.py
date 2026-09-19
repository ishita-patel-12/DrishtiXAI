"""
tests/test_lesions.py

Basic software-behavior tests for modules/lesions.py.

These tests verify dictionary schema, expected keys, and that no lesion
evidence is fabricated when annotations are absent. They do NOT verify any
"detection accuracy" - this prototype uses expert annotations, not a
model, per the module's scientific-honesty requirement.

Run with:
    python -m pytest tests/test_lesions.py -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.lesions import analyze_lesions

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEMO_CASES_DIR = os.path.join(PROJECT_ROOT, "demo_cases")

REQUIRED_KEYS = {
    "microaneurysms",
    "hemorrhages",
    "hard_exudates",
    "soft_exudates",
    "overlay_path",
}


def _load(case_id):
    path = os.path.join(DEMO_CASES_DIR, case_id, "image.jpg")
    image = cv2.imread(path)
    if image is None:
        pytest.skip(
            f"{path} not found - run generate_demo_assets.py before tests."
        )
    return image


def test_moderate_case_returns_expected_schema_and_some_true_flags():
    image = _load("moderate")
    result = analyze_lesions(image, case_id="moderate")

    assert REQUIRED_KEYS.issubset(result.keys())
    for key in ("microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"):
        assert isinstance(result[key], bool)

    # This demo case was generated with MA / HE / EX masks but no SE mask.
    assert result["microaneurysms"] is True
    assert result["hemorrhages"] is True
    assert result["hard_exudates"] is True
    assert result["soft_exudates"] is False

    assert result["overlay_path"] is not None
    assert os.path.isfile(result["overlay_path"])


def test_normal_case_has_no_fabricated_lesion_evidence():
    image = _load("normal")
    result = analyze_lesions(image, case_id="normal")

    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["microaneurysms"] is False
    assert result["hemorrhages"] is False
    assert result["hard_exudates"] is False
    assert result["soft_exudates"] is False
    assert result["overlay_path"] is None


def test_poor_quality_case_has_no_fabricated_lesion_evidence():
    image = _load("poor_quality")
    result = analyze_lesions(image, case_id="poor_quality")

    assert REQUIRED_KEYS.issubset(result.keys())
    assert all(result[k] is False for k in
               ("microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"))
    assert result["overlay_path"] is None


def test_no_case_id_returns_all_false_without_error():
    image = _load("normal")
    result = analyze_lesions(image, case_id=None)

    assert REQUIRED_KEYS.issubset(result.keys())
    assert all(result[k] is False for k in
               ("microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"))
    assert result["overlay_path"] is None


def test_unknown_case_id_does_not_crash():
    image = _load("normal")
    result = analyze_lesions(image, case_id="this_case_does_not_exist")
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["overlay_path"] is None


def test_missing_image_raises_value_error():
    with pytest.raises(ValueError):
        analyze_lesions(None, case_id="moderate")


def test_does_not_modify_original_image():
    image = _load("moderate")
    original_copy = image.copy()
    _ = analyze_lesions(image, case_id="moderate")
    assert np.array_equal(image, original_copy), "analyze_lesions mutated its input"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
