"""
modules/lesions.py
===================

Lesion-level evidence layer for the Diabetic Retinopathy screening pipeline.

SCIENTIFIC HONESTY - READ THIS FIRST
-------------------------------------
This module does NOT run an automated lesion-detection model. No such model
exists in this prototype. Instead, for a small set of prepared demo cases,
it reads pre-existing expert annotation masks (in the style of the IDRiD
segmentation dataset) and overlays them on the retinal image to demonstrate
what a "lesion-level evidence layer" would look like in the final system.

Correct description of this component:
    "Lesion-level evidence layer. For the current prototype, lesion
    localization is demonstrated using expert annotation masks. The final
    system can replace this annotation-based evidence with automated
    lesion segmentation."

Incorrect / forbidden description:
    "Our model detected these lesions."

If no annotation masks are available for a given case, this module returns
all lesion flags as False rather than fabricating evidence.

EXPECTED ANNOTATION FORMAT
---------------------------
For a demo case directory such as:

    demo_cases/moderate/lesion_mask/

this module looks for one binary/grayscale mask image per lesion type,
matched by filename substring (case-insensitive), mirroring the four
lesion categories used by the IDRiD segmentation sub-dataset:

    microaneurysms  <- filename contains "ma" or "microaneurysm"
    hemorrhages     <- filename contains "he" or "hemorrhage"/"haemorrhage"
    hard_exudates   <- filename contains "ex" or "hard_exudate" or "hardexudate"
    soft_exudates   <- filename contains "se" or "soft_exudate" or "softexudate"

Each mask is expected to be a single-channel (or 3-channel, will be
converted) image where non-zero pixels indicate lesion presence at that
pixel. A lesion type is considered "present" if its mask exists and
contains at least one non-zero pixel. If real IDRiD masks are not
available locally, place any correctly-named placeholder mask image in the
same folder - this module does not fabricate masks itself, it only reads
whatever is provided (see `MISSING_ANNOTATION_POLICY` below).

This module does NOT assume the full IDRiD dataset is vendored into the
repository. It only reads whatever prepared masks exist under
`demo_cases/<case_id>/lesion_mask/`.
"""

import os
import glob

import cv2
import numpy as np


# Root of the demo_cases/ directory, resolved relative to this file so the
# module works regardless of the caller's current working directory.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
DEMO_CASES_DIR = os.path.join(_PROJECT_ROOT, "demo_cases")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

# Filename substrings used to identify each lesion type's mask file.
# Matching is case-insensitive and checks whether ANY substring in the list
# appears in the mask filename.
_LESION_FILENAME_HINTS = {
    "microaneurysms": ["microaneurysm", "_ma_", "-ma-", "ma.", "ma_", "1.ma", "ma1"],
    "hemorrhages": ["hemorrhage", "haemorrhage", "_he_", "-he-", "he.", "he_", "2.he", "he1"],
    "hard_exudates": ["hard_exudate", "hardexudate", "_ex_", "-ex-", "ex.", "ex_", "3.ex", "ex1"],
    "soft_exudates": ["soft_exudate", "softexudate", "_se_", "-se-", "se.", "se_", "4.se", "se1"],
}

# Display colors (BGR) used when drawing each lesion type on the overlay,
# chosen so overlapping lesion classes remain visually distinguishable.
_LESION_COLORS_BGR = {
    "microaneurysms": (0, 255, 255),   # yellow
    "hemorrhages": (0, 0, 255),        # red
    "hard_exudates": (0, 255, 0),      # green
    "soft_exudates": (255, 0, 0),      # blue
}

_LESION_KEYS = ["microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"]

# Alpha used when blending lesion color overlays onto the original image.
_OVERLAY_ALPHA = 0.45


def _validate_image(image):
    if image is None:
        raise ValueError("Input image is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Input image must be a NumPy array (e.g. from cv2.imread), "
            f"got {type(image)!r}."
        )
    if image.size == 0:
        raise ValueError("Input image is empty (size 0).")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Input image has unsupported shape {image.shape}; "
            f"expected a 2D grayscale or 3D BGR array."
        )


def _case_mask_dir(case_id):
    return os.path.join(DEMO_CASES_DIR, case_id, "lesion_mask")


def _find_mask_file(mask_dir, hints):
    """Return the path to the first file in mask_dir whose lowercase
    filename contains one of the given hint substrings, or None."""
    if not os.path.isdir(mask_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(mask_dir, "*")))
    for path in candidates:
        filename_lower = os.path.basename(path).lower()
        for hint in hints:
            if hint in filename_lower:
                return path
    return None


def _load_binary_mask(path, target_shape):
    """Load a mask image as a single-channel uint8 array of 0/255 values,
    resized to match target_shape (height, width) if necessary."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    height, width = target_shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return binary_mask


def _load_case_annotations(case_id, image_shape):
    """Look for lesion masks under demo_cases/<case_id>/lesion_mask/.

    Returns a dict: lesion_key -> (present: bool, mask: np.ndarray or None)
    """
    mask_dir = _case_mask_dir(case_id)
    results = {}
    for lesion_key, hints in _LESION_FILENAME_HINTS.items():
        mask_path = _find_mask_file(mask_dir, hints)
        if mask_path is None:
            results[lesion_key] = (False, None)
            continue
        binary_mask = _load_binary_mask(mask_path, image_shape)
        if binary_mask is None or not np.any(binary_mask):
            results[lesion_key] = (False, None)
        else:
            results[lesion_key] = (True, binary_mask)
    return results


def _build_overlay(image, annotations):
    """Blend any available lesion masks onto a copy of the image using
    transparent, distinguishable colors per lesion type."""
    overlay_base = image.copy()
    if overlay_base.ndim == 2:
        overlay_base = cv2.cvtColor(overlay_base, cv2.COLOR_GRAY2BGR)

    color_layer = overlay_base.copy()
    any_drawn = False

    for lesion_key, (present, mask) in annotations.items():
        if not present or mask is None:
            continue
        color = _LESION_COLORS_BGR[lesion_key]
        colored = np.zeros_like(overlay_base)
        colored[:, :] = color
        mask_bool = mask.astype(bool)
        color_layer[mask_bool] = colored[mask_bool]
        any_drawn = True

    if not any_drawn:
        return overlay_base, False

    blended = cv2.addWeighted(
        color_layer, _OVERLAY_ALPHA, overlay_base, 1 - _OVERLAY_ALPHA, 0
    )
    return blended, True


def analyze_lesions(image, case_id=None):
    """Return lesion-level evidence for a fundus image.

    THIS DOES NOT RUN AN AUTOMATED LESION DETECTOR. For prepared demo
    cases (identified by `case_id`, matching a folder under
    demo_cases/<case_id>/lesion_mask/), this function loads expert
    annotation masks if present and overlays them on the image to
    demonstrate the architecture's lesion-level evidence layer. If no
    `case_id` is given, or no annotations exist for that case, all lesion
    flags are returned as False - evidence is never fabricated.

    Parameters
    ----------
    image : np.ndarray
        A BGR (or grayscale) image, e.g. from cv2.imread(). Not modified.
    case_id : str, optional
        Name of a prepared demo case folder under demo_cases/, e.g.
        "moderate". If None or the case/annotations don't exist, this
        function returns an all-False evidence result rather than raising,
        so the rest of the pipeline can keep running.

    Returns
    -------
    dict
        {
            "microaneurysms": bool,
            "hemorrhages": bool,
            "hard_exudates": bool,
            "soft_exudates": bool,
            "overlay_path": str or None,
            # Extra, non-breaking metadata for transparency (safe to ignore):
            "evidence_source": str,   # e.g. "expert_annotation" | "none"
            "note": str,
        }

    Raises
    ------
    ValueError
        If `image` is None, empty, or has an unsupported shape.
    """
    _validate_image(image)
    working_image = image.copy()

    base_result = {key: False for key in _LESION_KEYS}
    base_result["overlay_path"] = None

    if case_id is None:
        base_result["evidence_source"] = "none"
        base_result["note"] = (
            "No case_id provided; no annotation-based lesion evidence "
            "available. This does not mean the image is lesion-free - it "
            "means this prototype has no evidence source for it."
        )
        return base_result

    mask_dir = _case_mask_dir(case_id)
    if not os.path.isdir(mask_dir):
        base_result["evidence_source"] = "none"
        base_result["note"] = (
            f"No lesion_mask/ directory found for case_id={case_id!r}. "
            f"No annotation-based lesion evidence available."
        )
        return base_result

    annotations = _load_case_annotations(case_id, working_image.shape)
    any_present = any(present for present, _ in annotations.values())

    if not any_present:
        base_result["evidence_source"] = "expert_annotation"
        base_result["note"] = (
            "Lesion mask directory exists but no recognized lesion masks "
            "with non-zero pixels were found for this case."
        )
        return base_result

    overlay_image, drawn = _build_overlay(working_image, annotations)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    overlay_filename = f"lesion_overlay_{case_id}.png" if case_id else "lesion_overlay.png"
    overlay_path = os.path.join(OUTPUTS_DIR, overlay_filename)
    cv2.imwrite(overlay_path, overlay_image)

    # Also refresh the generic, spec-required path so it always reflects
    # the most recently analyzed case.
    generic_overlay_path = os.path.join(OUTPUTS_DIR, "lesion_overlay.png")
    cv2.imwrite(generic_overlay_path, overlay_image)

    result = {
        key: annotations[key][0] for key in _LESION_KEYS
    }
    result["overlay_path"] = generic_overlay_path
    result["evidence_source"] = "expert_annotation"
    result["note"] = (
        "Lesion-level evidence layer: localization shown here comes from "
        "prepared expert annotation masks, not an automated lesion "
        "detector. The final system can replace this annotation-based "
        "evidence with automated lesion segmentation."
    )
    return result
