"""
modules/quality.py
===================

Image quality assessment and enhancement for fundus (retinal) images.

This module is part of the Computer Vision support layer of a larger
Diabetic Retinopathy (DR) screening pipeline. It is deliberately simple and
self-contained:

    - It does NOT call, import, or know about the DR classifier.
    - It does NOT contain UI code.
    - It exposes exactly two public functions with stable dictionary /
      image return types so another developer (Person 3) can call them
      directly:

        assess_quality(image) -> dict
        enhance_image(image)  -> np.ndarray (BGR image)

IMPORTANT - PROTOTYPE DISCLAIMER
---------------------------------
All thresholds used in this module (FOCUS_LOW_THRESHOLD, illumination
bounds, field-of-view bounds, PASS/BORDERLINE/REJECT cut points, etc.) are
**prototype engineering thresholds** chosen heuristically for a student
project demo. They are NOT clinically validated values and must not be
presented as clinical or regulatory-grade thresholds.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Prototype thresholds (NOT clinically validated - see module docstring)
# ---------------------------------------------------------------------------

# --- Focus (variance of Laplacian) ---
# Below this raw variance, an image is considered essentially out of focus.
FOCUS_LOW_THRESHOLD = 50.0
# At or above this raw variance, an image is considered comfortably sharp.
FOCUS_GOOD_THRESHOLD = 800.0

# --- Illumination (mean grayscale brightness, 0-255) ---
# Below this mean brightness, the image is considered too dark.
ILLUMINATION_DARK_THRESHOLD = 40.0
# Above this mean brightness, the image is considered too bright/washed out.
ILLUMINATION_BRIGHT_THRESHOLD = 220.0
# The brightness range considered "ideal" sits between these two bounds.
ILLUMINATION_GOOD_LOW = 70.0
ILLUMINATION_GOOD_HIGH = 190.0

# --- Field of view (fraction of frame occupied by the bright fundus region) ---
# Below this fraction of non-background pixels, the FOV is considered too small.
FOV_LOW_FRACTION = 0.20
# At or above this fraction, the FOV is considered comfortably large.
FOV_GOOD_FRACTION = 0.45

# --- Overall quality decision thresholds (on the combined 0-1 score) ---
QUALITY_PASS_THRESHOLD = 0.70
QUALITY_BORDERLINE_THRESHOLD = 0.45
# score < QUALITY_BORDERLINE_THRESHOLD => REJECT


def _validate_image(image):
    """Raise a clear ValueError for invalid inputs instead of letting
    OpenCV fail with a cryptic error later on."""
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
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Input image has unsupported number of channels: {image.shape[2]}."
        )


def _to_gray(image):
    """Return a grayscale copy of the image without modifying the original."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Single-channel image stored as 3D array
    return image[:, :, 0].copy()


def _normalize_piecewise(value, low, high):
    """Linearly map `value` from [low, high] to [0, 1], clamped at the ends.

    This is a simple, deterministic piecewise-linear normalization:
        value <= low   -> 0.0
        value >= high  -> 1.0
        otherwise      -> linear interpolation between 0 and 1
    """
    if high <= low:
        raise ValueError("high threshold must be greater than low threshold")
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _compute_focus_score(gray):
    """Prototype focus score using variance of Laplacian.

    Returns a tuple (focus_score_0_to_1, raw_variance).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_raw = laplacian.var()
    focus_score = _normalize_piecewise(
        focus_raw, FOCUS_LOW_THRESHOLD, FOCUS_GOOD_THRESHOLD
    )
    return focus_score, focus_raw


def _compute_illumination_score(gray):
    """Prototype illumination score based on mean brightness.

    Penalizes both very dark and very bright/washed-out images.
    Returns a tuple (illumination_score_0_to_1, mean_brightness, category)
    where category is one of "GOOD", "BORDERLINE", "POOR" (internal only).
    """
    mean_brightness = float(gray.mean())

    if mean_brightness <= ILLUMINATION_DARK_THRESHOLD or \
            mean_brightness >= ILLUMINATION_BRIGHT_THRESHOLD:
        # Extremely dark or extremely bright/saturated image.
        category = "POOR"
        # Score how far past the extreme bound we are, but keep it low.
        if mean_brightness <= ILLUMINATION_DARK_THRESHOLD:
            extreme_score = _normalize_piecewise(
                mean_brightness, 0.0, ILLUMINATION_DARK_THRESHOLD
            )
        else:
            extreme_score = 1.0 - _normalize_piecewise(
                mean_brightness, ILLUMINATION_BRIGHT_THRESHOLD, 255.0
            )
        # Cap poor-illumination images to a low score band (0.0 - 0.35).
        illumination_score = 0.35 * extreme_score
    elif ILLUMINATION_GOOD_LOW <= mean_brightness <= ILLUMINATION_GOOD_HIGH:
        category = "GOOD"
        illumination_score = 1.0
    else:
        category = "BORDERLINE"
        if mean_brightness < ILLUMINATION_GOOD_LOW:
            illumination_score = 0.35 + 0.65 * _normalize_piecewise(
                mean_brightness, ILLUMINATION_DARK_THRESHOLD, ILLUMINATION_GOOD_LOW
            )
        else:
            illumination_score = 0.35 + 0.65 * (
                1.0 - _normalize_piecewise(
                    mean_brightness, ILLUMINATION_GOOD_HIGH, ILLUMINATION_BRIGHT_THRESHOLD
                )
            )

    return illumination_score, mean_brightness, category


def _compute_fov_score(gray):
    """Prototype field-of-view heuristic.

    Fundus photographs typically show a bright, roughly circular/oval
    retinal region surrounded by a dark background. This heuristic
    thresholds the grayscale image and measures what fraction of the frame
    is occupied by that bright foreground region using a simple Otsu
    threshold + contour area approach.

    This is NOT a real retinal segmentation model - it is a simple,
    documented prototype heuristic intended to distinguish "a fundus image
    with a reasonably large visible retinal field" from "a mostly black,
    mostly blown-out, or otherwise invalid image".

    Returns a tuple (fov_score_0_to_1, foreground_fraction).
    """
    # Blur slightly to reduce noise before thresholding.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's method automatically picks a threshold that separates the
    # bright fundus region from the dark background in typical fundus shots.
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    total_pixels = binary.size
    foreground_pixels = int(np.count_nonzero(binary))
    foreground_fraction = foreground_pixels / total_pixels if total_pixels else 0.0

    # If Otsu labeled the majority of the image as "foreground" in a way
    # that looks like it just picked up on a nearly uniform bright image
    # (e.g. an overexposed frame), we still rely on the fraction; a
    # uniform black image will have foreground_fraction close to 0, and a
    # uniform bright image will have it close to 1, which is an acceptable
    # simplification for this prototype heuristic.
    fov_score = _normalize_piecewise(
        foreground_fraction, FOV_LOW_FRACTION, FOV_GOOD_FRACTION
    )
    return fov_score, foreground_fraction


def _decide_reason(focus, illumination, field_of_view):
    """Pick a single human-readable rejection reason based on the weakest
    of the three component scores."""
    components = {
        "Insufficient focus": focus,
        "Poor illumination": illumination,
        "Insufficient field of view": field_of_view,
    }
    weakest_reason = min(components, key=components.get)
    weakest_value = components[weakest_reason]

    # If nothing is particularly bad individually but the combined score is
    # still low, fall back to a generic reason instead of blaming one
    # component unfairly.
    if weakest_value >= 0.5:
        return "Image quality below prototype threshold"
    return weakest_reason


def assess_quality(image):
    """Assess the usability of a fundus image for downstream DR grading.

    This is a PROTOTYPE quality gate. It combines three simple, documented
    heuristics - focus (variance of Laplacian), illumination (mean
    brightness), and field of view (bright-region fraction) - into a single
    0-1 score and a PASS / BORDERLINE / REJECT status.

    IMPORTANT: The thresholds used here are prototype engineering values,
    not clinically validated cutoffs. See FOCUS_LOW_THRESHOLD,
    FOCUS_GOOD_THRESHOLD, ILLUMINATION_* and FOV_* constants at the top of
    this file.

    Parameters
    ----------
    image : np.ndarray
        A BGR (or grayscale) image as returned by cv2.imread(). The image
        is only read, never modified.

    Returns
    -------
    dict
        {
            "score": float,          # overall 0-1 quality score
            "status": str,           # "PASS" | "BORDERLINE" | "REJECT"
            "focus": float,          # 0-1
            "illumination": float,   # 0-1
            "field_of_view": float,  # 0-1
            "reason": str or None,   # populated when status != "PASS"
        }

    Raises
    ------
    ValueError
        If `image` is None, empty, or has an unsupported shape.
    """
    _validate_image(image)

    # Work on a copy so we never mutate the caller's array.
    working_image = image.copy()
    gray = _to_gray(working_image)

    focus_score, _focus_raw = _compute_focus_score(gray)
    illumination_score, _mean_brightness, _illum_category = _compute_illumination_score(gray)
    fov_score, _fg_fraction = _compute_fov_score(gray)

    # Documented weights (see SKILL / project spec): 0.4 focus, 0.3
    # illumination, 0.3 field of view. Not changed without strong reason.
    score = (
        0.4 * focus_score +
        0.3 * illumination_score +
        0.3 * fov_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score >= QUALITY_PASS_THRESHOLD:
        status = "PASS"
        reason = None
    elif score >= QUALITY_BORDERLINE_THRESHOLD:
        status = "BORDERLINE"
        reason = _decide_reason(focus_score, illumination_score, fov_score)
    else:
        status = "REJECT"
        reason = _decide_reason(focus_score, illumination_score, fov_score)

    return {
        "score": round(score, 4),
        "status": status,
        "focus": round(float(focus_score), 4),
        "illumination": round(float(illumination_score), 4),
        "field_of_view": round(float(fov_score), 4),
        "reason": reason,
    }


def enhance_image(image):
    """Enhance a fundus image using CLAHE on the luminance channel.

    Pipeline: BGR -> LAB -> extract L -> CLAHE -> replace L -> LAB -> BGR.

    This function is independent of the quality module and the classifier:
    Person 3 decides whether/when to call it (e.g. always, or only for
    BORDERLINE images). It never saves files itself - that is left to the
    demo script or caller.

    Parameters
    ----------
    image : np.ndarray
        A BGR image as returned by cv2.imread(). Not modified in place.

    Returns
    -------
    np.ndarray
        A new BGR image (same shape/dtype as input) with contrast-limited
        adaptive histogram equalization applied to its luminance channel.

    Raises
    ------
    ValueError
        If `image` is None, empty, or has an unsupported shape.
    """
    _validate_image(image)

    working_image = image.copy()

    if working_image.ndim == 2:
        # Grayscale input: treat the whole image as the luminance channel.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(working_image)
        return enhanced

    if working_image.shape[2] == 4:
        bgr = cv2.cvtColor(working_image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = working_image

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Mild denoising on the luminance channel only, kept light so retinal
    # structures (vessels, lesions) are not blurred away.
    l_enhanced = cv2.fastNlMeansDenoising(l_enhanced, h=3)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced_bgr
