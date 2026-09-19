"""
generate_demo_assets.py
========================

Utility script (NOT part of the required module API) that generates
SYNTHETIC placeholder images and lesion masks under demo_cases/, so the
pipeline can be demonstrated in an environment without access to the real
IDRiD dataset.

IMPORTANT: These are synthetic, programmatically-drawn stand-ins that only
mimic the coarse visual structure of fundus photographs (a bright circular
retinal field on a dark background, with simple blob-like "lesions"). They
are NOT real retinal photographs and NOT real IDRiD expert annotations.

To use this module with REAL data instead:
    1. Replace demo_cases/normal/image.jpg with a real, good-quality
       fundus photo.
    2. Replace demo_cases/moderate/image.jpg with a real fundus photo that
       has visible lesions, and place the corresponding real IDRiD-style
       lesion masks in demo_cases/moderate/lesion_mask/ (see
       modules/lesions.py docstring for the expected filename/format).
    3. Replace demo_cases/poor_quality/image.jpg with a real poor-quality
       (blurry/dark/underfilled) fundus photo.

Run this script once (`python generate_demo_assets.py`) before running
demo.py or the test suite if the demo_cases/ images do not already exist.
"""

import os

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_CASES_DIR = os.path.join(_THIS_DIR, "demo_cases")

IMAGE_SIZE = 512


def _make_synthetic_fundus(seed=0, add_vessels=True):
    """Draw a bright, roughly-circular "fundus" region on a dark
    background, with faint vessel-like lines, using a fixed seed for
    reproducibility."""
    rng = np.random.default_rng(seed)
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
    radius = int(IMAGE_SIZE * 0.42)

    # Base retinal field color (warm orange/red, like a real fundus photo).
    base_color = (30, 90, 180)  # BGR
    cv2.circle(img, center, radius, base_color, thickness=-1)

    # Add smooth brightness variation so it isn't perfectly flat.
    noise = rng.normal(0, 8, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=15)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32) + noise
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    if add_vessels:
        # A handful of curved, darker "vessel" strokes radiating from the
        # optic-disc-like region.
        disc_center = (center[0] + int(radius * 0.3), center[1])
        cv2.circle(img, disc_center, int(radius * 0.18), (60, 140, 220), -1)
        for i in range(10):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(radius * 0.5, radius * 0.9)
            end_x = int(disc_center[0] + length * np.cos(angle))
            end_y = int(disc_center[1] + length * np.sin(angle))
            thickness = rng.integers(1, 3)
            cv2.line(img, disc_center, (end_x, end_y), (15, 45, 90), int(thickness))

    # Mask out everything outside the circular field to black (simulates
    # the dark surround typical of fundus camera captures).
    field_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.circle(field_mask, center, radius, 255, -1)
    img = cv2.bitwise_and(img, img, mask=field_mask)

    return img, center, radius


def _add_blob_lesions(mask_canvas, center, radius, rng, count, blob_size_range):
    """Draw a few small filled circular blobs within the retinal field to
    stand in for a lesion type's annotation mask."""
    for _ in range(count):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, radius * 0.85)
        x = int(center[0] + dist * np.cos(angle))
        y = int(center[1] + dist * np.sin(angle))
        blob_radius = rng.integers(*blob_size_range)
        cv2.circle(mask_canvas, (x, y), int(blob_radius), 255, -1)
    return mask_canvas


def generate_normal_case():
    out_dir = os.path.join(DEMO_CASES_DIR, "normal")
    os.makedirs(out_dir, exist_ok=True)
    img, _, _ = _make_synthetic_fundus(seed=1)
    cv2.imwrite(os.path.join(out_dir, "image.jpg"), img)
    print(f"Wrote {os.path.join(out_dir, 'image.jpg')}")


def generate_moderate_case():
    out_dir = os.path.join(DEMO_CASES_DIR, "moderate")
    mask_dir = os.path.join(out_dir, "lesion_mask")
    os.makedirs(mask_dir, exist_ok=True)

    img, center, radius = _make_synthetic_fundus(seed=2)
    cv2.imwrite(os.path.join(out_dir, "image.jpg"), img)
    print(f"Wrote {os.path.join(out_dir, 'image.jpg')}")

    rng = np.random.default_rng(42)

    lesion_specs = {
        "microaneurysms_MA.png": dict(count=8, blob_size_range=(2, 4)),
        "hemorrhages_HE.png": dict(count=4, blob_size_range=(5, 10)),
        "hard_exudates_EX.png": dict(count=5, blob_size_range=(6, 12)),
        # soft_exudates intentionally omitted for this synthetic case to
        # demonstrate a lesion type that is genuinely absent/False.
    }

    for filename, spec in lesion_specs.items():
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        canvas = _add_blob_lesions(canvas, center, radius, rng, **spec)
        mask_path = os.path.join(mask_dir, filename)
        cv2.imwrite(mask_path, canvas)
        print(f"Wrote {mask_path}")


def generate_poor_quality_case():
    out_dir = os.path.join(DEMO_CASES_DIR, "poor_quality")
    os.makedirs(out_dir, exist_ok=True)
    img, _, _ = _make_synthetic_fundus(seed=3, add_vessels=False)

    # Heavily blur (destroys focus) and darken (destroys illumination) to
    # simulate a genuinely ungradeable capture.
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=12)
    darkened = (blurred.astype(np.float32) * 0.15).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "image.jpg"), darkened)
    print(f"Wrote {os.path.join(out_dir, 'image.jpg')}")


if __name__ == "__main__":
    generate_normal_case()
    generate_moderate_case()
    generate_poor_quality_case()
    print("\nSynthetic demo assets generated. These are NOT real fundus "
          "photographs or real IDRiD annotations - see this script's "
          "module docstring for how to swap in real data.")
