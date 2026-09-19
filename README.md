# Computer Vision Support Layer — Person 2's Component

This is the Computer Vision (CV) support layer for a larger, multi-person
Explainable AI system for Diabetic Retinopathy (DR) screening. It is
designed to be **imported and called by Person 3's DR classification
pipeline**, not run as a standalone application.

It covers exactly three responsibilities:

1. **Image quality assessment** — is a fundus image usable at all?
2. **Image enhancement** — CLAHE-based contrast enhancement.
3. **Lesion evidence visualization** — for a small set of prepared demo
   cases, using expert annotation masks (not an automated detector).

It does **not** contain a DR classifier, Grad-CAM, a lesion segmentation
model, or any UI/frontend/API code.

## 1. Installation

```bash
pip install -r requirements.txt
```

Only two runtime dependencies: `opencv-python` and `numpy`. (`pytest` is
only needed to run the test suite.)

## 2. Project layout

```
modules/
    __init__.py
    quality.py     # assess_quality(), enhance_image()
    lesions.py     # analyze_lesions()

demo_cases/
    normal/            image.jpg
    moderate/          image.jpg, lesion_mask/*.png
    poor_quality/      image.jpg

outputs/
    enhanced.png
    lesion_overlay.png

tests/
    test_quality.py
    test_lesions.py

demo.py                      # end-to-end usage example
generate_demo_assets.py      # generates SYNTHETIC demo images (see below)
requirements.txt
```

### About `demo_cases/`

This environment does not have access to the real IDRiD dataset, so
`generate_demo_assets.py` generates **synthetic, programmatically-drawn
stand-in images** (a bright circular field on a dark background with
simple blob "lesions") so the pipeline can be demonstrated end-to-end.
These are clearly **not real retinal photographs or real expert
annotations**. To use real data, just replace the files under
`demo_cases/<case>/image.jpg` and `demo_cases/moderate/lesion_mask/*`
with real fundus photos and real IDRiD-style masks — no code changes are
required as long as the folder structure and filename conventions
described in `modules/lesions.py` are followed.

## 3. How Person 3 integrates this module

```python
import cv2
from modules.quality import assess_quality, enhance_image
from modules.lesions import analyze_lesions

image = cv2.imread(path)

quality = assess_quality(image)

if quality["status"] == "REJECT":
    # Do not send to classifier — ask the user to recapture the image.
    print("Rejected:", quality["reason"])
else:
    enhanced = enhance_image(image)
    classification = classify(enhanced)          # Person 3's own code
    evidence = analyze_lesions(image, case_id)    # only meaningful for
                                                   # prepared demo cases
```

The quality and lesion modules **never call the classifier** and have no
dependency on Person 3's code.

## 4. `assess_quality(image)`

```python
def assess_quality(image) -> dict
```

Input: a NumPy BGR (or grayscale) image, e.g. from `cv2.imread()`.

Returns:

```python
{
    "score": 0.7453,          # overall 0-1 quality score
    "status": "PASS",         # "PASS" | "BORDERLINE" | "REJECT"
    "focus": 0.5148,          # 0-1, variance-of-Laplacian based
    "illumination": 0.798,    # 0-1, mean-brightness based
    "field_of_view": 1.0,     # 0-1, bright-region-fraction heuristic
    "reason": None,           # populated with a short string when
                               # status is BORDERLINE or REJECT
}
```

Combined as `score = 0.4*focus + 0.3*illumination + 0.3*field_of_view`.

**All thresholds (focus, illumination, field-of-view, and the
PASS/BORDERLINE/REJECT cut points) are prototype engineering values,
chosen heuristically for this student project. They are not clinically
validated and must not be presented as clinical or regulatory-grade
cutoffs.** See the named constants at the top of `modules/quality.py`
(`FOCUS_LOW_THRESHOLD`, `ILLUMINATION_DARK_THRESHOLD`, `FOV_LOW_FRACTION`,
`QUALITY_PASS_THRESHOLD`, etc.) for the exact values and rationale.

Raises `ValueError` for `None`, empty, or malformed input instead of
letting OpenCV throw a cryptic error.

## 5. `enhance_image(image)`

```python
def enhance_image(image) -> np.ndarray
```

Applies CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`) to the L
(luminance) channel of the image in LAB color space, then converts back
to BGR, with a light denoising pass on the luminance channel only (to
avoid blurring retinal structures). Returns a **new** image; the input is
never modified. Does not save any files and does not depend on the
quality module or the classifier — Person 3 decides when to call it.

## 6. `analyze_lesions(image, case_id=None)`

```python
def analyze_lesions(image, case_id=None) -> dict
```

**This does not run an automated lesion detector.** For prepared demo
cases identified by `case_id` (a folder name under `demo_cases/`), it
looks for expert annotation masks under
`demo_cases/<case_id>/lesion_mask/` and, if found, overlays them on the
image. If no `case_id` is given, or no case/annotations exist, it returns
all lesion flags as `False` — **evidence is never fabricated.**

Returns:

```python
{
    "microaneurysms": True,
    "hemorrhages": True,
    "hard_exudates": True,
    "soft_exudates": False,
    "overlay_path": "outputs/lesion_overlay.png",   # or None
    # Extra, non-breaking fields for transparency (safe to ignore):
    "evidence_source": "expert_annotation",          # or "none"
    "note": "Lesion-level evidence layer: ...",
}
```

The core five keys (`microaneurysms`, `hemorrhages`, `hard_exudates`,
`soft_exudates`, `overlay_path`) are stable and match the integration
contract. The two extra keys are additive and can be safely ignored by
Person 3's code.

### Scientific honesty requirement

Wherever this evidence is surfaced (code comments, UI text, reports),
it **must** be described as:

> "Lesion-level evidence layer. For the current prototype, lesion
> localization is demonstrated using expert annotation masks. The final
> system can replace this annotation-based evidence with automated
> lesion segmentation."

and **must not** be described as "our model detected these lesions" —
no lesion-detection model exists in this prototype.

### Expected annotation format

`demo_cases/<case_id>/lesion_mask/` should contain one mask image per
lesion type, matched by filename substring (case-insensitive):
microaneurysms / hemorrhages / hard_exudates / soft_exudates (see the
`_LESION_FILENAME_HINTS` table in `modules/lesions.py` for the exact
substrings, which mirror IDRiD's four segmentation categories). Each mask
should be a single-channel (or convertible-to-grayscale) image where
non-zero pixels mark lesion locations. If a lesion type's mask file is
absent, that lesion type is reported as `False`.

## 7. Running the demo

```bash
python generate_demo_assets.py   # only once, to create synthetic demo images
python demo.py
```

This prints the quality result, saves `outputs/enhanced.png`, prints the
lesion evidence dictionary, and saves `outputs/lesion_overlay.png` for the
`moderate` case.

## 8. Running the tests

```bash
python -m pytest tests/ -v
```

Tests check: score ranges (0–1), presence of required dictionary keys,
that `status` is one of `PASS`/`BORDERLINE`/`REJECT`, that
`enhance_image` returns a same-shape/dtype image without mutating the
input, and that `analyze_lesions` never fabricates evidence when
annotations are unavailable. **These tests verify software behavior
only — they do not establish clinical validity.**

## 9. Known limitations / assumptions

- Field-of-view is estimated with a simple Otsu-threshold + bright-area
  heuristic, not real retinal segmentation.
- Lesion evidence is only available for cases with prepared annotation
  masks; it is not computed for arbitrary uploaded images.
- Demo images shipped in this repo are synthetic placeholders (see
  `generate_demo_assets.py`) because the real IDRiD dataset was not
  available in this environment; swap in real fundus photos/annotations
  for a realistic demo.
