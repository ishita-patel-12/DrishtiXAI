"""
demo.py
=======

Simple demonstration of how Person 3 would use this Computer Vision
support layer. This script does NOT run any DR classifier - where the
classifier would be called is marked with a comment.

Usage:
    python generate_demo_assets.py   # only needed once, if demo_cases/
                                      # images do not already exist
    python demo.py
"""

import os

import cv2

from modules.quality import assess_quality, enhance_image
from modules.lesions import analyze_lesions


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_CASES_DIR = os.path.join(THIS_DIR, "demo_cases")
OUTPUTS_DIR = os.path.join(THIS_DIR, "outputs")


def run_case(case_id):
    print(f"\n=== Case: {case_id} ===")
    image_path = os.path.join(DEMO_CASES_DIR, case_id, "image.jpg")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {image_path}. "
              f"Run generate_demo_assets.py first.")
        return

    # 1. Quality gate
    quality = assess_quality(image)
    print("Quality result:", quality)

    if quality["status"] == "REJECT":
        print("-> Image rejected. Would prompt for recapture; "
              "not sending to classifier.")
        return

    # 2. Enhancement (independent of the classifier / quality decision)
    enhanced = enhance_image(image)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    enhanced_path = os.path.join(OUTPUTS_DIR, "enhanced.png")
    cv2.imwrite(enhanced_path, enhanced)
    print(f"-> Enhanced image saved to {enhanced_path}")

    # 3. Classification would happen here in the real pipeline:
    #        classification = classify(enhanced)
    #    Person 2's modules never call the classifier directly.

    # 4. Lesion-level evidence (annotation-based prototype, see
    #    modules/lesions.py docstring for the honesty requirements)
    evidence = analyze_lesions(image, case_id=case_id)
    print("Lesion evidence:", evidence)


if __name__ == "__main__":
    for case in ["normal", "moderate", "poor_quality"]:
        run_case(case)
