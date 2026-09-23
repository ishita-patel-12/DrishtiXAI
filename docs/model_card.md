# DrishtiXAI Model Card — DR Classifier v0.1.0

## Model Details
- **Model Name**: DrishtiXAI EfficientNet-B0 DR Classifier
- **Model Architecture**: EfficientNet-B0 transfer learning backbone with custom 5-class classification head
- **Version**: `dr_classifier_v0.1.0`
- **Output Classes**: 
  - `0`: No DR
  - `1`: Mild NPDR
  - `2`: Moderate NPDR
  - `3`: Severe NPDR
  - `4`: Proliferative DR

## Intended Use
- **Primary Use Case**: AI-assisted screening for diabetic retinopathy in resource-constrained community screening camps and primary health centres.
- **Target User**: Trained healthcare worker or ophthalmologist assistant.
- **Out of Scope Use**: Autonomous diagnosis without clinical supervision or secondary opinion.

## Preprocessing & Data Pipeline
- Retinal field cropping (removing black outer margin)
- Bounding square resize ($512 \times 512$)
- Selective CLAHE contrast enhancement for borderline images
- ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$)

## Confidence Calibration
- **Temperature Scaling**: Logits scaled by temperature parameter $T = 1.24$ fitted on validation split.
- **Expected Calibration Error (ECE)**: $0.024$ ($2.4\%$)

## Explainability (XAI)
- **Grad-CAM**: Generates class-activation maps from the final convolutional layer (`features[-1]`).
- **Lesion Evidence**: Multi-class U-Net segmentation maps detecting microaneurysms, hemorrhages, hard exudates, and soft exudates.

## Risk & Uncertainty Mitigation
- Poor quality images ($S < 50.0$) are rejected with an explicit `RECAPTURE` recommendation before classification.
- Low-confidence or high-uncertainty cases ($C < 0.45$) trigger `SPECIALIST_REVIEW`.
