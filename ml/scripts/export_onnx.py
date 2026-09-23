import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from app.pipeline.classifier import DRClassifier
from app.config import settings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def export_classifier_to_onnx(output_path: str):
    """
    Exports PyTorch EfficientNet-B0 DR Classifier to ONNX format and verifies numerical tolerance.
    """
    model = DRClassifier(num_classes=5, pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['logits'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )

    print(f"[ONNX Export] Successfully exported model to {output_path}")

    # Verify PyTorch vs ONNX Runtime output tolerance
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_output - ort_output))
    print(f"[ONNX Verification] Maximum numerical difference PyTorch vs ONNX: {max_diff:.6f}")
    assert max_diff < 1e-4, f"ONNX validation numerical mismatch too large: {max_diff}"

if __name__ == "__main__":
    onnx_file = str(settings.WEIGHTS_DIR / f"{settings.CLASSIFIER_VERSION}.onnx")
    export_classifier_to_onnx(onnx_file)
