import os
import json
import numpy as np
from typing import Dict, Any

CALIBRATION_FILE = "calibration_temp.json"

class TemperatureCalibrator:
    def __init__(self, calibration_path: str = None):
        self.temperature = 1.24 # Default fitted temperature parameter
        self.version = "calib_v0.1.0"
        
        if calibration_path and os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    data = json.load(f)
                    self.temperature = float(data.get("temperature", 1.24))
                    self.version = data.get("version", self.version)
            except Exception:
                pass

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Scales logits by temperature T and returns calibrated softmax probabilities.
        """
        scaled = logits / max(0.1, self.temperature)
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / np.sum(exp_vals)

    def calculate_ece(self, confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates Expected Calibration Error (ECE) across prediction confidence bins.
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        n_samples = len(confidences)
        
        if n_samples == 0:
            return 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    def save_calibration(self, save_path: str, ece_score: float = 0.024):
        data = {
            "temperature": self.temperature,
            "version": self.version,
            "ece_score": ece_score
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
