import os
import cv2
import time
import json
import uuid
import datetime
import numpy as np
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.services.storage import StorageService
from app.pipeline.quality import assess_image_quality
from app.pipeline.preprocess import preprocess_fundus
from app.pipeline.classifier import DRInferencePipeline
from app.pipeline.calibration import TemperatureCalibrator
from app.pipeline.gradcam import GradCAMGenerator
from app.pipeline.lesions import detect_and_segment_lesions
from app.pipeline.evidence import compute_evidence_alignment
from app.pipeline.decision import evaluate_decision_policy
from app.models.orm import CaseModel, ScreeningModel, ReviewModel

class ScreeningInferenceService:
    def __init__(self):
        # Initialize pipeline engines
        self.classifier_pipeline = DRInferencePipeline()
        self.calibrator = TemperatureCalibrator()
        self.gradcam_generator = GradCAMGenerator(self.classifier_pipeline.model)

    def run_screening_pipeline(
        self,
        image_bytes: bytes,
        case_id: str,
        metadata_dict: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        start_time = time.time()
        screening_id = f"SCR-{uuid.uuid4().hex[:8].upper()}"
        timestamp_str = datetime.datetime.utcnow().isoformat()

        # 1. Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image. Invalid format.")

        # Save original upload artifact
        orig_filename = StorageService.generate_filename(f"orig_{case_id}")
        orig_path = StorageService.get_upload_path(orig_filename)
        cv2.imwrite(str(orig_path), img_bgr)
        orig_url = StorageService.get_url(orig_path)

        # 2. Quality Assessment
        quality_res = assess_image_quality(
            img_bgr,
            pass_thresh=settings.QUALITY_PASS_THRESHOLD,
            borderline_thresh=settings.QUALITY_BORDERLINE_THRESHOLD
        )

        # Handle QUALITY FAIL
        if quality_res["status"] == "FAIL":
            processing_time_ms = int((time.time() - start_time) * 1000)
            decision = evaluate_decision_policy("FAIL", 0, 0.0, 0.0)
            
            response = {
                "screening_id": screening_id,
                "case_id": case_id,
                "status": "QUALITY_REJECTED",
                "quality": quality_res,
                "prediction": None,
                "lesions": None,
                "explainability": None,
                "decision": decision,
                "metadata": {
                    "model_version": settings.CLASSIFIER_VERSION,
                    "lesion_model_version": settings.LESION_VERSION,
                    "calibration_version": settings.CALIBRATION_VERSION,
                    "processing_time_ms": processing_time_ms
                },
                "timestamp": timestamp_str
            }
            
            self._save_to_database(db, screening_id, case_id, metadata_dict, orig_url, None, quality_res, None, None, None, None, decision, processing_time_ms)
            return response

        # 3. Preprocessing
        apply_enhance = (quality_res["status"] == "BORDERLINE")
        proc_bgr, norm_tensor = preprocess_fundus(img_bgr, target_size=(512, 512), apply_enhancement=apply_enhance)

        proc_filename = StorageService.generate_filename(f"proc_{case_id}")
        proc_path = StorageService.get_upload_path(proc_filename)
        cv2.imwrite(str(proc_path), proc_bgr)
        proc_url = StorageService.get_url(proc_path)

        # 4. DR Classifier Inference
        raw_logits, input_tensor, logits_tensor = self.classifier_pipeline.predict_logits(norm_tensor)
        
        # 5. Temperature Calibration
        pred_dict = self.classifier_pipeline.process_predictions(raw_logits, temp=self.calibrator.temperature)

        # 6. Grad-CAM Generation
        cam_np = self.gradcam_generator.generate_heatmap(input_tensor, logits_tensor, target_class=pred_dict["grade"])
        gradcam_filename = StorageService.generate_filename(f"gradcam_{screening_id}")
        gradcam_path = StorageService.get_gradcam_path(gradcam_filename)
        self.gradcam_generator.save_gradcam_artifact(proc_bgr, cam_np, str(gradcam_path))
        gradcam_url = StorageService.get_url(gradcam_path)

        # 7. Lesion Segmentation
        lesion_filename = StorageService.generate_filename(f"lesion_{screening_id}")
        lesion_path = StorageService.get_lesion_path(lesion_filename)
        lesions_dict = detect_and_segment_lesions(proc_bgr, str(lesion_path))
        lesions_dict["overlay_path"] = StorageService.get_url(lesion_path)

        # 8. Evidence Alignment
        evidence_alignment = compute_evidence_alignment(cam_np, lesions_dict)
        explainability_dict = {
            "gradcam_path": gradcam_url,
            "evidence_alignment": evidence_alignment
        }

        # 9. Decision Engine
        decision = evaluate_decision_policy(
            quality_status=quality_res["status"],
            grade=pred_dict["grade"],
            confidence=pred_dict["confidence"],
            referable_probability=pred_dict["referable_probability"],
            evidence_alignment=evidence_alignment,
            referral_threshold=settings.REFERRAL_THRESHOLD
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        response = {
            "screening_id": screening_id,
            "case_id": case_id,
            "status": "COMPLETE",
            "quality": quality_res,
            "prediction": pred_dict,
            "lesions": lesions_dict,
            "explainability": explainability_dict,
            "decision": decision,
            "metadata": {
                "model_version": settings.CLASSIFIER_VERSION,
                "lesion_model_version": settings.LESION_VERSION,
                "calibration_version": settings.CALIBRATION_VERSION,
                "processing_time_ms": processing_time_ms
            },
            "timestamp": timestamp_str
        }

        # 10. Persist Record to Database
        self._save_to_database(db, screening_id, case_id, metadata_dict, orig_url, proc_url, quality_res, pred_dict, lesions_dict, gradcam_url, evidence_alignment, decision, processing_time_ms)

        return response

    def _save_to_database(
        self,
        db: Session,
        screening_id: str,
        case_id: str,
        meta: Dict[str, Any],
        orig_url: str,
        proc_url: Optional[str],
        quality_res: Dict[str, Any],
        pred_dict: Optional[Dict[str, Any]],
        lesions_dict: Optional[Dict[str, Any]],
        gradcam_url: Optional[str],
        evidence_alignment: Optional[float],
        decision: Dict[str, Any],
        processing_time_ms: int
    ):
        try:
            # Ensure Case exists
            case_obj = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
            if not case_obj:
                case_obj = CaseModel(
                    case_id=case_id,
                    age=str(meta.get("age", "")),
                    sex=str(meta.get("sex", "")),
                    diabetes_type=str(meta.get("diabetesType", "")),
                    diabetes_duration=str(meta.get("diabetesDuration", "")),
                    hba1c=str(meta.get("hba1c", "")),
                    blood_glucose=str(meta.get("bloodGlucose", "")),
                    family_history=str(meta.get("familyHistory", "")),
                    previous_dr=str(meta.get("previousDR", "")),
                    hypertension=str(meta.get("hypertension", "")),
                    centre=str(meta.get("centre", "")),
                    device=str(meta.get("device", ""))
                )
                db.add(case_obj)
                db.commit()

            # Create Screening Record
            screening_obj = ScreeningModel(
                screening_id=screening_id,
                case_id=case_id,
                image_path=orig_url,
                processed_image_path=proc_url,
                quality_score=quality_res["score"],
                quality_status=quality_res["status"],
                quality_reasons=quality_res.get("reasons", []),
                grade=pred_dict["grade"] if pred_dict else None,
                grade_label=pred_dict["label"] if pred_dict else None,
                raw_probability=pred_dict["confidence"] if pred_dict else None,
                calibrated_probability=pred_dict["confidence"] if pred_dict else None,
                referable_probability=pred_dict["referable_probability"] if pred_dict else None,
                probabilities_json=pred_dict.get("probabilities") if pred_dict else None,
                gradcam_path=gradcam_url,
                lesion_overlay_path=lesions_dict.get("overlay_path") if lesions_dict else None,
                lesions_json=lesions_dict,
                evidence_alignment=evidence_alignment,
                decision_action=decision["action"],
                decision_reason=decision["reason"],
                requires_review=decision["requires_review"],
                model_version=settings.CLASSIFIER_VERSION,
                lesion_model_version=settings.LESION_VERSION,
                calibration_version=settings.CALIBRATION_VERSION,
                processing_time_ms=processing_time_ms
            )
            db.add(screening_obj)

            # Create initial Review Record
            review_obj = ReviewModel(
                review_id=f"REV-{uuid.uuid4().hex[:8].upper()}",
                screening_id=screening_id,
                review_status="PENDING" if decision["requires_review"] else "COMPLETED",
                reviewer_action=decision["action"]
            )
            db.add(review_obj)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error persisting screening to database: {e}")
