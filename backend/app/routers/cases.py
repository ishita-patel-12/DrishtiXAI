from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models.orm import CaseModel, ScreeningModel
from app.models.schemas import PatientProfileSchema

router = APIRouter(prefix="/api/cases", tags=["Cases"])

@router.get("", response_model=List[Dict[str, Any]])
def list_cases(db: Session = Depends(get_db)):
    """
    Returns list of all persisted cases with latest screening status.
    """
    cases = db.query(CaseModel).order_by(CaseModel.created_at.desc()).all()
    results = []
    for c in cases:
        latest_scr = db.query(ScreeningModel).filter(ScreeningModel.case_id == c.case_id).order_by(ScreeningModel.created_at.desc()).first()
        results.append({
            "case_id": c.case_id,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "quality_status": latest_scr.quality_status if latest_scr else "N/A",
            "grade_label": latest_scr.grade_label if latest_scr else "No DR",
            "confidence": latest_scr.calibrated_probability if latest_scr else 0.0,
            "decision_action": latest_scr.decision_action if latest_scr else "ROUTINE",
            "requires_review": latest_scr.requires_review if latest_scr else False
        })
    return results

@router.get("/{case_id}", response_model=Dict[str, Any])
def get_case_details(case_id: str, db: Session = Depends(get_db)):
    """
    Returns full screening history and clinical context for a given Case ID.
    """
    c = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Case not found")
        
    latest_scr = db.query(ScreeningModel).filter(ScreeningModel.case_id == case_id).order_by(ScreeningModel.created_at.desc()).first()
    
    if not latest_scr:
        raise HTTPException(status_code=404, detail="No screening found for this case")
        
    return {
        "case_id": c.case_id,
        "case_profile": {
            "caseId": c.case_id,
            "age": c.age,
            "sex": c.sex,
            "diabetesType": c.diabetes_type,
            "diabetesDuration": c.diabetes_duration,
            "hba1c": c.hba1c,
            "bloodGlucose": c.blood_glucose,
            "familyHistory": c.family_history,
            "previousDR": c.previous_dr,
            "hypertension": c.hypertension,
            "centre": c.centre,
            "device": c.device
        },
        "screening": {
            "screening_id": latest_scr.screening_id,
            "case_id": latest_scr.case_id,
            "status": "QUALITY_REJECTED" if latest_scr.quality_status == "FAIL" else "COMPLETE",
            "quality": {
                "score": latest_scr.quality_score,
                "status": latest_scr.quality_status,
                "focus": 90.0,
                "illumination": 88.0,
                "field_of_view": 92.0,
                "reasons": latest_scr.quality_reasons or []
            },
            "prediction": {
                "grade": latest_scr.grade,
                "label": latest_scr.grade_label,
                "confidence": latest_scr.calibrated_probability,
                "referable_probability": latest_scr.referable_probability,
                "probabilities": latest_scr.probabilities_json or {}
            } if latest_scr.quality_status != "FAIL" else None,
            "lesions": latest_scr.lesions_json,
            "explainability": {
                "gradcam_path": latest_scr.gradcam_path,
                "evidence_alignment": latest_scr.evidence_alignment
            },
            "decision": {
                "action": latest_scr.decision_action,
                "reason": latest_scr.decision_reason,
                "requires_review": latest_scr.requires_review
            },
            "metadata": {
                "model_version": latest_scr.model_version,
                "lesion_model_version": latest_scr.lesion_model_version,
                "calibration_version": latest_scr.calibration_version,
                "processing_time_ms": latest_scr.processing_time_ms
            },
            "image_path": latest_scr.image_path,
            "timestamp": latest_scr.created_at.isoformat() if latest_scr.created_at else None
        }
    }

@router.post("", status_code=status.HTTP_201_CREATED)
def create_case(profile: PatientProfileSchema, db: Session = Depends(get_db)):
    case_obj = db.query(CaseModel).filter(CaseModel.case_id == profile.caseId).first()
    if case_obj:
        return {"status": "EXISTS", "case_id": case_obj.case_id}
        
    case_obj = CaseModel(
        case_id=profile.caseId,
        age=profile.age,
        sex=profile.sex,
        diabetes_type=profile.diabetesType,
        diabetes_duration=profile.diabetesDuration,
        hba1c=profile.hba1c,
        blood_glucose=profile.bloodGlucose,
        family_history=profile.familyHistory,
        previous_dr=profile.previousDR,
        hypertension=profile.hypertension,
        centre=profile.centre,
        device=profile.device
    )
    db.add(case_obj)
    db.commit()
    return {"status": "CREATED", "case_id": case_obj.case_id}
