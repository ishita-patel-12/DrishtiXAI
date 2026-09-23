from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models.orm import ScreeningModel, CaseModel
from app.services.report_service import ReportService
from app.config import settings

router = APIRouter(prefix="/api/reports", tags=["Reports"])

@router.get("/{screening_id}", response_model=Dict[str, Any])
def get_report_json(screening_id: str, db: Session = Depends(get_db)):
    scr = db.query(ScreeningModel).filter(ScreeningModel.screening_id == screening_id).first()
    if not scr:
        raise HTTPException(status_code=404, detail="Screening report not found")
        
    case = db.query(CaseModel).filter(CaseModel.case_id == scr.case_id).first()
    
    return {
        "screening_id": scr.screening_id,
        "case_id": scr.case_id,
        "metadata_profile": {
            "age": case.age if case else "—",
            "sex": case.sex if case else "—",
            "diabetesType": case.diabetes_type if case else "—",
            "diabetesDuration": case.diabetes_duration if case else "—",
            "hba1c": case.hba1c if case else "—"
        },
        "quality": {"score": scr.quality_score, "status": scr.quality_status},
        "prediction": {
            "grade": scr.grade,
            "label": scr.grade_label,
            "confidence": scr.calibrated_probability,
            "referable_probability": scr.referable_probability,
            "probabilities": scr.probabilities_json or {}
        },
        "lesions": scr.lesions_json or {},
        "explainability": {"gradcam_path": scr.gradcam_path},
        "decision": {"action": scr.decision_action, "reason": scr.decision_reason},
        "timestamp": scr.created_at.isoformat() if scr.created_at else None
    }

@router.get("/{screening_id}/pdf")
def get_report_pdf(screening_id: str, db: Session = Depends(get_db)):
    data = get_report_json(screening_id, db)
    pdf_url = ReportService.create_pdf_for_screening(data)
    
    # Strip leading slash for local Path lookup
    pdf_path = settings.BASE_DIR / pdf_url.lstrip("/")
    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="PDF generation failed")
        
    return FileResponse(
        path=pdf_path,
        filename=f"DrishtiXAI-Report-{screening_id}.pdf",
        media_type="application/pdf"
    )
