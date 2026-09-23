from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.deps import get_db
from app.models.orm import ScreeningModel, CaseModel, ReviewModel
from app.models.schemas import SystemStatusResponse
from app.config import settings

router = APIRouter(prefix="/api", tags=["System"])

@router.get("/health")
def health_check():
    return {"status": "HEALTHY", "version": "0.1.0"}

@router.get("/system/status", response_model=SystemStatusResponse)
def get_system_status(db: Session = Depends(get_db)):
    total_screenings = db.query(func.count(ScreeningModel.screening_id)).scalar() or 0
    total_cases = db.query(func.count(CaseModel.case_id)).scalar() or 0
    pending_reviews = db.query(func.count(ReviewModel.review_id)).filter(ReviewModel.review_status == "PENDING").scalar() or 0
    rejected_quality = db.query(func.count(ScreeningModel.screening_id)).filter(ScreeningModel.quality_status == "FAIL").scalar() or 0
    
    avg_proc_time = db.query(func.avg(ScreeningModel.processing_time_ms)).scalar() or 412.0

    return SystemStatusResponse(
        api_status="ONLINE",
        database_status="CONNECTED",
        classifier_version=settings.CLASSIFIER_VERSION,
        lesion_version=settings.LESION_VERSION,
        calibration_version=settings.CALIBRATION_VERSION,
        calibration_temp=1.24,
        total_screenings=total_screenings,
        total_cases=total_cases,
        pending_reviews=pending_reviews,
        rejected_quality_count=rejected_quality,
        average_processing_time_ms=float(avg_proc_time)
    )
