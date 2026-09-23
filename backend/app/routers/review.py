import datetime
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models.orm import ScreeningModel, ReviewModel
from app.models.schemas import ReviewCreateRequest

router = APIRouter(prefix="/api/review", tags=["Review"])

@router.post("/{screening_id}", response_model=Dict[str, Any])
def submit_specialist_review(
    screening_id: str,
    review_req: ReviewCreateRequest,
    db: Session = Depends(get_db)
):
    scr = db.query(ScreeningModel).filter(ScreeningModel.screening_id == screening_id).first()
    if not scr:
        # Check if case_id was passed instead of screening_id
        scr = db.query(ScreeningModel).filter(ScreeningModel.case_id == screening_id).order_by(ScreeningModel.created_at.desc()).first()
        
    if not scr:
        raise HTTPException(status_code=404, detail="Screening record not found")

    rev = db.query(ReviewModel).filter(ReviewModel.screening_id == scr.screening_id).first()
    if not rev:
        rev = ReviewModel(
            review_id=f"REV-{uuid.uuid4().hex[:8].upper()}",
            screening_id=scr.screening_id
        )
        db.add(rev)

    rev.review_status = review_req.review_status
    rev.reviewer_action = review_req.reviewer_action or "SPECIALIST_REVIEW"
    rev.reviewer_notes = review_req.reviewer_notes
    rev.reviewed_at = datetime.datetime.utcnow()

    scr.requires_review = False # Mark review resolved
    db.commit()

    return {
        "status": "SUCCESS",
        "review_id": rev.review_id,
        "screening_id": scr.screening_id,
        "review_status": rev.review_status,
        "reviewer_action": rev.reviewer_action
    }
