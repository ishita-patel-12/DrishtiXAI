import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models.schemas import AnalysisResponse
from app.services.inference import ScreeningInferenceService

router = APIRouter(prefix="/api", tags=["Analyze"])
inference_service = ScreeningInferenceService()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_retinal_image(
    image: UploadFile = File(...),
    case_id: str = Form("DR-2026-001"),
    metadata: Optional[str] = Form("{}"),
    db: Session = Depends(get_db)
):
    """
    Primary DrishtiXAI Analysis Endpoint:
    Processes fundus image upload through Quality Gate -> DR Classifier -> Temperature Calibration -> Grad-CAM -> Lesion Segmentation -> Evidence Engine -> Decision Policy -> DB Persistence.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image format."
        )
        
    try:
        contents = await image.read()
        metadata_dict = json.loads(metadata) if metadata else {}
        
        result = inference_service.run_screening_pipeline(
            image_bytes=contents,
            case_id=case_id,
            metadata_dict=metadata_dict,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )
