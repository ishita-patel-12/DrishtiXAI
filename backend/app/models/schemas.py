from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class QualityResponse(BaseModel):
    score: float
    status: str  # PASS, BORDERLINE, FAIL
    focus: float
    illumination: float
    field_of_view: float
    reasons: List[str] = []

class PredictionResponse(BaseModel):
    grade: int
    label: str
    confidence: float
    referable_probability: float
    probabilities: Dict[str, float]

class LesionsResponse(BaseModel):
    microaneurysms: bool = False
    hemorrhages: bool = False
    hard_exudates: bool = False
    soft_exudates: bool = False
    overlay_path: Optional[str] = None

class ExplainabilityResponse(BaseModel):
    gradcam_path: Optional[str] = None
    evidence_alignment: Optional[float] = None

class DecisionResponse(BaseModel):
    action: str  # REFER, ROUTINE, RECAPTURE, MONITOR, SPECIALIST_REVIEW
    reason: str
    requires_review: bool

class MetadataResponse(BaseModel):
    model_version: str
    lesion_model_version: str
    calibration_version: str
    processing_time_ms: int

class AnalysisResponse(BaseModel):
    screening_id: Optional[str] = None
    case_id: str
    status: str  # COMPLETE, QUALITY_REJECTED, INFERENCE_UNAVAILABLE
    quality: QualityResponse
    prediction: Optional[PredictionResponse] = None
    lesions: Optional[LesionsResponse] = None
    explainability: Optional[ExplainabilityResponse] = None
    decision: DecisionResponse
    metadata: MetadataResponse
    timestamp: Optional[str] = None

class PatientProfileSchema(BaseModel):
    caseId: str
    age: Optional[str] = None
    sex: Optional[str] = None
    diabetesType: Optional[str] = None
    diabetesDuration: Optional[str] = None
    hba1c: Optional[str] = None
    bloodGlucose: Optional[str] = None
    familyHistory: Optional[str] = None
    previousDR: Optional[str] = None
    hypertension: Optional[str] = None
    centre: Optional[str] = None
    device: Optional[str] = None

class ReviewCreateRequest(BaseModel):
    review_status: str = "REVIEWED"
    reviewer_action: Optional[str] = "SPECIALIST_REVIEW"
    reviewer_notes: Optional[str] = None

class SystemStatusResponse(BaseModel):
    api_status: str = "ONLINE"
    database_status: str = "CONNECTED"
    classifier_version: str
    lesion_version: str
    calibration_version: str
    calibration_temp: float
    total_screenings: int
    total_cases: int
    pending_reviews: int
    rejected_quality_count: int
    average_processing_time_ms: float
