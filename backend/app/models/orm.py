import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class CaseModel(Base):
    __tablename__ = "cases"

    case_id = Column(String(50), primary_key=True, index=True)
    age = Column(String(10), nullable=True)
    sex = Column(String(20), nullable=True)
    diabetes_type = Column(String(50), nullable=True)
    diabetes_duration = Column(String(20), nullable=True)
    hba1c = Column(String(20), nullable=True)
    blood_glucose = Column(String(20), nullable=True)
    family_history = Column(String(20), nullable=True)
    previous_dr = Column(String(20), nullable=True)
    hypertension = Column(String(20), nullable=True)
    centre = Column(String(100), nullable=True)
    device = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    screenings = relationship("ScreeningModel", back_populates="case", cascade="all, delete-orphan")


class ScreeningModel(Base):
    __tablename__ = "screenings"

    screening_id = Column(String(50), primary_key=True, index=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), nullable=False)
    image_path = Column(String(255), nullable=False)
    processed_image_path = Column(String(255), nullable=True)
    
    # Quality Gate
    quality_score = Column(Float, nullable=False)
    quality_status = Column(String(20), nullable=False)
    quality_reasons = Column(JSON, nullable=True)
    
    # Prediction
    grade = Column(Integer, nullable=True)
    grade_label = Column(String(50), nullable=True)
    raw_probability = Column(Float, nullable=True)
    calibrated_probability = Column(Float, nullable=True)
    referable_probability = Column(Float, nullable=True)
    probabilities_json = Column(JSON, nullable=True)
    
    # Artifacts & Evidence
    gradcam_path = Column(String(255), nullable=True)
    lesion_overlay_path = Column(String(255), nullable=True)
    lesions_json = Column(JSON, nullable=True)
    evidence_alignment = Column(Float, nullable=True)
    
    # Decision Engine
    decision_action = Column(String(50), nullable=False)
    decision_reason = Column(Text, nullable=True)
    requires_review = Column(Boolean, default=False)
    
    # Telemetry
    model_version = Column(String(50), nullable=False)
    lesion_model_version = Column(String(50), nullable=False)
    calibration_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("CaseModel", back_populates="screenings")
    reviews = relationship("ReviewModel", back_populates="screening", cascade="all, delete-orphan")


class ReviewModel(Base):
    __tablename__ = "reviews"

    review_id = Column(String(50), primary_key=True, index=True)
    screening_id = Column(String(50), ForeignKey("screenings.screening_id"), nullable=False)
    review_status = Column(String(50), nullable=False, default="PENDING")
    reviewer_action = Column(String(50), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=datetime.datetime.utcnow)

    screening = relationship("ScreeningModel", back_populates="reviews")


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    model_id = Column(String(50), primary_key=True, index=True)
    classifier_version = Column(String(50), nullable=False)
    lesion_version = Column(String(50), nullable=False)
    calibration_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)
