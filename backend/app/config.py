import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Database
    DATABASE_URL: str = "sqlite:///./drishtixai.db"

    # Model Versions
    CLASSIFIER_VERSION: str = "dr_classifier_v0.1.0"
    LESION_VERSION: str = "lesion_unet_v0.1.0"
    CALIBRATION_VERSION: str = "calib_v0.1.0"

    # Thresholds
    REFERRAL_THRESHOLD: float = 0.50
    QUALITY_PASS_THRESHOLD: float = 75.0
    QUALITY_BORDERLINE_THRESHOLD: float = 50.0

    # Paths
    BASE_DIR: Path = BASE_DIR
    OUTPUT_DIR: Path = BASE_DIR / "backend" / "outputs"
    WEIGHTS_DIR: Path = BASE_DIR / "backend" / "weights"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(settings.OUTPUT_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(settings.OUTPUT_DIR / "lesions").mkdir(parents=True, exist_ok=True)
(settings.OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
(settings.OUTPUT_DIR / "uploads").mkdir(parents=True, exist_ok=True)
settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
