import os
import uuid
from pathlib import Path
from app.config import settings

class StorageService:
    @staticmethod
    def generate_filename(prefix: str, ext: str = ".jpg") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"

    @staticmethod
    def get_upload_path(filename: str) -> Path:
        return settings.OUTPUT_DIR / "uploads" / filename

    @staticmethod
    def get_gradcam_path(filename: str) -> Path:
        return settings.OUTPUT_DIR / "gradcam" / filename

    @staticmethod
    def get_lesion_path(filename: str) -> Path:
        return settings.OUTPUT_DIR / "lesions" / filename

    @staticmethod
    def get_report_path(filename: str) -> Path:
        return settings.OUTPUT_DIR / "reports" / filename

    @staticmethod
    def get_url(path: Path) -> str:
        """Converts local file Path to web accessible URL path."""
        abs_path = path.resolve()
        base_abs = settings.BASE_DIR.resolve()
        try:
            rel_path = abs_path.relative_to(base_abs)
            return "/" + str(rel_path).replace("\\", "/")
        except ValueError:
            return "/" + str(path).replace("\\", "/")
