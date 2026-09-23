from pathlib import Path
from typing import Dict, Any
from app.report.pdf import generate_pdf_report
from app.services.storage import StorageService

class ReportService:
    @staticmethod
    def create_pdf_for_screening(screening_data: Dict[str, Any]) -> str:
        screening_id = screening_data.get("screening_id", "screening_001")
        filename = f"report_{screening_id}.pdf"
        output_path = StorageService.get_report_path(filename)
        
        pdf_file = generate_pdf_report(screening_data, str(output_path))
        return StorageService.get_url(Path(pdf_file))
