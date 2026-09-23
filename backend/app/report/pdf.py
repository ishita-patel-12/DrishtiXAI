import os
import datetime
from pathlib import Path
from typing import Dict, Any

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def generate_pdf_report(screening_data: Dict[str, Any], output_pdf_path: str) -> str:
    """
    Generates a PDF screening report for DrishtiXAI.
    """
    if not HAS_REPORTLAB:
        # Create plain text PDF fallback or simple doc
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        with open(output_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 Fallback Report\n1 0 obj<<>>endobj\ntrailer<<>>%%EOF\n")
        return output_pdf_path

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor('#CF3238'), spaceAfter=4)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#4A5B69'), spaceAfter=12)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#14202B'), spaceBefore=10, spaceAfter=6)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#2A3A4A'))
    bold_style = ParagraphStyle('Bold', parent=normal_style, fontName='Helvetica-Bold')

    # Header
    header_table = Table([[
        Paragraph("DRISHTI<b>X</b>AI", title_style),
        Paragraph("SCREENING REPORT", subtitle_style)
    ]], colWidths=[300, 240])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (1,0), (1,0), 'RIGHT')]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#14202B'), spaceAfter=12))

    # Patient Metadata Table
    case_id = screening_data.get("case_id", "DR-2026-001")
    timestamp = screening_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    meta = screening_data.get("metadata_profile", {})

    meta_data = [
        [Paragraph("<b>Case ID:</b>", normal_style), Paragraph(str(case_id), bold_style), Paragraph("<b>Date & Time:</b>", normal_style), Paragraph(str(timestamp), normal_style)],
        [Paragraph("<b>Age / Sex:</b>", normal_style), Paragraph(f"{meta.get('age', '—')} / {meta.get('sex', '—')}", normal_style), Paragraph("<b>Diabetes Type:</b>", normal_style), Paragraph(str(meta.get('diabetesType', '—')), normal_style)],
        [Paragraph("<b>Duration:</b>", normal_style), Paragraph(f"{meta.get('diabetesDuration', '—')} years", normal_style), Paragraph("<b>HbA1c:</b>", normal_style), Paragraph(f"{meta.get('hba1c', '—')}%", normal_style)]
    ]
    meta_table = Table(meta_data, colWidths=[90, 180, 90, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0'))
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Results Section
    story.append(Paragraph("SCREENING RESULTS & AI PREDICTION", heading_style))
    pred = screening_data.get("prediction", {})
    quality = screening_data.get("quality", {})
    decision = screening_data.get("decision", {})

    res_data = [
        [Paragraph("<b>Image Quality:</b>", normal_style), Paragraph(f"{quality.get('score', 0)}% ({quality.get('status', 'PASS')})", bold_style)],
        [Paragraph("<b>DR Severity Grade:</b>", normal_style), Paragraph(f"{pred.get('label', 'N/A')} (Grade {pred.get('grade', 0)})", bold_style)],
        [Paragraph("<b>Calibrated Confidence:</b>", normal_style), Paragraph(f"{(pred.get('confidence', 0)*100):.1f}%", normal_style)],
        [Paragraph("<b>Referable Probability:</b>", normal_style), Paragraph(f"{(pred.get('referable_probability', 0)*100):.1f}%", normal_style)],
        [Paragraph("<b>Screening Action:</b>", normal_style), Paragraph(f"<b>{decision.get('action', 'ROUTINE')}</b> - {decision.get('reason', '')}", bold_style)]
    ]
    res_table = Table(res_data, colWidths=[150, 390])
    res_table.setStyle(TableStyle([
        ('PADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9'))
    ]))
    story.append(res_table)
    story.append(Spacer(1, 14))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
    disc_text = "<b>MEDICAL DISCLAIMER:</b> AI-assisted screening prototype. Final clinical assessment must be performed by a qualified healthcare professional."
    story.append(Paragraph(disc_text, ParagraphStyle('Disc', parent=normal_style, fontSize=7, leading=9, textColor=colors.HexColor('#64748B'))))

    doc.build(story)
    return output_pdf_path
