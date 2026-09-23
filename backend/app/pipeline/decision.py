from typing import Dict, Any

def evaluate_decision_policy(
    quality_status: str,
    grade: int,
    confidence: float,
    referable_probability: float,
    evidence_alignment: float = 1.0,
    referral_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Deterministic Explainable Decision Policy Engine.
    Rules:
    - Quality FAIL -> RECAPTURE
    - Confidence < 0.45 or Low Evidence Alignment (< 0.30) -> SPECIALIST_REVIEW
    - Referable Probability >= Threshold (0.50) -> REFER
    - Grade 1 (Mild NPDR) -> MONITOR
    - Grade 0 (No DR) -> ROUTINE
    """
    if quality_status == "FAIL":
        return {
            "action": "RECAPTURE",
            "reason": "Image failed quality gate assessment. Recapture required before screening.",
            "requires_review": False
        }
        
    if confidence < 0.45 or evidence_alignment < 0.25:
        return {
            "action": "SPECIALIST_REVIEW",
            "reason": "High model uncertainty or low evidence alignment. Specialist review required.",
            "requires_review": True
        }
        
    if referable_probability >= referral_threshold or grade >= 2:
        return {
            "action": "REFER",
            "reason": f"Referable Diabetic Retinopathy detected (Grade {grade}). Specialist review recommended.",
            "requires_review": True
        }
        
    if grade == 1:
        return {
            "action": "MONITOR",
            "reason": "Mild non-proliferative DR detected. Follow-up screening recommended in 6-12 months.",
            "requires_review": False
        }
        
    return {
        "action": "ROUTINE",
        "reason": "No DR detected. Continue routine annual screening per local protocol.",
        "requires_review": False
    }
