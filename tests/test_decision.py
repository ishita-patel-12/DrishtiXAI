import pytest
from app.pipeline.decision import evaluate_decision_policy

def test_decision_recapture_on_fail():
    res = evaluate_decision_policy("FAIL", 0, 0.90, 0.05)
    assert res["action"] == "RECAPTURE"
    assert res["requires_review"] is False

def test_decision_referral_on_grade_2():
    res = evaluate_decision_policy("PASS", 2, 0.88, 0.92)
    assert res["action"] == "REFER"
    assert res["requires_review"] is True

def test_decision_routine_on_grade_0():
    res = evaluate_decision_policy("PASS", 0, 0.96, 0.02)
    assert res["action"] == "ROUTINE"
    assert res["requires_review"] is False

def test_decision_specialist_review_on_low_confidence():
    res = evaluate_decision_policy("PASS", 2, 0.35, 0.40)
    assert res["action"] == "SPECIALIST_REVIEW"
    assert res["requires_review"] is True
