import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import init_db

init_db()
client = TestClient(app)

def create_test_image_bytes():
    img = np.ones((512, 512, 3), dtype=np.uint8) * 120
    cv2.circle(img, (256, 256), 200, (30, 80, 180), -1)
    cv2.line(img, (100, 100), (400, 400), (255, 255, 255), 4)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

def test_system_status():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "ONLINE"
    assert "classifier_version" in data

def test_analyze_endpoint():
    img_bytes = create_test_image_bytes()
    files = {"image": ("test_fundus.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {"case_id": "TEST-CASE-99", "metadata": '{"age": "55", "sex": "Female"}'}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    
    assert res_json["case_id"] == "TEST-CASE-99"
    assert "quality" in res_json
    assert "decision" in res_json
    assert "metadata" in res_json
    assert res_json["status"] in ["COMPLETE", "QUALITY_REJECTED"]

def test_cases_endpoint():
    response = client.get("/api/cases")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_review_endpoint():
    # Submit review
    req_body = {
        "review_status": "REVIEWED",
        "reviewer_action": "SPECIALIST_REVIEW",
        "reviewer_notes": "Automated API Test Review"
    }
    response = client.post("/api/review/TEST-CASE-99", json=req_body)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
