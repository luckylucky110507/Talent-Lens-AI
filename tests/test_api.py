import json
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:5000"


def _get(path):
    with urlopen(f"{BASE_URL}{path}", timeout=10) as response:
        return response.status, json.loads(response.read())


def test_health_endpoint():
    status, data = _get("/api/health")
    assert status == 200
    assert data["status"] == "ok"


def test_interview_question_endpoint():
    status, data = _get("/api/interview/questions?category=SQL")
    assert status == 200
    assert data["count"] >= 1


def test_model_metrics_endpoint():
    status, data = _get("/api/model/metrics")
    assert status == 200
    assert "random_forest" in data
    assert "ann" in data