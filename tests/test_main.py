"""FastAPI contract tests that do not load GraphRAG models or datasets.

The real router and validation run against a synthetic retrieval service.
These tests cover API behaviour, not retrieval relevance or factual accuracy.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api():
    observed_questions: list[str] = []
    stub = ModuleType("app.graphrag_service")

    def fake_run_graphrag_question(question: str) -> dict:
        observed_questions.append(question)
        return {
            "question": question,
            "answer": "This is a synthetic test answer. [E1]",
            "evidence": [
                {
                    "speaker": "Example Speaker",
                    "party": "Lab",
                    "date": "2019-01-01",
                    "agenda": "Example debate",
                    "text": "A deliberately synthetic test passage.",
                    "source": "test",
                    "rerank_score": 0.9,
                }
            ],
            "retrieval": {},
            "coverage": {},
        }

    stub.run_graphrag_question = fake_run_graphrag_question

    # graphrag_routes imports the service when the router is imported.
    # Installing the stub first means no full corpus or model is required.
    with patch.dict(sys.modules, {"app.graphrag_service": stub}):
        from app.main import app

    with TestClient(app) as client:
        yield client, observed_questions


def test_health_check(api):
    client, _ = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_real_graphrag_route_exists(api):
    client, _ = api
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "post" in response.json()["paths"]["/graphrag/ask"]


def test_graphrag_response_contract(api):
    client, seen = api
    question = "What did MPs say about the Brexit customs union?"
    response = client.post("/graphrag/ask", json={"question": question})
    assert response.status_code == 200
    data = response.json()
    assert seen[-1] == question
    assert data["question"] == question
    assert "[E1]" in data["answer"]
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["speaker"] == "Example Speaker"
    assert isinstance(data["retrieval"], dict)
    assert isinstance(data["coverage"], dict)


def test_blank_question_rejected(api):
    client, _ = api
    response = client.post("/graphrag/ask", json={"question": "   "})
    assert response.status_code == 422


def test_question_trimmed(api):
    client, seen = api
    response = client.post("/graphrag/ask", json={"question": "  Brexit?  "})
    assert response.status_code == 200
    assert seen[-1] == "Brexit?"
    assert response.json()["question"] == "Brexit?"
