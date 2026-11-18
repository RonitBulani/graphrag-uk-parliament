
"""FastAPI contract tests for ParliamentAI.

These tests check routing and the response contract.
The expensive GraphRAG service is mocked.
"""

import pytest
from fastapi.testclient import TestClient

import app.graphrag_routes as graphrag_routes
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_real_graphrag_route_exists(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/graphrag/ask" in paths
    assert "post" in paths["/graphrag/ask"]


def test_graphrag_question_response(client, monkeypatch):
    """Test the real route without executing the retrieval engine."""

    question = "What did MPs say about the Brexit customs union?"

    captured = []

    def mock_graphrag_service(received_question):
        captured.append(received_question)

        return {
            "question": received_question,
            "answer": (
                "This is a test answer supported by "
                "parliamentary evidence. [E1]"
            ),
            "evidence": [
                {
                    "speaker": "Test Speaker",
                    "party": "Lab",
                    "date": "2019-01-01",
                    "agenda": "Test debate",
                    "text": "A sample parliamentary speech.",
                    "source": "test",
                    "rerank_score": 0.9,
                }
            ],
            "retrieval": {},
            "coverage": {},
        }

    monkeypatch.setattr(
        graphrag_routes,
        "run_graphrag_question",
        mock_graphrag_service,
    )

    response = client.post(
        "/graphrag/ask",
        json={"question": question},
    )

    assert response.status_code == 200

    data = response.json()

    assert captured == [question]
    assert data["question"] == question
    assert "[E1]" in data["answer"]
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["speaker"] == "Test Speaker"
    assert isinstance(data["retrieval"], dict)
    assert isinstance(data["coverage"], dict)
