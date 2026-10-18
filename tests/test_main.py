"""Integration tests for the Cultural Contextual Sentiment Analyzer API.

These tests verify that the FastAPI application responds correctly to health checks
and sentiment analysis requests. They exercise the real routing logic without
mocking the underlying model, ensuring that the request/response contract is
stable.

The test suite expects the following endpoints to be defined in the application:

* ``GET /health`` – returns a JSON payload with a ``status`` key set to ``"ok"``.
* ``POST /analyze`` – accepts a JSON body with a ``text`` field and returns a
  JSON payload containing at least ``sentiment``, ``confidence`` and ``culture``.
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application instance.  The project is expected to expose an
# ``app`` object from ``main.py`` at the repository root.
from main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a reusable TestClient for the FastAPI app.

    The client is instantiated once per test module to avoid the overhead of
    repeatedly starting and stopping the application.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """The health endpoint must be reachable and report a healthy status."""
    response = client.get("/health")
    assert response.status_code == 200, "Health check should return HTTP 200"
    payload = response.json()
    # The contract requires a ``status`` key with value ``ok``.
    assert payload.get("status") == "ok", "Health payload must contain status='ok'"


def test_analyze_endpoint_success(client: TestClient) -> None:
    """A valid request to ``/analyze`` should return a well‑formed sentiment payload."""
    request_body = {"text": "I love the new design of the app!"}
    response = client.post("/analyze", json=request_body)

    assert response.status_code == 200, "Valid analysis request should return HTTP 200"
    result = response.json()

    # Verify that the expected top‑level keys are present.
    expected_keys = {"sentiment", "confidence", "culture"}
    missing = expected_keys - result.keys()
    assert not missing, f"Response is missing expected keys: {missing}"

    # Sentiment must be one of the supported categories.
    assert result["sentiment"] in {"positive", "negative", "neutral"}, (
        "Sentiment value must be 'positive', 'negative', or 'neutral'"
    )

    # Confidence should be a float between 0 and 1.
    confidence = result["confidence"]
    assert isinstance(confidence, float), "Confidence must be a float"
    assert 0.0 <= confidence <= 1.0, "Confidence must be within the range [0.0, 1.0]"

    # Culture should be a non‑empty string.
    culture = result["culture"]
    assert isinstance(culture, str) and culture, "Culture must be a non‑empty string"


def test_analyze_endpoint_missing_text(client: TestClient) -> None:
    """Omitting the required ``text`` field must trigger a validation error."""
    response = client.post("/analyze", json={})
    # FastAPI uses 422 Unprocessable Entity for request validation failures.
    assert response.status_code == 422, "Missing 'text' should result in HTTP 422"


def test_analyze_endpoint_invalid_type(client: TestClient) -> None:
    """Providing a non‑string ``text`` value must also trigger a validation error."""
    response = client.post("/analyze", json={"text": 12345})
    assert response.status_code == 422, "Non‑string 'text' should result in HTTP 422"


def test_analyze_endpoint_long_text(client: TestClient) -> None:
    """The endpoint should handle longer inputs without error."""
    long_text = " ".join(["excellent"] * 200)  # 200-word positive text
    response = client.post("/analyze", json={"text": long_text})
    assert response.status_code == 200, "Long text should be processed successfully"
    result = response.json()
    # Basic sanity checks – same as in the success test.
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert isinstance(result["confidence"], float)
    assert isinstance(result["culture"], str) and result["culture"]