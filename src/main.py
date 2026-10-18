"""src/main.py
FastAPI entry point for the Cultural Contextual Sentiment Analyzer.

The service exposes a single `/sentiment` endpoint that receives a piece of text
(and optionally a cultural identifier) and returns sentiment predictions
adjusted for the supplied cultural context.  A lightweight health‑check endpoint
is also provided.

The implementation uses:
- **FastAPI** for the web framework.
- **Transformers** pipeline for out‑of‑the‑box sentiment analysis.
- **scikit‑learn** (imported for compliance with the tech stack; a simple rule‑based
  adjustment is demonstrated, but the hook can be replaced with a trained model).

All heavy objects are loaded once at application start‑up to minimise latency.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline

# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #
class SentimentRequest(BaseModel):
    """Payload for a sentiment analysis request.

    Attributes
    ----------
    text: str
        The raw text to be analysed. Must be non‑empty.
    culture: Optional[str]
        An optional cultural identifier (e.g., ``"US"``, ``"JP"``). The service
        applies a deterministic adjustment to the raw sentiment score based on
        this value. If omitted, no adjustment is performed.
    """

    text: str = Field(..., description="Input text to analyze")
    culture: Optional[str] = Field(
        None,
        description="Cultural context identifier (e.g., 'US', 'JP')",
        max_length=10,
    )


class SentimentResult(BaseModel):
    """Single sentiment prediction returned to the client."""

    label: str = Field(..., description="Sentiment label (e.g., POSITIVE, NEGATIVE)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score after cultural adjustment")
    culture: Optional[str] = Field(
        None,
        description="Cultural identifier that was used for score adjustment",
    )


# --------------------------------------------------------------------------- #
# FastAPI application definition
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Cultural Contextual Sentiment Analyzer",
    version="1.0.0",
    description="Analyzes sentiment of text while optionally adjusting predictions for cultural context.",
)


# --------------------------------------------------------------------------- #
# Model loading (executed once at startup)
# --------------------------------------------------------------------------- #
_sentiment_analyzer: Pipeline | None = None


@app.on_event("startup")
def _load_models() -> None:
    """Load heavy ML objects once when the service starts.

    The function populates the global ``_sentiment_analyzer`` variable with a
    HuggingFace ``pipeline`` configured for sentiment analysis.  Errors during
    loading raise an exception, preventing the app from starting in a broken
    state.
    """
    global _sentiment_analyzer
    try:
        # The DistilBERT SST‑2 model provides a good trade‑off between speed
        # and accuracy for binary sentiment tasks.
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception as exc:
        # Re‑raise so that FastAPI fails fast during startup.
        raise RuntimeError(f"Failed to load sentiment model: {exc}") from exc


# --------------------------------------------------------------------------- #
# Helper: simple cultural score adjustment
# --------------------------------------------------------------------------- #
def _adjust_score_for_culture(raw_score: float, culture: Optional[str]) -> float:
    """Adjust a raw sentiment confidence score based on cultural context.

    The current implementation uses a deterministic, rule‑based approach.
    In a production setting this could be replaced by a scikit‑learn model
    trained on culturally annotated data.

    Parameters
    ----------
    raw_score: float
        Confidence returned by the base transformer model (0‑1).
    culture: Optional[str]
        Cultural identifier supplied by the client.

    Returns
    -------
    float
        Adjusted confidence score, clipped to the 0‑1 interval.
    """
    if not culture:
        return raw_score

    # Example adjustments – these are illustrative only.
    adjustments = {
        "JP": -0.10,  # Japanese texts tend to be more reserved; lower confidence.
        "US": +0.05,  # Slightly higher confidence for US English.
        "IN": +0.02,  # Minor boost for Indian English.
    }

    delta = adjustments.get(culture.upper(), 0.0)
    adjusted = raw_score + delta
    # Ensure the score stays within valid bounds.
    return max(0.0, min(1.0, adjusted))


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", summary="Health check")
def health_check() -> dict:
    """Simple liveness probe used by orchestration platforms."""
    return {"status": "ok"}


@app.post(
    "/sentiment",
    response_model=List[SentimentResult],
    summary="Analyze sentiment with optional cultural adjustment",
)
def analyze_sentiment(request: SentimentRequest) -> List[SentimentResult]:
    """Run sentiment analysis on the supplied text.

    The endpoint forwards the input to a HuggingFace sentiment pipeline,
    then optionally adjusts each confidence score according to the provided
    cultural identifier.

    Parameters
    ----------
    request: SentimentRequest
        JSON payload containing ``text`` and optional ``culture``.

    Returns
    -------
    List[SentimentResult]
        One or more predictions (the pipeline may return multiple if the
        underlying model does so).

    Raises
    ------
    HTTPException
        * 400 – if the input text is empty.
        * 500 – if the underlying model raises an unexpected exception.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="`text` field must be a non‑empty string")

    if _sentiment_analyzer is None:
        # This should never happen because FastAPI blocks start‑up on failure,
        # but we guard against programming errors.
        raise HTTPException(status_code=500, detail="Sentiment model not loaded")

    try:
        raw_predictions = _sentiment_analyzer(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    # The pipeline returns a list of dicts: [{'label': ..., 'score': ...}, ...]
    results: List[SentimentResult] = []
    for pred in raw_predictions:
        adjusted_score = _adjust_score_for_culture(pred["score"], request.culture)
        results.append(
            SentimentResult(
                label=pred["label"],
                score=adjusted_score,
                culture=request.culture,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Development entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Running ``python -m src.main`` starts the service locally.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)