import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline as HFPipeline

# Configure module level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------------------------------------------------------------------------- #
# Constants & Helper Functions
# --------------------------------------------------------------------------- #

# Default paths – can be overridden by environment variables
BASE_DIR = Path(__file__).resolve().parent
IDIOM_MODEL_PATH = Path(
    os.getenv("IDIOM_MODEL_PATH", BASE_DIR / "models" / "idiom_classifier.joblib")
)
SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment"
)

# A very small built‑in idiom dictionary used as a fallback when a trained
# classifier is not available. Keys are language codes, values are lists of
# idiomatic expressions (lower‑cased) that are common in that culture.
BUILTIN_IDIOM_DICT: Dict[str, List[str]] = {
    "en": [
        "break a leg",
        "piece of cake",
        "once in a blue moon",
        "spill the beans",
        "kick the bucket",
    ],
    "es": [
        "estar en la luna",
        "dar la lata",
        "echar una mano",
        "ser pan comido",
    ],
    "fr": [
        "avoir le cafard",
        "casser les pieds",
        "mettre les pieds dans le plat",
    ],
    # Add more languages as needed
}


def _load_idiom_classifier() -> Tuple[Any, bool]:
    """
    Load a scikit‑learn classifier for idiom detection if it exists.

    Returns
    -------
    tuple
        (classifier, is_loaded) where *classifier* is the loaded model or
        ``None`` and *is_loaded* indicates whether the model was successfully
        loaded.
    """
    if IDIOM_MODEL_PATH.is_file():
        try:
            clf = joblib.load(IDIOM_MODEL_PATH)
            logger.info("Loaded idiom classifier from %s", IDIOM_MODEL_PATH)
            return clf, True
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load idiom classifier: %s", exc)
    else:
        logger.info(
            "Idiom classifier not found at %s – falling back to rule‑based detection",
            IDIOM_MODEL_PATH,
        )
    return None, False


def _load_sentiment_pipeline() -> HFPipeline:
    """
    Initialise a Hugging Face sentiment analysis pipeline.

    Returns
    -------
    transformers.Pipeline
        A ready‑to‑use sentiment analysis pipeline.
    """
    try:
        sentiment_pipe = pipeline(
            "sentiment-analysis", model=SENTIMENT_MODEL_NAME, tokenizer=SENTIMENT_MODEL_NAME
        )
        logger.info("Loaded sentiment model '%s'", SENTIMENT_MODEL_NAME)
        return sentiment_pipe
    except Exception as exc:  # pragma: no cover
        logger.exception("Could not load sentiment model: %s", exc)
        raise RuntimeError(f"Failed to load sentiment model '{SENTIMENT_MODEL_NAME}'") from exc


def _normalize_text(text: str) -> str:
    """Lower‑case and strip surrounding whitespace."""
    return text.lower().strip()


def _detect_idioms_rule_based(text: str, language: str) -> List[str]:
    """
    Simple rule‑based idiom detection using the built‑in idiom dictionary.

    Parameters
    ----------
    text : str
        Input text.
    language : str
        ISO‑639‑1 language code (e.g., ``'en'``).

    Returns
    -------
    list[str]
        List of idiomatic expressions found in the text.
    """
    idioms = BUILTIN_IDIOM_DICT.get(language, [])
    found: List[str] = []
    lowered = _normalize_text(text)
    for idiom in idioms:
        # Use word boundaries to avoid partial matches
        pattern = r"\b" + re.escape(idiom) + r"\b"
        if re.search(pattern, lowered):
            found.append(idiom)
    return found


# --------------------------------------------------------------------------- #
# Core Analyzer Class
# --------------------------------------------------------------------------- #

class CulturalSentimentAnalyzer:
    """
    High‑level wrapper that provides cultural idiom detection and sentiment scoring.

    The class lazily loads heavy resources (transformer model, scikit‑learn
    classifier) on first use to keep import time low.
    """

    def __init__(self) -> None:
        self._idiom_clf, self._idiom_clf_loaded = _load_idiom_classifier()
        self._sentiment_pipe: HFPipeline | None = None

    # ------------------------------------------------------------------- #
    # Lazy resource accessors
    # ------------------------------------------------------------------- #

    @property
    def sentiment_pipe(self) -> HFPipeline:
        """Return a ready sentiment pipeline, loading it on first call."""
        if self._sentiment_pipe is None:
            self._sentiment_pipe = _load_sentiment_pipeline()
        return self._sentiment_pipe

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def detect_cultural_idioms(self, text: str, language: str = "en") -> List[str]:
        """
        Detect idiomatic expressions that are culturally specific.

        If a trained classifier is available, it is used; otherwise a simple
        rule‑based lookup is performed.

        Parameters
        ----------
        text : str
            Input text.
        language : str, optional
            ISO‑639‑1 language code. Default is ``'en'``.

        Returns
        -------
        list[str]
            Detected idioms (lower‑cased). Empty list if none are found.
        """
        if self._idiom_clf_loaded:
            # The classifier is expected to accept a list of strings and output
            # a binary label per string. For simplicity we treat the whole text
            # as a single sample.
            try:
                pred = self._idiom_clf.predict([text])[0]
                logger.debug("Idiom classifier prediction: %s", pred)
                return [text] if pred else []
            except Exception as exc:  # pragma: no cover
                logger.error("Idiom classifier failed: %s", exc)
                # Fallback to rule‑based detection
        # Rule‑based fallback
        return _detect_idioms_rule_based(text, language)

    def score_sentiment(self, text: str, language: str = "en") -> Tuple[int, float]:
        """
        Compute a sentiment score for the given text.

        The underlying transformer model returns a label such as ``'1 star'``,
        ``'2 stars'`` … ``'5 stars'``. These are mapped to a signed integer range
        ``[-2, 2]`` where ``-2`` is very negative and ``2`` is very positive.
        The confidence score (probability) is also returned.

        Parameters
        ----------
        text : str
            Input text.
        language : str, optional
            Language code – currently unused because the model is multilingual.

        Returns
        -------
        tuple[int, float]
            ``(sentiment_score, confidence)`` where ``sentiment_score`` is in
            ``{-2, -1, 0, 1, 2}`` and ``confidence`` is a float between 0 and 1.
        """
        result = self.sentiment_pipe(_normalize_text(text))[0]  # type: ignore[index]
        label = result["label"]
        confidence = float(result["score"])

        # Extract star count from label (e.g., "3 stars")
        match = re.search(r"(\d)", label)
        if not match:  # pragma: no cover
            logger.warning("Unexpected sentiment label format: %s", label)
            star = 3  # neutral fallback
        else:
            star = int(match.group(1))

        # Map 1‑5 stars to -2…2
        sentiment_score = star - 3
        logger.debug(
            "Sentiment raw label=%s star=%d mapped_score=%d confidence=%f",
            label,
            star,
            sentiment_score,
            confidence,
        )
        return sentiment_score, confidence

    def analyze(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Run the full pipeline: idiom detection followed by sentiment scoring.

        Parameters
        ----------
        text : str
            Input text.
        language : str, optional
            ISO‑639‑1 language code.

        Returns
        -------
        dict
            ``{'idioms': [...], 'sentiment': {'score': int, 'confidence': float}}``
        """
        idioms = self.detect_cultural_idioms(text, language)
        sentiment_score, confidence = self.score_sentiment(text, language)

        return {
            "idioms": idioms,
            "sentiment": {"score": sentiment_score, "confidence": confidence},
        }


# --------------------------------------------------------------------------- #
# FastAPI Integration
# --------------------------------------------------------------------------- #

router = APIRouter()
_analyzer = CulturalSentimentAnalyzer()


class AnalyzeRequest(BaseModel):
    """Request model for the /analyze endpoint."""

    text: str = Field(..., min_length=1, description="Text to be analysed")
    language: str = Field(
        "en",
        min_length=2,
        max_length=2,
        description="ISO‑639‑1 language code (e.g., 'en', 'es', 'fr')",
    )


class AnalyzeResponse(BaseModel):
    """Response model for the /analyze endpoint."""

    idioms: List[str]
    sentiment: Dict[str, Any]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    FastAPI endpoint that returns cultural idioms and sentiment information.

    Raises
    ------
    HTTPException
        If the underlying analysis fails for any reason.
    """
    try:
        result = _analyzer.analyze(payload.text, payload.language)
        return AnalyzeResponse(**result)
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal analysis error") from exc

# Export a singleton for external modules
analyzer = _analyzer