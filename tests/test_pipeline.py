"""tests/test_pipeline.py

Unit tests for the idiom detection and sentiment analysis pipeline of the
*Cultural Contextual Sentiment Analyzer* project.

The tests cover:
* Correct identification of idiomatic expressions.
* Proper handling of non‑idiomatic text.
* Sentiment scoring with and without cultural context.
* Edge‑case handling such as empty strings and unsupported cultures.
"""

from __future__ import annotations

import pytest

# The project is expected to expose the core pipeline functions at the top level.
# Adjust the import path if the implementation resides in a sub‑package.
try:
    from pipeline import detect_idioms, analyze_sentiment
except ImportError as exc:  # pragma: no cover
    # Provide a clear error message for developers running the test suite.
    raise ImportError(
        "Unable to import `detect_idioms` and `analyze_sentiment`. "
        "Make sure the pipeline module is available on the PYTHONPATH."
    ) from exc


@pytest.mark.parametrize(
    ("text", "expected_idioms"),
    [
        ("Let's break the ice before the meeting.", {"break the ice"}),
        ("He kicked the bucket yesterday.", {"kick the bucket"}),
        ("She spilled the beans about the surprise.", {"spill the beans"}),
        ("No idioms here, just plain text.", set()),
        ("They decided to bite the bullet and start the project.", {"bite the bullet"}),
    ],
)
def test_detect_idioms_basic(text: str, expected_idioms: set[str]) -> None:
    """Ensure that known idioms are detected and non‑idioms are ignored."""
    result = detect_idioms(text)
    assert isinstance(result, set), "detect_idioms should return a set of strings"
    assert result == expected_idioms, f"Unexpected idiom detection for: {text!r}"


def test_detect_idioms_case_insensitivity() -> None:
    """Idiom detection must be case‑insensitive."""
    text = "She really KICKED THE BUCKET after the marathon."
    result = detect_idioms(text)
    assert "kick the bucket" in result, "Idiom detection should ignore case differences"


def test_detect_idioms_multiple_occurrences() -> None:
    """Multiple occurrences of the same idiom should be deduplicated."""
    text = "He decided to break the ice. Later, they tried to break the ice again."
    result = detect_idioms(text)
    assert result == {"break the ice"}, "Duplicate idioms must be collapsed into a set"


def test_detect_idioms_empty_string() -> None:
    """An empty input should return an empty set without error."""
    assert detect_idioms("") == set()


@pytest.mark.parametrize(
    ("text", "culture", "expected_sentiment"),
    [
        # Default (culture‑agnostic) positive sentiment
        ("I am thrilled with the results!", None, "positive"),
        # Default negative sentiment
        ("This is a terrible mistake.", None, "negative"),
        # Neutral sentiment when no polarity words are present
        ("The sky is blue today.", None, "neutral"),
        # Cultural adjustment: in some cultures, "I am thrilled" may be seen as modest
        ("I am thrilled with the results!", "japanese", "neutral"),
        # Cultural adjustment: sarcasm detection for a specific culture
        ("Great, another bug!", "american", "negative"),
    ],
)
def test_analyze_sentiment_basic(
    text: str, culture: str | None, expected_sentiment: str
) -> None:
    """Validate sentiment polarity across cultures and edge cases."""
    sentiment = analyze_sentiment(text, culture=culture)
    assert isinstance(sentiment, str), "Sentiment should be returned as a string"
    assert sentiment == expected_sentiment, (
        f"Sentiment mismatch for text={text!r}, culture={culture!r}"
    )


def test_analyze_sentiment_empty_string() -> None:
    """Empty input should return a neutral sentiment rather than error."""
    assert analyze_sentiment("") == "neutral"


def test_analyze_sentiment_invalid_culture() -> None:
    """Providing an unsupported culture should raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported culture"):
        analyze_sentiment("I love this!", culture="martian")


def test_analyze_sentiment_idioms_influence() -> None:
    """
    Idiomatic expressions can affect sentiment scoring.
    For example, "kick the bucket" conveys death → negative sentiment.
    """
    text = "Unfortunately, he kicked the bucket during the ceremony."
    sentiment = analyze_sentiment(text)
    assert sentiment == "negative", "Idiom-related sentiment not correctly inferred"


@pytest.mark.integration
def test_full_pipeline_integration() -> None:
    """
    Integration test that runs idiom detection and sentiment analysis together.
    The pipeline should return both the set of idioms and the final sentiment.
    """
    text = "She spilled the beans and everyone was thrilled."
    idioms = detect_idioms(text)
    sentiment = analyze_sentiment(text)

    assert idioms == {"spill the beans"}, "Integration idiom detection failed"
    assert sentiment == "positive", "Integration sentiment analysis failed"
