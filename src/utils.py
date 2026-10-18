import json
import logging
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data packages are available.
# These calls are idempotent; they will download the resources only once.
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

logger = logging.getLogger(__name__)

__all__ = [
    "IdiomLookup",
    "preprocess_text",
    "tokenize",
    "load_idiom_file",
]


class IdiomLookup:
    """
    Utility class for loading and querying idioms with their cultural meanings.

    The idiom data file is expected to be a JSON mapping where each key is an
    idiom (string) and the value is a dictionary containing at least a
    ``"meaning"`` field. Example::

        {
            "break the ice": {"meaning": "to initiate conversation"},
            "kick the bucket": {"meaning": "to die"}
        }

    Parameters
    ----------
    idiom_file : Union[str, Path]
        Path to the JSON file containing the idiom dictionary.
    """

    def __init__(self, idiom_file: Union[str, Path]) -> None:
        self.idiom_file = Path(idiom_file)
        self._idioms: Dict[str, Dict[str, str]] = {}
        self._load_idioms()

    def _load_idioms(self) -> None:
        """Load idioms from the JSON file into memory."""
        if not self.idiom_file.is_file():
            logger.error("Idiom file not found: %s", self.idiom_file)
            raise FileNotFoundError(f"Idiom file not found: {self.idiom_file}")

        try:
            with self.idiom_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.exception("Failed to parse idiom JSON file.")
            raise ValueError(f"Invalid JSON in idiom file: {self.idiom_file}") from exc

        if not isinstance(data, dict):
            logger.error("Idiom JSON must be a mapping of idiom -> details.")
            raise ValueError("Idiom JSON must be a mapping of idiom -> details.")

        # Normalise keys to lower case for case‑insensitive lookup.
        self._idioms = {k.lower(): v for k, v in data.items()}
        logger.info("Loaded %d idioms from %s", len(self._idioms), self.idiom_file)

    def get_meaning(self, idiom: str) -> Optional[str]:
        """
        Retrieve the cultural meaning of an idiom.

        Parameters
        ----------
        idiom : str
            The idiom to look up. Lookup is case‑insensitive.

        Returns
        -------
        Optional[str]
            The meaning if the idiom exists, otherwise ``None``.
        """
        key = idiom.lower()
        entry = self._idioms.get(key)
        if entry is None:
            logger.debug("Idiom not found: %s", idiom)
            return None
        meaning = entry.get("meaning")
        logger.debug("Found meaning for idiom '%s': %s", idiom, meaning)
        return meaning

    def list_idioms(self) -> List[str]:
        """
        Return a sorted list of all idioms known to the lookup.

        Returns
        -------
        List[str]
            Alphabetically sorted idioms.
        """
        return sorted(self._idioms.keys())

    def add_idiom(self, idiom: str, meaning: str) -> None:
        """
        Add a new idiom to the in‑memory dictionary. Does not persist to disk.

        Parameters
        ----------
        idiom : str
            Idiom text.
        meaning : str
            Cultural meaning of the idiom.
        """
        key = idiom.lower()
        self._idioms[key] = {"meaning": meaning}
        logger.info("Added idiom '%s' with meaning '%s' to lookup.", idiom, meaning)


def load_idiom_file(path: Union[str, Path]) -> IdiomLookup:
    """
    Convenience factory to create an :class:`IdiomLookup` from a path.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the idiom JSON file.

    Returns
    -------
    IdiomLookup
        An instantiated lookup object.
    """
    return IdiomLookup(path)


def _normalize_text(text: str) -> str:
    """
    Apply Unicode NFKC normalization and strip control characters.

    This helps to make the text robust against different encodings and
    invisible characters.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    normalized = unicodedata.normalize("NFKC", text)
    # Remove any remaining control characters (e.g., zero‑width spaces)
    cleaned = re.sub(r"[\u0000-\u001F\u007F]", "", normalized)
    return cleaned


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenisation after normalisation.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]
        List of tokens.
    """
    normalized = _normalize_text(text)
    return normalized.split()


def preprocess_text(
    text: str,
    *,
    lower: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Preprocess raw text for downstream NLP models.

    The pipeline consists of:
    1. Unicode normalisation.
    2. Optional lower‑casing.
    3. Optional punctuation removal.
    4. Tokenisation on whitespace.
    5. Optional stop‑word removal.
    6. Optional lemmatisation.

    Parameters
    ----------
    text : str
        Raw input text.
    lower : bool, default True
        Convert text to lower case.
    remove_punct : bool, default True
        Strip punctuation characters defined in :data:`string.punctuation`.
    remove_stopwords : bool, default True
        Remove English stop words (NLTK's corpus). ``custom_stopwords`` can
        extend or replace the default set.
    lemmatize : bool, default True
        Apply WordNet lemmatization.
    custom_stopwords : Optional[Set[str]]
        If provided, this set will be used instead of NLTK's default stop words.

    Returns
    -------
    List[str]
        List of cleaned tokens ready for vectorisation or model input.
    """
    # Step 1: Normalise Unicode.
    text = _normalize_text(text)

    # Step 2: Lower‑case.
    if lower:
        text = text.lower()

    # Step 3: Remove punctuation.
    if remove_punct:
        # Create translation table that maps punctuation to None.
        punct_table = str.maketrans("", "", string.punctuation)
        text = text.translate(punct_table)

    # Step 4: Tokenise.
    tokens = text.split()

    # Step 5: Stop‑word removal.
    if remove_stopwords:
        stop_words: Set[str] = (
            custom_stopwords if custom_stopwords is not None else set(stopwords.words("english"))
        )
        tokens = [tok for tok in tokens if tok not in stop_words]

    # Step 6: Lemmatisation.
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    logger.debug("Preprocessed text: %s -> %s", text, tokens)
    return tokens