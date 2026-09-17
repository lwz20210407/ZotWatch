from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled via runtime requirement
    SentenceTransformer = None  # type: ignore

DEFAULT_MODEL = "sentence-transformers/allenai-specter"
DEFAULT_SEPARATOR = "[SEP]"


class TextVectorizer:
    """Encode title/abstract text into unit-norm vectors.

    The model is configurable because the choice is domain specific: a general
    web-text model truncates most abstracts and carries no notion of scientific
    similarity. See config/embedding.yaml.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        text_separator: str = DEFAULT_SEPARATOR,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.text_separator = text_separator
        self.batch_size = batch_size
        self._model = None

    @classmethod
    def from_settings(cls, settings) -> "TextVectorizer":
        config = getattr(settings, "embedding", None)
        if config is None:
            return cls()
        return cls(
            config.model_name,
            text_separator=config.text_separator,
            batch_size=config.batch_size,
        )

    def load(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or adjust requirements."
            )
        logger.info("Loading embedding model %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        limit = getattr(self._model, "max_seq_length", None)
        if limit:
            logger.info("Model input limit: %s word pieces", limit)

    @property
    def model(self):  # type: ignore
        self.load()
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        self.load()
        batch = list(texts)
        if not batch:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            batch, show_progress_bar=False, batch_size=self.batch_size
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


__all__ = ["TextVectorizer", "DEFAULT_MODEL", "DEFAULT_SEPARATOR"]
