"""Text embedding, either through an OpenAI-compatible HTTP API or a local model.

The remote path exists so this project can reuse an embedding service that is
already configured elsewhere (ZotPilot points at SiliconFlow's
Qwen/Qwen3-Embedding-8B). Reusing the *encoder* is what actually matters: the
library vectors held by other tools cannot be reused directly, because the weekly
job has to embed a thousand brand-new papers that exist in no local index, and
those must be encoded by the same model as the library for the comparison to mean
anything.

Running remotely also removes sentence-transformers and torch from CI entirely.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import numpy as np
import requests

from .http_utils import request_with_retry

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional, only for provider "local"
    SentenceTransformer = None  # type: ignore

DEFAULT_SEPARATOR = "\n"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


class RemoteVectorizer:
    """OpenAI-compatible /embeddings client."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        *,
        dimensions: Optional[int] = None,
        text_separator: str = DEFAULT_SEPARATOR,
        batch_size: int = 32,
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.text_separator = text_separator
        self.batch_size = batch_size
        self.dimensions = dimensions
        # Describes the encoder that actually runs, so a profile can never claim a
        # model it was not built with. build_vectorizer() overwrites this with
        # config.cache_signature() on the remote path to keep existing caches valid.
        self.signature = f"openai-compatible:{model_name}:{dimensions or 'native'}"
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        self._dimension: Optional[int] = dimensions

    def load(self) -> None:  # parity with the local vectorizer
        return None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.encode(["dimension probe"]).shape[1])
        return self._dimension

    def _post(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        response = request_with_retry(
            self._session,
            "POST",
            f"{self.base_url}/embeddings",
            logger=logger,
            context=f"embeddings({len(texts)} texts)",
            json=payload,
            timeout=self.timeout,
        )
        body = response.json()
        rows = sorted(body["data"], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = [text if text and text.strip() else " " for text in texts]
        if not batch:
            return np.zeros((0, self._dimension or 0), dtype=np.float32)
        vectors: List[List[float]] = []
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start : start + self.batch_size]
            embeddings = self._post(chunk)
            if len(embeddings) != len(chunk):
                raise EmbeddingError(
                    f"Provider returned {len(embeddings)} embeddings for {len(chunk)} inputs"
                )
            vectors.extend(embeddings)
            if len(batch) > self.batch_size:
                logger.info("Embedded %d/%d texts", min(start + self.batch_size, len(batch)), len(batch))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class LocalVectorizer:
    """sentence-transformers backend, used when provider is "local"."""

    def __init__(
        self,
        model_name: str,
        *,
        text_separator: str = DEFAULT_SEPARATOR,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.text_separator = text_separator
        self.batch_size = batch_size
        self._model = None
        # Deliberately not probing self.dimension here -- that would download and
        # load the model just to name it. The model name already pins the width.
        self.signature = f"local:{model_name}"

    def load(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise EmbeddingError(
                "provider is 'local' but sentence-transformers is not installed. "
                "Install it, or set provider to 'openai-compatible' in config/embedding.yaml."
            )
        logger.info("Loading local embedding model %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

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
        return _normalize(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_vectorizer(config) -> RemoteVectorizer | LocalVectorizer:
    """Construct the vectorizer described by config/embedding.yaml.

    Falls back to the local model when the remote provider has no API key, so a
    missing secret degrades instead of failing at the first encode call.
    """
    provider = (getattr(config, "provider", "local") or "local").lower()
    if provider in {"openai-compatible", "openai", "remote"}:
        api_key = os.getenv(config.api_key_env, "")
        if api_key:
            logger.info(
                "Embedding via %s using %s (%s dims)",
                config.base_url, config.model_name, config.dimensions or "native",
            )
            remote = RemoteVectorizer(
                config.model_name,
                config.base_url,
                api_key,
                dimensions=config.dimensions,
                text_separator=config.text_separator,
                batch_size=config.batch_size,
                timeout=config.timeout_seconds,
            )
            # Keep the config-derived spelling on the remote path so the 4399
            # vectors already cached under it stay valid. Only the fallback below
            # gets a signature of its own, which is the case that was lying.
            remote.signature = config.cache_signature()
            return remote
        if not config.local_fallback_model:
            raise EmbeddingError(
                f"Environment variable {config.api_key_env} is not set and no "
                f"local_fallback_model is configured in config/embedding.yaml."
            )
        logger.warning(
            "%s is not set; falling back to the local model %s",
            config.api_key_env, config.local_fallback_model,
        )
        return LocalVectorizer(
            config.local_fallback_model,
            text_separator=config.text_separator,
            batch_size=config.batch_size,
        )
    return LocalVectorizer(
        config.model_name,
        text_separator=config.text_separator,
        batch_size=config.batch_size,
    )


class TextVectorizer:
    """Backwards-compatible entry point."""

    def __new__(cls, *args, **kwargs):  # pragma: no cover - thin shim
        raise TypeError("Use TextVectorizer.from_settings(settings) or build_vectorizer(config)")

    @staticmethod
    def from_settings(settings) -> RemoteVectorizer | LocalVectorizer:
        return build_vectorizer(settings.embedding)


__all__ = [
    "RemoteVectorizer",
    "LocalVectorizer",
    "TextVectorizer",
    "build_vectorizer",
    "EmbeddingError",
    "DEFAULT_SEPARATOR",
]
