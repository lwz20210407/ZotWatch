"""Content-addressed candidate vectors with bounded splitting and durable progress."""
import copy
import hashlib
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .vectorizer import EmbeddingError, RemoteVectorizer


class CandidateVectors:
    def __init__(self, encoder, directory, identity, dimension, *, batch_size=8, split_budget=2, workers=4):
        self.encoder = copy.copy(encoder) if isinstance(encoder, RemoteVectorizer) else encoder
        if isinstance(self.encoder, RemoteVectorizer):
            self.encoder.retry_attempts = 1  # This layer owns bounded candidate retries.
        self.directory = Path(directory)
        self.identity = identity
        self.dimension = int(dimension)
        self.text_separator = getattr(encoder, "text_separator", "\n")
        self.signature = getattr(encoder, "signature", identity)
        self.batch_size = batch_size
        self.split_budget = split_budget
        self.stats = {"hits": 0, "written": 0, "encoder_batches": 0, "splits": 0, "cache_errors": 0}
        # Batches go out concurrently: 734 uncached candidates in batches of 8 were ~90
        # serial round trips from the US runner to the embedding API, about six minutes
        # of the weekly run on 2026-10-05.
        self.workers = max(1, int(workers))
        self._lock = threading.Lock()
        self._local = threading.local()

    def _thread_encoder(self):
        """A per-thread copy with its own HTTP session; Sessions are not shared."""
        if not isinstance(self.encoder, RemoteVectorizer):
            return self.encoder  # local models and test doubles are used as given
        encoder = getattr(self._local, "encoder", None)
        if encoder is None:
            encoder = copy.copy(self.encoder)
            session = requests.Session()
            session.headers.update(self.encoder._session.headers)
            encoder._session = session
            self._local.encoder = encoder
        return encoder

    def _count(self, name):
        with self._lock:
            self.stats[name] += 1

    def _key(self, text):
        return hashlib.sha256(json.dumps([1, self.identity, self.dimension, self.text_separator, text], ensure_ascii=False).encode()).hexdigest()

    def _valid(self, vector):
        try:
            vector = np.asarray(vector, dtype=np.float32)
            return vector.shape == (self.dimension,) and np.isfinite(vector).all() and float(np.linalg.norm(vector)) > 0
        except (TypeError, ValueError):
            return False

    def cached(self, text):
        key = self._key(text)
        try:
            payload = json.loads((self.directory / (key + ".json")).read_text("utf-8"))
            if payload.get("key") == key and self._valid(payload.get("vector")):
                return np.asarray(payload["vector"], dtype=np.float32)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def store(self, text, vector):
        if not self._valid(vector):
            raise EmbeddingError("Invalid candidate vector shape or values")
        key = self._key(text)
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                json.dump({"key": key, "vector": np.asarray(vector, dtype=np.float32).tolist()}, f, allow_nan=False)
            os.replace(temporary, self.directory / (key + ".json"))
            self._count("written")
        except OSError:
            self._count("cache_errors")
        finally:
            if temporary and temporary.exists():
                temporary.unlink()  # Only the unique temporary file created by this call.

    def _batch(self, texts, remaining, result):
        self._count("encoder_batches")
        try:
            vectors = self._thread_encoder().encode(texts)
        except requests.Timeout:
            if len(texts) <= 1:
                raise
            with self._lock:  # the split budget is shared by all workers
                if remaining[0] <= 0:
                    raise
                remaining[0] -= 1
                self.stats["splits"] += 1
            mid = len(texts) // 2
            self._batch(texts[:mid], remaining, result)
            self._batch(texts[mid:], remaining, result)
            return
        if len(vectors) != len(texts) or any(not self._valid(v) for v in vectors):
            raise EmbeddingError("Invalid candidate embedding response")
        for text, vector in zip(texts, vectors):
            result[text] = np.asarray(vector, dtype=np.float32)
            self.store(text, vector)

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        result, missing = {}, []
        for text in dict.fromkeys(texts):
            vector = self.cached(text)
            if vector is None:
                missing.append(text)
            else:
                result[text] = vector
                self.stats["hits"] += 1
        remaining = [self.split_budget]
        chunks = [missing[start:start + self.batch_size] for start in range(0, len(missing), self.batch_size)]
        workers = min(self.workers, len(chunks))
        if workers <= 1:
            for chunk in chunks:
                self._batch(chunk, remaining, result)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                # result() in submission order re-raises the first failure, as the serial
                # loop did; batches not yet started are then cancelled.
                for future in [pool.submit(self._batch, chunk, remaining, result) for chunk in chunks]:
                    future.result()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        return np.stack([result[t] for t in texts])

    def encode_single(self, text):
        return self.encode([text])[0]

    def load(self):
        return self.encoder.load()
