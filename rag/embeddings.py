"""Local embedding generation through Foundry Local.

Vectors are L2 normalized as they are produced, so that cosine similarity later
reduces to a dot product. That turns the whole dense search into one matrix
multiplication and removes two square roots per comparison at query time.

The model is loaded lazily and kept for the lifetime of the process, because
loading costs several seconds and a Streamlit session asks many questions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

import config

logger = logging.getLogger(__name__)

# Requests are batched so a large document does not arrive as one huge call.
BATCH_SIZE = 32


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all zero rows untouched."""
    if vectors.size == 0:
        return vectors.astype(np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    """Wraps the Foundry Local embedding model behind a NumPy interface."""

    def __init__(self, model_alias: str | None = None) -> None:
        self.model_alias = model_alias or config.EMBEDDING_MODEL
        self._model = None
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return

        import foundry_local_sdk as fl

        try:
            fl.FoundryLocalManager.initialize(fl.Configuration(app_name=config.APP_NAME))
        except Exception:  # noqa: BLE001 - already initialized in this process
            logger.debug("Foundry Local manager was already initialized")

        manager = fl.FoundryLocalManager.instance
        model = manager.catalog.get_model(self.model_alias)
        if model is None:
            raise RuntimeError(f"Embedding model not in catalog: {self.model_alias}")
        if not model.is_cached:
            logger.info("Downloading %s", self.model_alias)
            model.download()
        model.load()
        self._model = model
        self._client = model.get_embedding_client()
        logger.info("Embedding model ready: %s", self.model_alias)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of texts, returning one unit length row per input."""
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        rows: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start : start + BATCH_SIZE])
            response = self._client.generate_embeddings(batch)
            rows.extend(item.embedding for item in response.data)

        if len(rows) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: asked for {len(texts)}, got {len(rows)}"
            )
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question, returning a one dimensional unit vector."""
        self._ensure_loaded()
        response = self._client.generate_embedding(text)
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        return l2_normalize(vector.reshape(1, -1))[0]

    def unload(self) -> None:
        """Release the model. Safe to call when nothing was ever loaded."""
        if self._model is not None:
            self._model.unload()
            self._model = None
            self._client = None
