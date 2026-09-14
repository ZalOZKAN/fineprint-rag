"""Cross-encoder reranking of retrieval candidates.

Dense and sparse retrieval are bi-encoders: they embed the query and each
passage separately and compare the two vectors. That is fast enough to run over
a whole corpus, but it cannot see how the query and a passage relate, only that
they land near each other. A cross-encoder reads the pair together, one forward
pass per (query, passage), and is far better at picking the passage that answers
the question from one that merely shares its vocabulary.

The cross-encoder is too slow to score every passage, so it never touches the
corpus. It reruns only the handful of candidates retrieval already surfaced
(config.RERANK_CANDIDATES) and reorders them; the top config.CONTEXT_CHUNKS of
that order become the model context.

The model is a small ONNX cross-encoder run through onnxruntime, the same
runtime Foundry Local uses. No torch, no transformers. It downloads once from
the Hugging Face hub and is then cached on disk like the Foundry models.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

import config
from rag.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

_MODEL_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"

# ms-marco-MiniLM was trained with 512 positions, but a query plus one clause is
# short; capping lower keeps a batch of 20 pairs fast without losing the passage.
_MAX_TOKENS = 384


class Reranker:
    """An ONNX cross-encoder that reorders retrieval candidates by relevance."""

    def __init__(self, repo: str | None = None, max_tokens: int = _MAX_TOKENS) -> None:
        self.repo = repo or config.RERANK_MODEL
        self.max_tokens = max_tokens
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(self.repo, _MODEL_FILE)
        tokenizer_path = hf_hub_download(self.repo, _TOKENIZER_FILE)

        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=self.max_tokens)
        self._tokenizer = tokenizer

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        logger.info("Reranker ready: %s", self.repo)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logit for each (query, passage) pair. Higher is better."""
        if not passages:
            return []
        self._ensure_loaded()

        encodings = self._tokenizer.encode_batch(
            [(query, passage) for passage in passages]
        )
        width = max(len(encoding.ids) for encoding in encodings)
        rows = len(encodings)
        input_ids = np.zeros((rows, width), dtype=np.int64)
        attention = np.zeros((rows, width), dtype=np.int64)
        type_ids = np.zeros((rows, width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            span = len(encoding.ids)
            input_ids[row, :span] = encoding.ids
            attention[row, :span] = encoding.attention_mask
            type_ids[row, :span] = encoding.type_ids

        feed = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = type_ids

        logits = self._session.run(None, feed)[0]
        return [float(value) for value in np.asarray(logits).reshape(-1)]

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        limit: int = config.CONTEXT_CHUNKS,
    ) -> list[ScoredChunk]:
        """Return the `limit` candidates the cross-encoder rates highest."""
        if not candidates:
            return []
        scores = self.score(query, [scored.chunk.content for scored in candidates])
        order = np.argsort(scores)[::-1][:limit]
        return [candidates[int(index)] for index in order]

    def unload(self) -> None:
        """Release the model. Safe to call when nothing was ever loaded."""
        self._session = None
        self._tokenizer = None
        self._input_names = set()
