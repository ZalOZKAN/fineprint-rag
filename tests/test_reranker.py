"""Tests for the cross-encoder reranker.

The ONNX model and tokenizer are replaced with fakes: what is under test is the
batching, the input assembly and the reordering, not the model weights.
"""

from __future__ import annotations

import numpy as np

from rag.db import Chunk
from rag.reranker import Reranker
from rag.retrieval import ScoredChunk


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class FakeTokenizer:
    """Turns each (query, passage) pair into ids as long as the passage text."""

    def enable_truncation(self, max_length: int) -> None:  # noqa: D401
        self.max_length = max_length

    def encode_batch(self, pairs):
        return [_Encoding(list(range(1, len(passage) + 2))) for _, passage in pairs]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    """Scores each row by how often the letter 'z' appears in its ids length.

    The test data is built so the intended winner is the longest passage, which
    makes the expected order obvious without depending on real model behaviour.
    """

    def __init__(self, scores_by_length: bool = True) -> None:
        self.scores_by_length = scores_by_length
        self.last_feed: dict | None = None

    def get_inputs(self):
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, _outputs, feed):
        self.last_feed = feed
        lengths = feed["input_ids"].astype(bool).sum(axis=1)
        return [lengths.astype(np.float32).reshape(-1, 1)]


def _reranker_with(session: FakeSession) -> Reranker:
    reranker = Reranker()
    reranker._session = session
    reranker._tokenizer = FakeTokenizer()
    reranker._input_names = {"input_ids", "attention_mask"}
    reranker._ensure_loaded = lambda: None  # type: ignore[method-assign]
    return reranker


def _chunk(text: str, chunk_id: int) -> ScoredChunk:
    chunk = Chunk(chunk_id, 1, chunk_id, "Heading", text, "doc.md", "Doc")
    return ScoredChunk(chunk, 0.1, 0.0, 0.01, chunk_id, None)


def test_score_returns_one_value_per_passage():
    reranker = _reranker_with(FakeSession())
    scores = reranker.score("q", ["a", "bb", "ccc"])
    assert len(scores) == 3
    assert scores[2] > scores[0]  # longer passage -> more non-zero ids


def test_score_of_nothing_is_empty():
    assert _reranker_with(FakeSession()).score("q", []) == []


def test_token_type_ids_are_only_sent_when_the_model_wants_them():
    session = FakeSession()
    reranker = _reranker_with(session)
    reranker._input_names = {"input_ids", "attention_mask"}
    reranker.score("q", ["passage"])
    assert "token_type_ids" not in session.last_feed


def test_rerank_reorders_candidates_by_score_and_truncates():
    reranker = _reranker_with(FakeSession())
    candidates = [_chunk("short", 1), _chunk("a much longer passage here", 2),
                  _chunk("medium length", 3)]
    top = reranker.rerank("q", candidates, limit=2)
    assert [scored.chunk.id for scored in top] == [2, 3]


def test_rerank_with_no_candidates_is_empty():
    assert _reranker_with(FakeSession()).rerank("q", [], limit=3) == []


def test_unload_clears_state():
    reranker = _reranker_with(FakeSession())
    reranker.unload()
    assert reranker._session is None
    assert reranker._tokenizer is None
