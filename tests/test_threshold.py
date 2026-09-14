"""Tests for automatic per-library relevance-gate calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rag import db
from rag.db import Chunk
from rag.threshold import calibrate_and_store, estimate_rerank_threshold


def _chunk(chunk_id: int, heading: str = "Heading", content: str = "some content") -> Chunk:
    return Chunk(chunk_id, 1, chunk_id, heading, content, "doc.md", "Doc")


class FixedReranker:
    """Scores a self-pair query `pos` and every off-topic probe `neg`.

    A query is treated as an off-topic probe when it is not the exact heading
    or leading words of any chunk passed in - which is how every one of
    `_OFF_TOPIC_PROBES` behaves against these tests' plain chunks, and how a
    fixed positive/negative split is enough to check the arithmetic (margin,
    the worst-case-not-average rule, degenerate cases) without depending on
    any particular scoring behaviour.
    """

    def __init__(self, pos: float, neg: float, worst_offset: float = 0.0) -> None:
        self.pos = pos
        self.neg = neg
        self.worst_offset = worst_offset
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages):
        self.calls.append((query, list(passages)))
        if len(passages) == 1:
            return [self.pos]
        # One off-topic probe call scores a whole sample of passages; bump the
        # first passage's score to make "take the max of the batch" checkable.
        return [self.neg + self.worst_offset] + [self.neg] * (len(passages) - 1)


def test_returns_none_with_no_reranker():
    assert estimate_rerank_threshold(None, [_chunk(1), _chunk(2)]) is None


def test_returns_none_with_fewer_than_two_chunks():
    assert estimate_rerank_threshold(FixedReranker(5, -5), [_chunk(1)]) is None
    assert estimate_rerank_threshold(FixedReranker(5, -5), []) is None


def test_threshold_sits_at_the_configured_margin_above_the_worst_off_topic_score():
    reranker = FixedReranker(pos=5.0, neg=-5.0)
    chunks = [_chunk(i) for i in range(5)]
    threshold = estimate_rerank_threshold(
        reranker, chunks, margin_fraction=0.5, rng_seed=1
    )
    # worst_off_topic + 0.5 * (mean_pos - worst_off_topic) = -5 + 0.5*10 = 0.0
    assert threshold == pytest.approx(0.0)


def test_uses_the_worst_off_topic_score_not_the_average():
    # One probe scores far higher against this corpus than the rest - the
    # threshold must sit above *that*, not the average of all the probes.
    reranker = FixedReranker(pos=5.0, neg=-5.0, worst_offset=8.0)
    chunks = [_chunk(i) for i in range(5)]
    threshold = estimate_rerank_threshold(reranker, chunks, margin_fraction=0.5, rng_seed=1)
    # worst_off_topic is now -5 + 8 = 3.0, not the average (~-3.4).
    assert threshold == pytest.approx(3.0 + 0.5 * (5.0 - 3.0))


def test_lower_margin_fraction_gives_a_more_permissive_threshold():
    reranker = FixedReranker(pos=5.0, neg=-5.0)
    chunks = [_chunk(i) for i in range(5)]
    lenient = estimate_rerank_threshold(reranker, chunks, margin_fraction=0.1, rng_seed=1)
    strict = estimate_rerank_threshold(reranker, chunks, margin_fraction=0.9, rng_seed=1)
    assert lenient < strict


def test_degenerate_scores_fall_back_to_none():
    # The "clearly relevant" proxy never actually scores above the worst
    # off-topic probe - trust the config default over noise.
    reranker = FixedReranker(pos=-2.0, neg=-2.0)
    chunks = [_chunk(i) for i in range(4)]
    assert estimate_rerank_threshold(reranker, chunks) is None


def test_empty_heading_falls_back_to_the_first_words_of_the_content():
    reranker = FixedReranker(pos=5.0, neg=-5.0)
    chunks = [
        _chunk(1, heading="", content="cancel within fourteen days of signing"),
        _chunk(2, heading="", content="a completely different clause entirely"),
    ]
    estimate_rerank_threshold(reranker, chunks, rng_seed=0)
    self_pair_queries = [query for query, passages in reranker.calls if len(passages) == 1]
    assert self_pair_queries == [
        "cancel within fourteen days of signing",
        "a completely different clause entirely",
    ]


def test_sampling_caps_the_number_of_self_pair_calls():
    reranker = FixedReranker(pos=5.0, neg=-5.0)
    chunks = [_chunk(i) for i in range(100)]
    estimate_rerank_threshold(reranker, chunks, sample_size=10, rng_seed=0)
    self_pair_calls = [call for call in reranker.calls if len(call[1]) == 1]
    assert len(self_pair_calls) == 10


def _add_document_with_chunks(conn, filename: str, headings_and_content) -> None:
    document_id = db.replace_document(conn, filename, filename, "hash-" + filename)
    vectors = np.zeros((len(headings_and_content), 4), dtype=np.float32)
    db.insert_chunks(conn, document_id, headings_and_content, vectors)


def test_calibrate_and_store_persists_the_estimate(conn):
    _add_document_with_chunks(
        conn,
        "a.md",
        [("Heading A", "content a"), ("Heading B", "content b"),
         ("Heading C", "content c")],
    )
    reranker = FixedReranker(pos=5.0, neg=-5.0)
    threshold = calibrate_and_store(conn, reranker)
    assert threshold == pytest.approx(0.0)
    stored = db.get_meta(conn, db.META_RERANK_THRESHOLD)
    assert float(stored) == pytest.approx(0.0)


def test_calibrate_and_store_clears_a_stale_value_when_it_cannot_calibrate(conn):
    db.set_meta(conn, db.META_RERANK_THRESHOLD, "1.23")
    # Only one chunk in the whole library - too little to calibrate from.
    _add_document_with_chunks(conn, "a.md", [("Heading", "only chunk")])
    result = calibrate_and_store(conn, FixedReranker(pos=5.0, neg=-5.0))
    assert result is None
    assert db.get_meta(conn, db.META_RERANK_THRESHOLD) is None
