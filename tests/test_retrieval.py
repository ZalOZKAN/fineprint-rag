"""Tests for dense, sparse and fused retrieval."""

from __future__ import annotations

import numpy as np
import pytest

import config
from rag import ingest, retrieval
from rag.retrieval import Index, Mode, reciprocal_rank_fusion


@pytest.fixture()
def indexed(conn, embedder, corpus):
    """A database with the sample corpus ingested and an index over it."""
    ingest.ingest_directory(conn, embedder, corpus)
    return conn, Index(conn), embedder


# --- RRF -----------------------------------------------------------------


def test_rrf_rewards_appearing_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    # 1 and 3 are in both lists, so they must outrank 2 and 4.
    assert fused[1] > fused[2]
    assert fused[3] > fused[4]


def test_rrf_prefers_better_ranks():
    fused = reciprocal_rank_fusion([[1, 2]], k=60)
    assert fused[1] > fused[2]


def test_rrf_uses_expected_formula():
    fused = reciprocal_rank_fusion([[7]], k=60)
    assert fused[7] == pytest.approx(1.0 / 61.0)


def test_rrf_on_empty_input():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}


# --- index ---------------------------------------------------------------


def test_index_reports_size(indexed):
    _, index, _ = indexed
    assert index.size > 0
    assert index.matrix.shape[0] == index.size


def test_index_on_empty_database(conn):
    index = Index(conn)
    assert index.size == 0
    assert index.search(np.zeros(4, dtype=np.float32), 3) == []


def test_index_search_returns_scores_in_descending_order(indexed):
    _, index, embedder = indexed
    hits = index.search(embedder.embed_query("cancellation fee"), limit=5)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_respects_limit(indexed):
    _, index, embedder = indexed
    assert len(index.search(embedder.embed_query("excess"), limit=2)) == 2


def test_index_refresh_picks_up_new_chunks(conn, embedder, corpus):
    index = Index(conn)
    assert index.size == 0
    ingest.ingest_directory(conn, embedder, corpus)
    index.refresh()
    assert index.size > 0


def test_score_of_unknown_chunk_is_zero(indexed):
    _, index, embedder = indexed
    assert index.score_of(999999, embedder.embed_query("anything")) == 0.0


# --- retrieve ------------------------------------------------------------


def test_hybrid_retrieval_finds_the_right_section(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "sublet the property")
    assert result.results
    assert any("sublet" in scored.chunk.content.lower() for scored in result.results)


def test_retrieval_respects_the_context_limit(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "excess", limit=2)
    assert len(result.results) <= 2


def test_empty_question_returns_nothing(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "   ")
    assert result.results == []
    assert not result.is_relevant


def test_retrieval_on_empty_corpus(conn, embedder):
    result = retrieval.retrieve(conn, Index(conn), embedder, "anything")
    assert result.results == []
    assert not result.is_relevant


def test_dense_mode_ignores_keyword_search(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "excess", mode=Mode.DENSE)
    assert all(scored.sparse_rank is None for scored in result.results)


def test_sparse_mode_ignores_vector_search(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "sublet", mode=Mode.SPARSE)
    assert result.results
    assert all(scored.dense_rank is None for scored in result.results)


def test_sparse_only_hits_still_receive_a_dense_score(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "sublet", mode=Mode.SPARSE)
    # Without this the relevance gate would reject every exact term match.
    assert any(scored.dense_score != 0.0 for scored in result.results)


class _WordCountReranker:
    """Scores a passage by how many query words it contains. No model."""

    def score(self, query, passages):
        words = set(query.lower().split())
        return [
            float(sum(word in passage.lower() for word in words))
            for passage in passages
        ]


def test_reranker_reorders_results_and_sets_its_top_score(indexed):
    conn, index, embedder = indexed
    plain = retrieval.retrieve(conn, index, embedder, "sublet the property")
    reranked = retrieval.retrieve(
        conn, index, embedder, "sublet the property",
        reranker=_WordCountReranker(),
    )
    assert reranked.best_rerank_score is not None
    assert reranked.best_dense_score == pytest.approx(plain.best_dense_score)
    top = reranked.results[0]
    assert top.rerank_score == reranked.best_rerank_score


def test_gate_uses_the_reranker_score_when_one_is_present(indexed, monkeypatch):
    conn, index, embedder = indexed
    monkeypatch.setattr(config, "RERANK_RELEVANCE_THRESHOLD", 99.0)
    result = retrieval.retrieve(
        conn, index, embedder, "sublet", reranker=_WordCountReranker(),
    )
    assert result.results
    # An impossible reranker threshold refuses even though dense would pass.
    assert not result.is_relevant


def test_results_are_ordered_by_fused_score(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "cancellation fee", limit=3)
    scores = [scored.fused_score for scored in result.results]
    assert scores == sorted(scores, reverse=True)


def test_gate_rejects_when_nothing_is_close_enough(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(
        conn, index, embedder, "cancellation", threshold=1.5
    )
    assert result.results
    assert not result.is_relevant, "an impossible threshold must never pass"


def test_gate_accepts_a_close_match(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(
        conn, index, embedder, "cancel the policy within 14 days", threshold=-1.0
    )
    assert result.is_relevant


def test_retrieval_records_the_threshold_used(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "excess", threshold=0.42)
    assert result.threshold == 0.42


def test_retrieval_defaults_to_the_configured_mode(indexed, monkeypatch):
    conn, index, embedder = indexed
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "sparse")
    result = retrieval.retrieve(conn, index, embedder, "excess")
    assert result.mode is Mode.SPARSE
    result = retrieval.retrieve(conn, index, embedder, "excess", mode=Mode.DENSE)
    assert result.mode is Mode.DENSE


def test_context_chunks_returns_plain_chunks(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "excess")
    chunks = result.context_chunks()
    assert len(chunks) == len(result.results)
    assert all(hasattr(chunk, "citation") for chunk in chunks)


def test_default_candidate_counts_come_from_config(indexed):
    conn, index, embedder = indexed
    result = retrieval.retrieve(conn, index, embedder, "policy terms cover")
    assert len(result.results) <= config.CONTEXT_CHUNKS


# --- query expansion -----------------------------------------------------


def test_expansion_embeds_every_variant_of_the_question(indexed, monkeypatch):
    """Each rewritten query must reach the embedder, not just the original."""
    conn, index, embedder = indexed
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", True)

    asked: list[str] = []
    original_embed = embedder.embed_query

    def recording(text: str):
        asked.append(text)
        return original_embed(text)

    monkeypatch.setattr(embedder, "embed_query", recording)
    question = "Under the policy, what excess applies to a flood claim?"
    retrieval.retrieve(conn, index, embedder, question, mode=Mode.DENSE)

    assert asked[0] == question, "the question as asked must be retrieved first"
    assert len(asked) > 1, "expansion must add at least one more phrasing"
    assert len(asked) <= config.QUERY_VARIANTS


def test_expansion_off_embeds_only_the_question(indexed, monkeypatch):
    conn, index, embedder = indexed
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", False)

    asked: list[str] = []
    original_embed = embedder.embed_query
    monkeypatch.setattr(
        embedder, "embed_query", lambda t: (asked.append(t), original_embed(t))[1]
    )
    retrieval.retrieve(
        conn, index, embedder, "what excess applies", mode=Mode.DENSE
    )
    assert len(asked) == 1


def test_expansion_keeps_scores_tied_to_the_original_question(indexed, monkeypatch):
    """Reported dense scores must describe the question the user asked."""
    conn, index, embedder = indexed
    question = "Under the policy, what excess applies to a flood claim?"

    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", False)
    plain = retrieval.retrieve(conn, index, embedder, question, mode=Mode.DENSE)
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", True)
    expanded = retrieval.retrieve(conn, index, embedder, question, mode=Mode.DENSE)

    shared = {s.chunk.id: s.dense_score for s in plain.results}
    for scored in expanded.results:
        if scored.chunk.id in shared:
            assert scored.dense_score == pytest.approx(shared[scored.chunk.id])
