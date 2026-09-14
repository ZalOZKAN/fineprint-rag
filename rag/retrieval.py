"""Find the passages that answer a question.

Two searches run over the same database. Dense search compares embeddings and
finds passages that mean the same thing as the question even when they share no
words. Sparse search (BM25 through SQLite FTS5) finds exact tokens: clause
numbers, amounts, dates, defined terms. Contracts are full of those, and an
embedding blurs them, so neither search is sufficient alone.

Optionally the question is also retrieved with a couple of rephrasings of itself
(see rag/rewrite.py), each contributing its own candidate list, which is how a
passage the original wording misses can still reach the pool.

The candidate lists are combined with Reciprocal Rank Fusion, which scores a
passage by its position in each list rather than by the raw scores. That matters
because a cosine similarity of 0.7 and a BM25 score of 12 are not comparable
numbers, while ranks always are. A passage that both searches rank highly beats
one that only a single search loves.

Retrieval finally applies a gate: if no candidate is semantically close enough to
the question, the caller is told the corpus has no answer and the chat model is
never invoked. Refusing in code is deterministic and testable, which asking a
small local model to refuse is not.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

import config
from rag import db, rewrite
from rag.db import Chunk


class Mode(str, Enum):
    """Retrieval strategies. The default is set in config.RETRIEVAL_MODE."""

    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


def default_mode() -> "Mode":
    """The configured retrieval strategy."""
    return Mode(config.RETRIEVAL_MODE)


@dataclass(frozen=True)
class ScoredChunk:
    """One retrieved passage with the scores that selected it."""

    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Retrieval:
    """The outcome of one search over the corpus."""

    question: str
    mode: Mode
    results: list[ScoredChunk]
    best_dense_score: float
    threshold: float
    best_rerank_score: float | None = None
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD

    @property
    def is_relevant(self) -> bool:
        """Whether anything was close enough to be worth answering from.

        When a reranker ran, its top score is the gate signal: a cross-encoder
        judges whether a passage actually answers the question, and its scores
        move far less than a bi-encoder cosine when the corpus grows and the
        vector space gets crowded. Without a reranker the dense cosine is used.

        RERANK_RELEVANCE_THRESHOLD was calibrated on the sample regulatory
        corpus (ADR 8); a cross-encoder's raw score is not portable to an
        unrelated document the way a probability would be, so a personal
        library of a different kind of document can sit at a very different
        level and needs its own cutoff. rerank_threshold is a parameter, not a
        constant, so a caller (the app's Advanced settings) can override it
        per library instead of everyone sharing one hardcoded number.
        """
        if not self.results:
            return False
        if self.best_rerank_score is not None:
            return self.best_rerank_score >= self.rerank_threshold
        return self.best_dense_score >= self.threshold

    def context_chunks(self) -> list[Chunk]:
        return [scored.chunk for scored in self.results]


class Index:
    """In memory copy of the embedding matrix.

    Reading every vector out of SQLite on each question would dominate query
    time, so the matrix is loaded once and reused. Vectors were L2 normalized at
    write time, which makes cosine similarity a single dot product and the whole
    search one matrix multiplication.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.chunk_ids: list[int] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._position: dict[int, int] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload vectors from the database, for example after ingestion."""
        self.chunk_ids, self.matrix = db.load_embeddings(self._conn)
        self._position = {chunk_id: i for i, chunk_id in enumerate(self.chunk_ids)}

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query_vector: np.ndarray, limit: int) -> list[tuple[int, float]]:
        """Return the closest chunk ids with their cosine similarity, best first."""
        if self.size == 0 or query_vector.size == 0:
            return []
        scores = self.matrix @ query_vector.astype(np.float32)
        count = min(limit, len(scores))
        # argpartition finds the top values without sorting the whole array.
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    def score_of(self, chunk_id: int, query_vector: np.ndarray) -> float:
        """Cosine similarity of one chunk, or 0.0 if it is not in the index."""
        position = self._position.get(chunk_id)
        if position is None or self.matrix.size == 0:
            return 0.0
        return float(self.matrix[position] @ query_vector.astype(np.float32))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = config.RRF_K
) -> dict[int, float]:
    """Fuse ranked id lists into one score per id.

    Each list contributes 1 / (k + rank) with rank starting at 1. The constant k
    damps the difference between the top positions, so being present in several
    lists outweighs being first in only one.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (k + rank)
    return fused


def retrieve(
    conn: sqlite3.Connection,
    index: Index,
    embedder,
    question: str,
    mode: Mode | None = None,
    limit: int = config.CONTEXT_CHUNKS,
    threshold: float = config.RELEVANCE_THRESHOLD,
    reranker=None,
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD,
) -> Retrieval:
    """Search the corpus and return the passages worth answering from.

    When a reranker is passed, retrieval casts a wider net (RERANK_CANDIDATES
    per search instead of DENSE/SPARSE_CANDIDATES) and the cross-encoder reorders
    the pooled candidates; the top `limit` of that order are returned.
    """
    mode = mode or default_mode()
    question = question.strip()
    if not question:
        return Retrieval(question, mode, [], 0.0, threshold, rerank_threshold=rerank_threshold)

    pool = config.RERANK_CANDIDATES if reranker is not None else None
    dense_n = pool or config.DENSE_CANDIDATES
    sparse_n = pool or config.SPARSE_CANDIDATES

    # The question is retrieved with as written, and optionally with a couple of
    # rephrasings alongside it. Every phrasing contributes its own ranking to the
    # fusion, so a passage that only one of them surfaces can still reach the
    # pool. The original is always first and stays the scored one: the gate
    # threshold and the reranker are both calibrated against that phrasing.
    queries = rewrite.expand(question) or [question]
    query_vector = embedder.embed_query(question)

    dense_hits: list[tuple[int, float]] = []
    sparse_hits: list[tuple[int, float]] = []
    rankings: list[list[int]] = []
    for position, query in enumerate(queries):
        vector = query_vector if position == 0 else embedder.embed_query(query)
        if mode in (Mode.HYBRID, Mode.DENSE):
            hits = index.search(vector, dense_n)
            if position == 0:
                dense_hits = hits
            if hits:
                rankings.append([cid for cid, _ in hits])
        if mode in (Mode.HYBRID, Mode.SPARSE):
            hits = db.search_fts(conn, query, sparse_n)
            if position == 0:
                sparse_hits = hits
            if hits:
                rankings.append([cid for cid, _ in hits])

    # Reported per-passage scores and ranks describe the original question, so
    # that what the interface shows is the question the user actually asked.
    dense_scores = dict(dense_hits)
    sparse_scores = dict(sparse_hits)
    dense_ranks = {cid: i for i, (cid, _) in enumerate(dense_hits, start=1)}
    sparse_ranks = {cid: i for i, (cid, _) in enumerate(sparse_hits, start=1)}

    fused = reciprocal_rank_fusion(rankings)
    if not fused:
        return Retrieval(question, mode, [], 0.0, threshold, rerank_threshold=rerank_threshold)

    # Without a reranker the fused order is final, so only `limit` are built.
    # With one, the candidate pool is built, scored by the cross-encoder, and cut
    # to `limit` afterwards. Expansion fuses several rankings and so produces a
    # larger pool; it is capped back to RERANK_CANDIDATES so that the variable
    # under test stays *which* candidates are reranked and not *how many*, which
    # ADR 8 already measured as neutral at best.
    if reranker is None:
        keep = limit
    elif len(queries) > 1:
        keep = config.RERANK_CANDIDATES
    else:
        keep = len(fused)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:keep]
    chunk_ids = [chunk_id for chunk_id, _ in ordered]
    chunks = {chunk.id: chunk for chunk in db.fetch_chunks(conn, chunk_ids)}

    candidates: list[ScoredChunk] = []
    for chunk_id, fused_score in ordered:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            continue
        # A passage found only by BM25 still needs a dense score, otherwise the
        # relevance gate would reject exactly the exact term matches that
        # hybrid search was added to catch.
        dense_score = dense_scores.get(chunk_id)
        if dense_score is None:
            dense_score = index.score_of(chunk_id, query_vector)
        candidates.append(
            ScoredChunk(
                chunk=chunk,
                dense_score=dense_score,
                sparse_score=sparse_scores.get(chunk_id, 0.0),
                fused_score=fused_score,
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
            )
        )

    # The gate asks "was anything relevant retrieved at all", so it looks at the
    # whole pool, not just the passages that survive reranking.
    best_dense = max((scored.dense_score for scored in candidates), default=0.0)

    best_rerank: float | None = None
    if reranker is not None and candidates:
        scores = reranker.score(question, [c.chunk.content for c in candidates])
        scored_pairs = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        results = [
            replace(chunk, rerank_score=score)
            for chunk, score in scored_pairs[:limit]
        ]
        best_rerank = scored_pairs[0][1] if scored_pairs else None
    else:
        results = candidates[:limit]

    return Retrieval(
        question, mode, results, best_dense, threshold,
        best_rerank_score=best_rerank, rerank_threshold=rerank_threshold,
    )
