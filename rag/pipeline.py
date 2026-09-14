"""The end to end question answering pipeline.

This is the only module the interfaces talk to. Both the CLI and the Streamlit
app call ask() and render the result, which keeps the retrieval and generation
logic testable without a user interface attached.

A question travels through three stages, and can stop at any of them:

    advice guard  ->  retrieval gate  ->  grounded generation

Stopping early is a feature. Two of the three evaluation categories are
questions the assistant is supposed to refuse.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import config
from rag import prompts, retrieval
from rag.retrieval import Index, Mode, ScoredChunk

logger = logging.getLogger(__name__)


class Outcome(str, Enum):
    """Why the pipeline produced the answer it did."""

    ANSWERED = "answered"
    NOT_IN_CORPUS = "not_in_corpus"
    ADVICE_REFUSED = "advice_refused"
    EMPTY_QUESTION = "empty_question"
    GENERATION_FAILED = "generation_failed"


@dataclass
class Answer:
    """A reply, together with everything needed to explain and evaluate it."""

    question: str
    text: str
    outcome: Outcome
    sources: list[ScoredChunk] = field(default_factory=list)
    best_dense_score: float = 0.0
    best_rerank_score: float | None = None
    threshold: float = config.RELEVANCE_THRESHOLD
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD
    mode: Mode = Mode.DENSE
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    generation_retried: bool = False

    @property
    def answered(self) -> bool:
        return self.outcome is Outcome.ANSWERED

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    def citations(self) -> list[str]:
        """Unique source references, in the order they were retrieved."""
        seen: list[str] = []
        for scored in self.sources:
            citation = scored.chunk.citation()
            if citation not in seen:
                seen.append(citation)
        return seen

    def to_log_record(self) -> dict:
        """A flat dictionary for the JSONL query log."""
        return {
            "question": self.question,
            "outcome": self.outcome.value,
            "mode": self.mode.value,
            "best_dense_score": round(self.best_dense_score, 4),
            "best_rerank_score": (
                round(self.best_rerank_score, 3)
                if self.best_rerank_score is not None
                else None
            ),
            "threshold": self.threshold,
            "rerank_threshold": self.rerank_threshold,
            "citations": self.citations(),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "generation_retried": self.generation_retried,
            "chunk_ids": [scored.chunk.id for scored in self.sources],
        }


def _refusal(question: str, text: str, outcome: Outcome, **kwargs) -> Answer:
    return Answer(question=question, text=text, outcome=outcome, **kwargs)


def ask(
    conn: sqlite3.Connection,
    index: Index,
    embedder,
    chat_model,
    question: str,
    mode: Mode | None = None,
    threshold: float = config.RELEVANCE_THRESHOLD,
    log_path: Path | None = None,
    reranker=None,
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD,
) -> Answer:
    """Answer a question, refusing when the guards say so."""
    mode = mode or retrieval.default_mode()
    answer, found = _guard(
        conn, index, embedder, question, mode, threshold, reranker, rerank_threshold
    )
    if answer is not None:
        log_answer(answer, log_path)
        return answer

    messages = prompts.build_messages(found.question, found.context_chunks())
    started = time.perf_counter()
    text = chat_model.complete(messages)
    generation_ms = (time.perf_counter() - started) * 1000

    # The model intermittently returns nothing at all. An empty string is not an
    # answer, and silently presenting one as though it were would hide the
    # failure. One retry is made because the same request usually succeeds on a
    # second attempt; if it fails again the failure is reported rather than hidden.
    retried = False
    quick_enough = generation_ms < config.RETRY_EMPTY_MAX_ELAPSED_SECONDS * 1000
    if not text.strip() and config.RETRY_ON_EMPTY_GENERATION and quick_enough:
        logger.warning("Empty generation, retrying once")
        retried = True
        started = time.perf_counter()
        text = chat_model.complete(messages)
        generation_ms += (time.perf_counter() - started) * 1000
    elif not text.strip():
        logger.warning(
            "Empty generation after %.0fs, not retrying (likely a stuck stream)",
            generation_ms / 1000,
        )

    outcome = Outcome.ANSWERED if text.strip() else Outcome.GENERATION_FAILED
    if outcome is Outcome.GENERATION_FAILED:
        text = prompts.GENERATION_FAILED_MESSAGE

    answer = Answer(
        question=found.question,
        text=text,
        outcome=outcome,
        sources=found.results,
        best_dense_score=found.best_dense_score,
        best_rerank_score=found.best_rerank_score,
        threshold=threshold,
        rerank_threshold=rerank_threshold,
        mode=mode,
        retrieval_ms=found.retrieval_ms,
        generation_ms=generation_ms,
        generation_retried=retried,
    )
    log_answer(answer, log_path)
    return answer


def ask_streaming(
    conn: sqlite3.Connection,
    index: Index,
    embedder,
    chat_model,
    question: str,
    mode: Mode | None = None,
    threshold: float = config.RELEVANCE_THRESHOLD,
    reranker=None,
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD,
) -> tuple[Answer, Iterator[str]]:
    """Return a partly filled answer plus a generator of answer fragments.

    The caller renders the fragments as they arrive, then reads the finished
    text from answer.text once the generator is exhausted.
    """
    mode = mode or retrieval.default_mode()
    answer, found = _guard(
        conn, index, embedder, question, mode, threshold, reranker, rerank_threshold
    )
    if answer is not None:
        return answer, iter([answer.text])

    answer = Answer(
        question=found.question,
        text="",
        outcome=Outcome.ANSWERED,
        sources=found.results,
        best_dense_score=found.best_dense_score,
        best_rerank_score=found.best_rerank_score,
        threshold=threshold,
        rerank_threshold=rerank_threshold,
        mode=mode,
        retrieval_ms=found.retrieval_ms,
    )
    messages = prompts.build_messages(found.question, found.context_chunks())

    def generate() -> Iterator[str]:
        started = time.perf_counter()
        pieces: list[str] = []
        from rag.llm import clean_answer

        for piece in chat_model.stream(messages):
            pieces.append(piece)
            yield piece

        # Same intermittent empty result as the blocking path. Retry only when
        # the empty attempt was quick: an attempt that hit the time ceiling is
        # stuck, and a retry just spends the budget a second time.
        elapsed = time.perf_counter() - started
        quick_enough = elapsed < config.RETRY_EMPTY_MAX_ELAPSED_SECONDS
        if (
            not "".join(pieces).strip()
            and config.RETRY_ON_EMPTY_GENERATION
            and quick_enough
        ):
            logger.warning("Empty generation, retrying once")
            answer.generation_retried = True
            for piece in chat_model.stream(messages):
                pieces.append(piece)
                yield piece
        elif not "".join(pieces).strip():
            logger.warning(
                "Empty generation after %.0fs, not retrying (likely a stuck stream)",
                elapsed,
            )

        answer.text = clean_answer("".join(pieces))
        answer.generation_ms = (time.perf_counter() - started) * 1000
        if not answer.text.strip():
            answer.outcome = Outcome.GENERATION_FAILED
            answer.text = prompts.GENERATION_FAILED_MESSAGE

    return answer, generate()


@dataclass
class _Found:
    """Retrieval output carried between the guard and generation stages."""

    question: str
    results: list[ScoredChunk]
    best_dense_score: float
    retrieval_ms: float
    best_rerank_score: float | None = None

    def context_chunks(self):
        return [scored.chunk for scored in self.results]


def _guard(
    conn: sqlite3.Connection,
    index: Index,
    embedder,
    question: str,
    mode: Mode | None,
    threshold: float,
    reranker=None,
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD,
) -> tuple[Answer | None, _Found | None]:
    """Apply both guards. Returns (refusal, None) or (None, retrieval output)."""
    mode = mode or retrieval.default_mode()
    question = (question or "").strip()
    if not question:
        return (
            _refusal(question, prompts.NO_ANSWER_MESSAGE, Outcome.EMPTY_QUESTION,
                     mode=mode, threshold=threshold, rerank_threshold=rerank_threshold),
            None,
        )

    if prompts.is_advice_request(question):
        return (
            _refusal(question, prompts.ADVICE_MESSAGE, Outcome.ADVICE_REFUSED,
                     mode=mode, threshold=threshold, rerank_threshold=rerank_threshold),
            None,
        )

    started = time.perf_counter()
    found = retrieval.retrieve(
        conn, index, embedder, question, mode=mode, threshold=threshold,
        reranker=reranker, rerank_threshold=rerank_threshold,
    )
    retrieval_ms = (time.perf_counter() - started) * 1000

    if not found.is_relevant:
        return (
            _refusal(question, prompts.NO_ANSWER_MESSAGE, Outcome.NOT_IN_CORPUS,
                     best_dense_score=found.best_dense_score,
                     best_rerank_score=found.best_rerank_score, threshold=threshold,
                     rerank_threshold=rerank_threshold,
                     mode=mode, retrieval_ms=retrieval_ms),
            None,
        )

    return None, _Found(
        question=question,
        results=found.results,
        best_dense_score=found.best_dense_score,
        best_rerank_score=found.best_rerank_score,
        retrieval_ms=retrieval_ms,
    )


def log_answer(answer: Answer, log_path: Path | None) -> None:
    """Append one JSON line describing the query, for later inspection."""
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(answer.to_log_record(), ensure_ascii=False) + "\n")
    except OSError as error:  # noqa: BLE001 - logging must never break a query
        logger.warning("Could not write query log: %s", error)
