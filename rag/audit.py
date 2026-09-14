"""Explain an answer sentence by sentence.

A grounded answer is only trustworthy if you can see where each claim came from.
This module takes a finished answer and its retrieved passages and, for every
sentence, reports which passage it most likely draws on, how close the match is,
and whether the two retrievers agreed on that passage. The interfaces show this
as a "how this answer was built" breakdown.

It is a diagnostic, not a second guardrail: it never changes the answer, only
describes it. A sentence with a weak best match is flagged, not removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.retrieval import ScoredChunk

# A sentence whose closest passage sits below this cosine similarity is flagged
# as weakly supported. Calibrated loosely: the point is to draw the eye, not to
# make a pass/fail call.
SUPPORT_THRESHOLD = 0.45

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Trailing citation lines the model appends, e.g. "[1] Source: policy.md, VII".
# These are not prose and should not be scored as answer sentences.
_CITATION_LINE = re.compile(r"^\[?\d+\]?[\s.:-]*(source|kaynak)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SentenceTrace:
    """Where one sentence of the answer most likely came from."""

    sentence: str
    source_citation: str | None
    similarity: float
    dense_rank: int | None
    sparse_rank: int | None
    supported: bool

    @property
    def retriever(self) -> str:
        """Which retriever surfaced the matched passage."""
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "none"


@dataclass(frozen=True)
class AnswerAudit:
    """The full sentence-by-sentence account of an answer."""

    traces: list[SentenceTrace]

    @property
    def supported_fraction(self) -> float:
        """Share of sentences whose best passage clears the support threshold."""
        if not self.traces:
            return 0.0
        return sum(t.supported for t in self.traces) / len(self.traces)

    @property
    def weak_sentences(self) -> list[SentenceTrace]:
        return [t for t in self.traces if not t.supported]


def split_sentences(text: str) -> list[str]:
    """Break answer text into sentences, keeping ones of real length.

    Lines that are just a source citation the model tacked on are dropped: they
    are not claims the answer makes, so scoring them as unsupported prose only
    drags the supported fraction down.
    """
    kept: list[str] = []
    for line in text.strip().splitlines():
        if _CITATION_LINE.match(line.strip()):
            continue
        for part in _SENTENCE_SPLIT.split(line.strip()):
            if len(part.strip()) > 12 and not _CITATION_LINE.match(part.strip()):
                kept.append(part.strip())
    return kept


def audit_answer(
    answer_text: str,
    sources: list[ScoredChunk],
    embedder,
    threshold: float = SUPPORT_THRESHOLD,
) -> AnswerAudit:
    """Trace every answer sentence back to its closest retrieved passage.

    Each sentence and each source passage is embedded once, and the sentence is
    attributed to the passage with the highest cosine similarity. Because the
    embedder L2 normalizes, cosine similarity is a dot product.
    """
    sentences = split_sentences(answer_text)
    if not sentences or not sources:
        return AnswerAudit(traces=[])

    source_texts = [scored.chunk.content for scored in sources]
    source_vectors = embedder.embed_texts(source_texts)
    sentence_vectors = embedder.embed_texts(sentences)

    traces: list[SentenceTrace] = []
    for sentence, vector in zip(sentences, sentence_vectors):
        similarities = source_vectors @ vector
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        scored = sources[best]
        traces.append(
            SentenceTrace(
                sentence=sentence,
                source_citation=scored.chunk.citation(),
                similarity=score,
                dense_rank=scored.dense_rank,
                sparse_rank=scored.sparse_rank,
                supported=score >= threshold,
            )
        )
    return AnswerAudit(traces=traces)


def format_audit(audit: AnswerAudit) -> str:
    """Render an audit as plain text for the command line."""
    if not audit.traces:
        return "  (no sentences to trace)"
    lines = [
        f"  {audit.supported_fraction:.0%} of sentences have a strong source match"
    ]
    for index, trace in enumerate(audit.traces, start=1):
        mark = "ok  " if trace.supported else "WEAK"
        lines.append(
            f"  [{mark}] {index}. sim {trace.similarity:.2f}  "
            f"via {trace.retriever}  {trace.source_citation}"
        )
        lines.append(f"         {trace.sentence[:96]}")
    return "\n".join(lines)
