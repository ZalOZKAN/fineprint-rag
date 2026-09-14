"""Automatic relevance-gate calibration for a library's own documents.

RERANK_RELEVANCE_THRESHOLD in config.py was calibrated against the sample
regulatory corpus with a labelled golden set (see evaluation/calibrate_threshold.py
and docs/adr/0008). A cross-encoder score is a raw logit, not a probability, so
it is not portable to a document of a different kind the way a percentage would
be: a genuinely relevant passage in a project plan or a lease can score far
below what a genuinely relevant passage in a federal regulation scores - low
enough that the fixed 1.7 refuses it outright (see Log.md, 2026-09-14).

There is no labelled golden set for a personal library, so this estimates a
threshold without one, from two proxy signals scored against the library's own
chunks:

  - "clearly relevant": each sampled chunk turned into a pseudo-question (its
    own heading, or its first few words) and scored against its own content.
    An easy case - the passage all but repeats the question - so it reads as
    an optimistic ceiling, not a realistic answer score.
  - "clearly not relevant": a handful of fixed, generic, off-topic questions
    (weather, recipes, sports) that have nothing to do with any real document,
    scored against a sample of this library's chunks. The *worst* (highest)
    score any of them reaches against this specific corpus is what actually
    matters: it is the concrete risk that a genuinely unrelated question
    accidentally scores high enough to slip through, and it is what an
    earlier version of this calibration got wrong by averaging instead of
    taking that worst case, which let "What is the capital of France?" pass
    (score -8.5) against a threshold estimated at -8.8.

The gate sits at a margin above that worst case, not at the midpoint between
the two signals: the "clearly relevant" ceiling is optimistic (see above), so
splitting the difference with it evenly is closer to the risky side than it
looks.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from rag import db
from rag.db import Chunk

# Ordinary questions about ordinary things, chosen only for having nothing to
# do with the kind of document this app is for. Scoring these against the
# library's own chunks measures the concrete risk of this specific corpus,
# not a generic assumption about the cross-encoder.
_OFF_TOPIC_PROBES = (
    "What is the capital of France?",
    "What's a good recipe for chocolate chip cookies?",
    "Who won the most recent World Cup?",
    "Can you recommend a good movie to watch tonight?",
    "What's the weather forecast for tomorrow?",
)

# Where above the worst off-topic score to place the cutoff, as a fraction of
# the gap up to the (optimistic) self-similarity ceiling. Kept on the strict
# side of the midpoint: a refusal the user can rephrase around costs less than
# a confident answer built from an unrelated passage.
_MARGIN_FRACTION = 0.5

# Enough chunks to average out noise from any one odd pseudo-question, and to
# give the off-topic probes a real sample of the corpus to score against; few
# enough that recalibrating after every upload stays fast. Each sampled chunk
# costs one reranker call for the positive signal; the probes cost one call
# each (a reranker call scores a whole batch of passages against one query).
_SAMPLE_SIZE = 40


def _pseudo_question(chunk: Chunk) -> str:
    """A stand-in for "a question this chunk would answer": its own heading,
    or, when a chunk has none, its first dozen or so words."""
    heading = chunk.heading.strip()
    if heading:
        return heading
    return " ".join(chunk.content.split()[:12])


def estimate_rerank_threshold(
    reranker,
    chunks: Sequence[Chunk],
    sample_size: int = _SAMPLE_SIZE,
    margin_fraction: float = _MARGIN_FRACTION,
    rng_seed: int = 0,
) -> float | None:
    """Estimate a relevance-gate cutoff from a library's own chunks.

    Returns None when there is too little to calibrate from (fewer than two
    chunks, every pseudo-question came up empty, or the two proxy signals do
    not separate at all) - the caller should fall back to
    config.RERANK_RELEVANCE_THRESHOLD in that case.
    """
    if reranker is None or len(chunks) < 2:
        return None

    rng = random.Random(rng_seed)
    sample = list(chunks) if len(chunks) <= sample_size else rng.sample(list(chunks), sample_size)

    positives: list[float] = []
    for chunk in sample:
        question = _pseudo_question(chunk)
        if not question:
            continue
        (score,) = reranker.score(question, [chunk.content])
        positives.append(score)
    if not positives:
        return None
    mean_pos = sum(positives) / len(positives)

    contents = [chunk.content for chunk in sample]
    worst_off_topic = max(
        (max(reranker.score(probe, contents), default=float("-inf")))
        for probe in _OFF_TOPIC_PROBES
    )

    if mean_pos <= worst_off_topic:
        # No separation between "answers a question" and "answers nothing in
        # particular" - a number computed from this is noise, not a signal.
        return None
    return worst_off_topic + margin_fraction * (mean_pos - worst_off_topic)


def calibrate_and_store(conn, reranker) -> float | None:
    """Recalibrate from the library's current chunks and persist the result.

    Called after any change to the library (an upload, a removal, loading the
    sample corpus) so the gate always reflects what is actually in it. Clears
    the stored value when calibration is not possible, so a stale number from
    a previously larger library never lingers.
    """
    chunk_ids, _ = db.load_embeddings(conn)
    chunks = db.fetch_chunks(conn, chunk_ids)
    threshold = estimate_rerank_threshold(reranker, chunks)
    if threshold is None:
        db.delete_meta(conn, db.META_RERANK_THRESHOLD)
    else:
        db.set_meta(conn, db.META_RERANK_THRESHOLD, repr(threshold))
    return threshold
