"""Turn one question into the few queries worth retrieving with.

Retrieval is the measured ceiling on this project: the answering clause reaches
the top three passages about 69 % of the time on the main set and 53 % on the
hard set, and reranking cannot lift that because it only reorders the pool it is
handed (see docs/adr/0008). The pool itself is built from a single embedding of
the question as the user typed it, and that phrasing is not always the phrasing
that retrieves well.

Two things get in the way. A question carries a long interrogative opening
("within how many hours must an employer report...") that the answering clause
never contains, and it names the rule it is asking about ("under the Cooling-Off
Rule") in words that a near neighbour also uses, so the scope gets lost among
documents that share the vocabulary but not the subject.

So the question is expanded into a small set of queries, each retrieved with
separately, and their rankings fused by the same Reciprocal Rank Fusion that
already fuses dense and sparse. A passage that several phrasings agree on rises.
This is deterministic string work, no model call and no network: the rewrite
costs microseconds against tens of seconds of generation.
"""

from __future__ import annotations

import re

import config
from rag.db import STOPWORDS, TERM_PATTERN

# The rule or document a question scopes itself to: "under the Cooling-Off
# Rule", "under the Dwelling Form". The span ends at the first punctuation or
# at the word that starts the rest of the question, so that "under the Mail
# Order Rule, how must a refund be sent" yields the rule and not the whole line.
_SCOPE = re.compile(
    r"\bunder\s+(?:the\s+)?(.+?)"
    r"(?=[,?.;]|\s+(?:when|what|which|who|how|why|by|is|are|must|does|do|can|may)\b|$)",
    re.IGNORECASE,
)

# Agencies and programmes are written as acronyms in both questions and clauses,
# and they are the strongest scope signal a question can carry.
_ACRONYM = re.compile(r"\b[A-Z]{3,}\b")

_MAX_SCOPE_CHARS = 60


def content_words(question: str) -> str:
    """The question with its interrogative opening and filler words removed.

    Reuses the stopword list the keyword search already uses, so the two agree
    on what carries no signal. Order is kept and repeats are dropped.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for term in TERM_PATTERN.findall(question or ""):
        lowered = term.lower()
        if lowered in STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        kept.append(term)
    return " ".join(kept)


def scope(question: str) -> str:
    """The rule or document the question names, or an empty string."""
    match = _SCOPE.search(question or "")
    if match:
        named = match.group(1).strip(" ,.?;")
        if named and len(named) <= _MAX_SCOPE_CHARS:
            return named
    found = _ACRONYM.search(question or "")
    return found.group(0) if found else ""


def expand(question: str) -> list[str]:
    """The queries to retrieve with, the question itself always first.

    The original is never dropped: it is the phrasing the relevance gate and the
    reranker are calibrated against, and the variants are additions to the
    candidate pool rather than a replacement for it.
    """
    question = (question or "").strip()
    if not question:
        return []

    variants = [question]
    if config.QUERY_EXPANSION_ENABLED:
        content = content_words(question)
        named = scope(question)
        # Scope first: repeating the rule next to the content words is what
        # separates a clause about business days in one rule from the identical
        # defined term in another.
        if named and content:
            variants.append(f"{named} {content}")
        if content:
            variants.append(content)

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique[: config.QUERY_VARIANTS]
