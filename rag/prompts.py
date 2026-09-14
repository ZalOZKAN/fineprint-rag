"""Prompt construction and the rules about what Fineprint will not answer.

Two guards sit in front of the model, both deterministic:

1. The relevance gate in retrieval.py stops questions the corpus cannot answer.
2. The advice heuristic below stops questions that ask for a recommendation
   rather than for what a document says.

Neither relies on the model choosing to refuse. A small local model asked
politely to decline will often answer anyway, so refusal is decided in code
where it can be tested and measured.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.db import Chunk

SYSTEM_PROMPT = """You are Fineprint, an assistant that answers questions about \
the user's own documents.

Rules:
- Answer using only the context below. Never use outside knowledge.
- Be brief: two or three sentences at most. Give the figure or rule directly,
  then stop. Do not restate the question or explain your reasoning.
- Start with the answer itself. Never begin with a label such as "Answer:"
  or its translation, and never copy a "[n] Source: ..." context block, or
  any part of one, into your reply - the reader already sees the passages
  separately and only wants your answer.
- Quote figures, dates and deadlines exactly as they appear.
- Name the section your answer comes from, for example "under Cancellation".
  Write it as plain text, never as a markdown link.
- If the context does not contain the answer, say so plainly and stop.
- You report what the document says. You do not give legal or financial advice,
  and you do not tell the user what they should do.

Context:
{context}"""

NO_ANSWER_MESSAGE = (
    "I could not find this in your documents. Try rephrasing the question, or "
    "check that the relevant document has been added to the library."
)

GENERATION_FAILED_MESSAGE = (
    "The model returned an empty answer. This happens occasionally with a local "
    "model of this size. Please ask again."
)

ADVICE_MESSAGE = (
    "I can only tell you what your documents say, not what you should do. "
    "Ask me what a document states about this, or consult a qualified adviser."
)

# Phrasings that ask for a recommendation or a legal judgement rather than for
# the contents of a document. Matched against the whole question, case
# insensitively. This is a heuristic and is measured as such in the evaluation
# set: it trades some recall for a refusal that is deterministic and testable.
ADVICE_PATTERNS = (
    r"\bshould i\b",
    r"\bshould we\b",
    r"\bwhat would you do\b",
    r"\bdo you (?:recommend|think|advise)\b",
    r"\bwhat do you (?:recommend|advise|suggest)\b",
    r"\bis it (?:legal|illegal|lawful|worth|worth it|worthwhile|a good idea)\b",
    # "is this fair", "is the deductible fair compared with...", "is that a scam".
    # Excludes the legal terms "fair market value" and "fair proportion" so a
    # question about those clauses is not mistaken for a request for judgement.
    r"\bis (?:it|this|that|the [\w'\- ]{1,40}?) "
    r"(?:a )?(?:fair|reasonable|worth it|worthwhile|a scam|a rip-?off|a good deal)\b"
    r"(?!\s+(?:market|value|proportion|share|dealing))",
    r"\bcan i sue\b",
    r"\bshould i sue\b",
    r"\bdo i have a case\b",
    r"\bwill i win\b",
    r"\badvise me\b",
    r"\byour (?:opinion|advice)\b",
)

_ADVICE_REGEX = re.compile("|".join(ADVICE_PATTERNS), re.IGNORECASE)


def is_advice_request(question: str) -> bool:
    """Whether a question asks for a recommendation rather than a document fact."""
    return bool(_ADVICE_REGEX.search(question or ""))


def format_context(chunks: Sequence[Chunk]) -> str:
    """Render retrieved passages as a numbered, cited block for the prompt."""
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{position}] Source: {chunk.citation()}\n{chunk.content}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(chunks))},
        {"role": "user", "content": question},
    ]
