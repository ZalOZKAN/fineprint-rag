"""Split markdown documents into retrievable passages.

Policies and contracts are heavily structured: "Cancellation", "Excess",
"What is not covered". Splitting on a fixed character count would cut across
those boundaries and produce passages that no longer say which clause they came
from. Instead this module splits on markdown headings first, and only falls back
to size based splitting inside a section that is too long to embed usefully.

Each chunk carries its heading, which serves two purposes: the heading is
prepended to the embedded text so that retrieval sees the section name, and it
is shown to the user as the citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Numbered clause openers such as "4.2 Cancellation" or "Section 8 - Termination".
# The number must be followed by a real separator (a dot, paren, colon or dash),
# or be a dotted decimal. A bare "1 Pay its own appraiser; and" is an ordinary
# list item, not a heading, and must not match: it has no separator, ends with a
# conjunction, and is a sentence fragment rather than a title.
CLAUSE_PATTERN = re.compile(
    r"^\s*(?:section\s+)?"
    r"(\d+\.\d+(?:\.\d+)*|\d+\s*[.):-])\s*"      # "4.2" or "8." / "8)" / "8:" / "8-"
    r"([A-Z][A-Za-z][^.;]{1,38})\s*$",           # short Title-ish phrase, no ; or .
    re.IGNORECASE,
)

# Phrases that mark the title as a sentence fragment rather than a clause name.
_FRAGMENT_TAIL = re.compile(r"\b(and|or|the|of|to|a|an|is|are|will|must|may)\s*$", re.I)


@dataclass(frozen=True)
class Section:
    """A run of body text that appeared under one heading."""

    heading: str
    body: str


def split_sections(markdown: str) -> list[Section]:
    """Group document lines into sections keyed by the heading above them.

    Text appearing before the first heading is kept under an empty heading so
    that front matter is never silently dropped.
    """
    sections: list[Section] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(heading=heading, body=body))
        buffer.clear()

    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)
        if match is None:
            match = CLAUSE_PATTERN.match(line)
            if match is not None and not _FRAGMENT_TAIL.search(match.group(2)):
                # Rebuild "4.2 Cancellation" as a single heading string.
                number = match.group(1).rstrip(".):-").strip()
                candidate = f"{number} {match.group(2)}".strip()
            else:
                candidate = None
        else:
            candidate = match.group(2).strip()

        if candidate:
            flush()
            heading = candidate
            continue
        buffer.append(line)

    flush()
    return sections


def split_paragraphs(body: str) -> list[str]:
    """Break a section body into paragraphs, discarding blank runs."""
    return [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]


def overlap_tail(text: str, ratio: float) -> str:
    """Return the trailing slice of text used to overlap the next chunk.

    The slice is trimmed forward to the next word boundary so that a chunk never
    begins mid word.
    """
    if ratio <= 0 or not text:
        return ""
    size = int(len(text) * ratio)
    if size <= 0:
        return ""
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def pack(paragraphs: list[str]) -> list[str]:
    """Accumulate paragraphs into chunks close to the configured target size.

    A paragraph longer than CHUNK_MAX_CHARS on its own is split by sentence.
    Consecutive chunks share an overlapping tail so a statement that straddles a
    boundary is still found by search.
    """
    chunks: list[str] = []
    current = ""

    def commit() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        for piece in _split_oversized(paragraph):
            if not current:
                current = piece
                continue
            if len(current) + len(piece) + 2 <= config.CHUNK_TARGET_CHARS:
                current = f"{current}\n\n{piece}"
                continue
            commit()
            # commit() drops whitespace only content, so chunks may still be
            # empty here and there is nothing to overlap with.
            tail = (
                overlap_tail(chunks[-1], config.CHUNK_OVERLAP_RATIO) if chunks else ""
            )
            # Overlap is a convenience, never a reason to exceed the maximum.
            if tail and len(tail) + len(piece) + 2 <= config.CHUNK_MAX_CHARS:
                current = f"{tail}\n\n{piece}"
            else:
                current = piece

    commit()
    return chunks


def _split_oversized(paragraph: str) -> list[str]:
    """Split a single paragraph that exceeds the maximum chunk size."""
    if len(paragraph) <= config.CHUNK_MAX_CHARS:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > config.CHUNK_TARGET_CHARS:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current.strip())

    # A single sentence longer than the maximum is cut on character count.
    # Whitespace only remainders are dropped, since they carry no meaning and
    # would otherwise become empty chunks.
    result: list[str] = []
    for piece in pieces:
        while len(piece) > config.CHUNK_MAX_CHARS:
            head = piece[: config.CHUNK_MAX_CHARS]
            if head.strip():
                result.append(head)
            piece = piece[config.CHUNK_MAX_CHARS :]
        if piece.strip():
            result.append(piece)
    return result


def chunk_document(markdown: str) -> list[tuple[str, str]]:
    """Turn a markdown document into (heading, content) pairs ready to embed.

    Chunks shorter than CHUNK_MIN_CHARS are merged into the previous chunk of the
    same section, since a fragment such as a lone heading line carries no
    retrievable meaning on its own.
    """
    results: list[tuple[str, str]] = []
    for section in split_sections(markdown):
        pieces = pack(split_paragraphs(section.body))
        for piece in pieces:
            if (
                len(piece) < config.CHUNK_MIN_CHARS
                and results
                and results[-1][0] == section.heading
            ):
                previous_heading, previous_content = results[-1]
                results[-1] = (previous_heading, f"{previous_content}\n\n{piece}")
                continue
            results.append((section.heading, piece))
    return results


def embedding_text(heading: str, content: str) -> str:
    """Text actually sent to the embedding model.

    The heading is prepended so that a passage about cancellation still matches a
    question phrased around cancellation even when the body never repeats the
    word.
    """
    return f"{heading}\n\n{content}" if heading else content
