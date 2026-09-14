"""Edge cases in chunking that produced real failures.

Each test here corresponds to a bug found while building the pipeline rather
than to a feature, so they are kept together and named after the failure.
"""

from __future__ import annotations

import config
from rag import chunking


def test_whitespace_tail_after_a_long_cut_does_not_crash():
    # A sentence longer than the maximum is cut on character count. When the
    # remainder is only spaces, commit() drops it, and reaching for the previous
    # chunk to build an overlap used to raise IndexError.
    paragraph = "z" * config.CHUNK_MAX_CHARS + "     "
    packed = chunking.pack([paragraph])
    assert packed
    assert all(piece.strip() for piece in packed)


def test_document_of_only_whitespace_produces_nothing():
    assert chunking.chunk_document("   \n\n\t\n   ") == []


def test_heading_with_no_body_is_dropped():
    chunks = chunking.chunk_document("## Empty Section\n\n## Real Section\n\nbody text")
    assert [heading for heading, _ in chunks] == ["Real Section"]


def test_overlap_never_pushes_a_chunk_past_the_maximum():
    paragraph = "w" * (config.CHUNK_TARGET_CHARS - 5)
    packed = chunking.pack([paragraph] * 4)
    assert all(len(piece) <= config.CHUNK_MAX_CHARS for piece in packed)


def test_all_produced_chunks_are_non_empty_after_cutting():
    packed = chunking.pack(["q" * (config.CHUNK_MAX_CHARS * 3) + "   "])
    assert packed
    assert all(piece.strip() for piece in packed)


def test_headings_deeper_than_two_levels_are_recognised():
    chunks = chunking.chunk_document("###### Deep\n\nbody text here")
    assert chunks[0][0] == "Deep"


def test_a_line_of_only_hashes_is_not_a_heading():
    sections = chunking.split_sections("###\n\nbody text")
    assert sections[0].heading == ""
