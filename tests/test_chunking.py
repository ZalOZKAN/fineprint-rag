"""Tests for heading aware chunking."""

from __future__ import annotations

import config
from rag import chunking

POLICY = """# Home Insurance Policy

This document sets out the terms of your cover.

## Cancellation

You may cancel this policy within 14 days of the start date.

A cancellation fee of 25 EUR applies after that period.

## Excess

The excess is 500 EUR for each claim.

## What is not covered

Damage caused by wear and tear is not covered.
"""


def test_sections_are_keyed_by_heading():
    sections = chunking.split_sections(POLICY)
    headings = [section.heading for section in sections]
    assert headings == [
        "Home Insurance Policy",
        "Cancellation",
        "Excess",
        "What is not covered",
    ]


def test_text_before_first_heading_is_kept():
    sections = chunking.split_sections("intro line\n\n# Later\n\nbody")
    assert sections[0].heading == ""
    assert "intro line" in sections[0].body


def test_numbered_clause_lines_become_headings():
    text = "4.2 Termination\n\nEither party may terminate with notice.\n"
    sections = chunking.split_sections(text)
    assert sections[0].heading == "4.2 Termination"
    assert "terminate" in sections[0].body


def test_section_heading_travels_with_each_chunk():
    chunks = chunking.chunk_document(POLICY)
    excess = [content for heading, content in chunks if heading == "Excess"]
    assert excess and "500 EUR" in excess[0]


def test_paragraph_splitting_ignores_blank_runs():
    assert chunking.split_paragraphs("a\n\n\n\nb") == ["a", "b"]
    assert chunking.split_paragraphs("   ") == []


def test_short_paragraphs_are_packed_together():
    paragraphs = ["one sentence.", "another sentence.", "a third one."]
    packed = chunking.pack(paragraphs)
    assert len(packed) == 1


def test_packing_splits_when_target_exceeded():
    paragraph = "x" * (config.CHUNK_TARGET_CHARS - 10)
    packed = chunking.pack([paragraph, paragraph])
    assert len(packed) == 2


def test_oversized_paragraph_is_split_below_maximum():
    sentence = "This clause is long. " * 200
    packed = chunking.pack([sentence])
    assert len(packed) > 1
    assert all(len(piece) <= config.CHUNK_MAX_CHARS for piece in packed)


def test_single_unbroken_sentence_is_cut_by_length():
    packed = chunking.pack(["y" * (config.CHUNK_MAX_CHARS * 2)])
    assert all(len(piece) <= config.CHUNK_MAX_CHARS for piece in packed)


def test_consecutive_chunks_share_an_overlap():
    first = "alpha beta gamma delta epsilon"
    tail = chunking.overlap_tail(first, 0.5)
    assert tail
    assert first.endswith(tail)
    assert not tail.startswith(" ")


def test_overlap_tail_handles_edges():
    assert chunking.overlap_tail("", 0.5) == ""
    assert chunking.overlap_tail("text", 0.0) == ""


def test_tiny_trailing_chunk_is_merged_into_previous():
    body = "x" * (config.CHUNK_TARGET_CHARS - 10)
    document = f"## Section\n\n{body}\n\nshort\n"
    chunks = chunking.chunk_document(document)
    assert all(
        len(content) >= config.CHUNK_MIN_CHARS or heading != "Section"
        for heading, content in chunks
    )


def test_empty_document_produces_no_chunks():
    assert chunking.chunk_document("") == []
    assert chunking.chunk_document("\n\n   \n") == []


def test_embedding_text_prepends_heading():
    assert chunking.embedding_text("Excess", "500 EUR") == "Excess\n\n500 EUR"
    assert chunking.embedding_text("", "500 EUR") == "500 EUR"


def test_every_chunk_is_non_empty():
    chunks = chunking.chunk_document(POLICY)
    assert chunks
    assert all(content.strip() for _, content in chunks)
