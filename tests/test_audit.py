"""Tests for the sentence-by-sentence answer audit."""

from __future__ import annotations


from rag import audit, ingest
from rag.retrieval import Index, retrieve


def test_split_sentences_keeps_real_sentences():
    text = "The excess is 500 EUR. It applies to every claim. OK."
    parts = audit.split_sentences(text)
    assert parts == ["The excess is 500 EUR.", "It applies to every claim."]


def test_split_sentences_drops_fragments():
    assert audit.split_sentences("Yes. No.") == []
    assert audit.split_sentences("") == []


def test_split_sentences_drops_trailing_citation_lines():
    text = (
        "You must send a proof of loss within 60 days after the loss.\n"
        "[1] Source: nfip_flood_insurance_policy.md, VII. General Conditions\n"
        "[2] Source: nfip_condominium_form.md, VIII. General Conditions"
    )
    parts = audit.split_sentences(text)
    assert parts == ["You must send a proof of loss within 60 days after the loss."]


def test_audit_with_no_sources_is_empty():
    result = audit.audit_answer("A sentence long enough to keep.", [], embedder=None)
    assert result.traces == []
    assert result.supported_fraction == 0.0


def _prepare(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    return Index(conn)


def test_audit_traces_each_sentence_to_a_source(conn, embedder, corpus):
    index = _prepare(conn, embedder, corpus)
    found = retrieve(conn, index, embedder, "notice period to end the tenancy")
    answer = "Either party may end the tenancy with 30 days written notice."
    result = audit.audit_answer(answer, found.results, embedder)

    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.source_citation is not None
    assert 0.0 <= trace.similarity <= 1.0
    assert trace.retriever in {"both", "dense", "sparse", "none"}


def test_audit_flags_sentences_below_the_threshold(conn, embedder, corpus):
    index = _prepare(conn, embedder, corpus)
    found = retrieve(conn, index, embedder, "notice period")
    answer = (
        "Either party may end the tenancy with thirty days written notice. "
        "Spacecraft propulsion relies on orbital mechanics and delta-v budgets."
    )
    # An impossible threshold: no sentence can clear it, so every sentence is
    # reported as weak and none as supported. (The fake embedder's absolute
    # similarity scores are not meaningful, so the threshold is asserted through
    # its effect, not through a specific score.)
    result = audit.audit_answer(answer, found.results, embedder, threshold=1.5)

    assert len(result.traces) == 2
    assert result.supported_fraction == 0.0
    assert len(result.weak_sentences) == 2
    assert all(not trace.supported for trace in result.traces)


def test_supported_fraction_counts_strong_matches(conn, embedder, corpus):
    index = _prepare(conn, embedder, corpus)
    found = retrieve(conn, index, embedder, "subletting the property")
    answer = "The tenant may not sublet without the landlord's written consent."
    result = audit.audit_answer(answer, found.results, embedder, threshold=-1.0)
    assert result.supported_fraction == 1.0
    assert result.weak_sentences == []


def test_retriever_label_reflects_which_search_found_the_passage():
    from rag.db import Chunk

    chunk = Chunk(1, 1, 0, "Excess", "text", "policy.md", "Policy")
    both = audit.SentenceTrace("s", "policy.md, Excess", 0.8, 1, 2, True)
    dense_only = audit.SentenceTrace("s", "c", 0.8, 1, None, True)
    sparse_only = audit.SentenceTrace("s", "c", 0.8, None, 3, True)
    neither = audit.SentenceTrace("s", "c", 0.8, None, None, True)
    assert both.retriever == "both"
    assert dense_only.retriever == "dense"
    assert sparse_only.retriever == "sparse"
    assert neither.retriever == "none"
    assert chunk.citation() == "policy.md, Excess"


def test_format_audit_is_readable(conn, embedder, corpus):
    index = _prepare(conn, embedder, corpus)
    found = retrieve(conn, index, embedder, "notice period")
    answer = "Either party may end the tenancy with 30 days notice."
    text = audit.format_audit(audit.audit_answer(answer, found.results, embedder))
    assert "sentences have a strong source match" in text
    assert "notice" in text.lower()


def test_format_audit_handles_empty():
    assert "no sentences" in audit.format_audit(audit.AnswerAudit(traces=[]))
