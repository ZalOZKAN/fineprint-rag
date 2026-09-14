"""Tests for document loading and the ingestion pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from rag import db, ingest, loaders
from rag.embeddings import l2_normalize


# --- loaders -------------------------------------------------------------


def test_discover_finds_supported_files_sorted(corpus):
    (corpus / "notes.rtf").write_text("ignored", encoding="utf-8")
    found = [path.name for path in loaders.discover_documents(corpus)]
    assert found == ["lease.md", "policy.md"]


def test_discover_on_missing_directory(tmp_path):
    assert loaders.discover_documents(tmp_path / "nope") == []


def test_load_markdown_document(corpus):
    document = loaders.load_document(corpus / "policy.md")
    assert document.title == "Home Insurance Policy"
    assert "Cancellation" in document.markdown
    assert len(document.sha256) == 64


def test_unsupported_suffix_is_rejected(tmp_path):
    path = tmp_path / "scan.rtf"
    path.write_text("x" * 500, encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        loaders.load_document(path)


def test_empty_document_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        loaders.load_document(path)


def test_short_text_file_is_allowed(tmp_path):
    # The length floor guards PDF extraction only, a short note is legitimate.
    path = tmp_path / "note.md"
    path.write_text("# Note\n\nThe excess is 500 EUR.\n", encoding="utf-8")
    assert loaders.load_document(path).title == "Note"


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one", encoding="utf-8")
    first = loaders.file_hash(path)
    path.write_text("two", encoding="utf-8")
    assert loaders.file_hash(path) != first


def test_title_falls_back_to_filename(tmp_path):
    path = tmp_path / "my_policy_document.txt"
    path.write_text("no heading here, just body text. " * 20, encoding="utf-8")
    document = loaders.load_document(path)
    assert document.title == "my policy document"


# --- embeddings helper ---------------------------------------------------


def test_l2_normalize_gives_unit_rows():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert normalized.dtype == np.float32


def test_l2_normalize_leaves_zero_rows_alone():
    normalized = l2_normalize(np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(normalized, 0.0)


def test_l2_normalize_on_empty_input():
    assert l2_normalize(np.zeros((0, 0), dtype=np.float32)).size == 0


# --- ingestion -----------------------------------------------------------


def test_ingest_stores_documents_and_chunks(conn, embedder, corpus):
    report = ingest.ingest_directory(conn, embedder, corpus)
    assert sorted(report.ingested) == ["lease.md", "policy.md"]
    assert report.failed == []
    assert db.count_documents(conn) == 2
    assert db.count_chunks(conn) > 0
    assert report.chunk_count == db.count_chunks(conn)


def test_ingest_records_embedding_metadata(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    assert db.get_meta(conn, db.META_EMBEDDING_DIMENSION) == str(embedder.dimension)
    assert db.get_meta(conn, db.META_EMBEDDING_MODEL) == "fake-embedder"


def test_second_run_skips_unchanged_documents(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    chunks_after_first = db.count_chunks(conn)
    calls_after_first = embedder.calls

    report = ingest.ingest_directory(conn, embedder, corpus)

    assert report.ingested == []
    assert sorted(report.skipped) == ["lease.md", "policy.md"]
    assert db.count_chunks(conn) == chunks_after_first
    assert embedder.calls == calls_after_first, "unchanged files must not be re-embedded"


def test_changed_document_is_reindexed_without_duplicates(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    before = db.count_documents(conn)

    (corpus / "policy.md").write_text(
        "# Home Insurance Policy\n\n## Excess\n\n"
        "The excess payable is now 750 EUR for each and every claim made under "
        "this policy, replacing the previous amount stated in earlier versions.\n",
        encoding="utf-8",
    )
    report = ingest.ingest_directory(conn, embedder, corpus)

    assert report.ingested == ["policy.md"]
    assert db.count_documents(conn) == before, "reingesting must not duplicate documents"
    ids, _ = db.load_embeddings(conn)
    contents = " ".join(chunk.content for chunk in db.fetch_chunks(conn, ids))
    assert "750 EUR" in contents
    assert "500 EUR" not in contents, "stale chunks must be removed"


def test_force_reindexes_unchanged_documents(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    report = ingest.ingest_directory(conn, embedder, corpus, force=True)
    assert sorted(report.ingested) == ["lease.md", "policy.md"]
    assert report.skipped == []


def test_broken_document_is_reported_without_stopping_the_run(conn, embedder, corpus):
    (corpus / "broken.txt").write_text("   ", encoding="utf-8")
    report = ingest.ingest_directory(conn, embedder, corpus)
    assert [name for name, _ in report.failed] == ["broken.txt"]
    assert sorted(report.ingested) == ["lease.md", "policy.md"]


def test_fts_index_is_populated_after_ingest(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    results = db.search_fts(conn, "sublet", limit=5)
    assert results
    matched = db.fetch_chunks(conn, [results[0][0]])[0]
    assert "sublet" in matched.content.lower()


def test_ingest_on_empty_directory(conn, embedder, tmp_path):
    report = ingest.ingest_directory(conn, embedder, tmp_path)
    assert report.ingested == []
    assert report.chunk_count == 0
    assert "0 ingested" in report.summary()


def test_title_skips_a_blockquote_banner(tmp_path):
    # Public-domain documents open with a `>` licence banner. The real heading
    # comes after it and should still become the title.
    path = tmp_path / "policy.md"
    path.write_text(
        "> PUBLIC DOMAIN. Reproduced from 44 CFR Part 61.\n\n"
        "# Standard Flood Insurance Policy\n\n"
        "This policy insures against direct physical loss by or from flood.\n",
        encoding="utf-8",
    )
    assert loaders.load_document(path).title == "Standard Flood Insurance Policy"


def test_title_still_falls_back_when_there_is_no_heading(tmp_path):
    path = tmp_path / "notes_about_cover.md"
    path.write_text(
        "> a banner line\n\nplain body text with no heading at all, several words.\n",
        encoding="utf-8",
    )
    assert loaders.load_document(path).title == "notes about cover"
