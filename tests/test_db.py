"""Tests for the SQLite storage layer."""

from __future__ import annotations

import numpy as np
import pytest

from rag import db


@pytest.fixture()
def conn(tmp_path):
    """A fresh database in a temporary directory."""
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def add_document(connection, filename="policy.pdf", chunks=None, dimension=4):
    """Insert a document with chunks and deterministic unit length embeddings."""
    chunks = chunks or [("Cancellation", "You may cancel within 14 days.")]
    document_id = db.replace_document(connection, filename, filename, "hash-" + filename)
    vectors = np.zeros((len(chunks), dimension), dtype=np.float32)
    for index in range(len(chunks)):
        vectors[index, index % dimension] = 1.0
    db.insert_chunks(connection, document_id, chunks, vectors)
    db.rebuild_fts(connection)
    return document_id


def test_schema_is_created_empty(conn):
    assert db.count_documents(conn) == 0
    assert db.count_chunks(conn) == 0


def test_meta_roundtrip(conn):
    assert db.get_meta(conn, "missing") is None
    db.set_meta(conn, "k", "v1")
    db.set_meta(conn, "k", "v2")
    assert db.get_meta(conn, "k") == "v2"


def test_insert_and_count(conn):
    add_document(conn, chunks=[("A", "first"), ("B", "second")])
    assert db.count_documents(conn) == 1
    assert db.count_chunks(conn) == 2


def test_list_documents_reports_title_and_passage_count(conn):
    assert db.list_documents(conn) == []
    add_document(conn, filename="beta.md", chunks=[("A", "first"), ("B", "second")])
    add_document(conn, filename="alpha.md", chunks=[("C", "third")])
    listed = db.list_documents(conn)
    assert [entry["filename"] for entry in listed] == ["alpha.md", "beta.md"]
    passages = {entry["filename"]: entry["passages"] for entry in listed}
    assert passages == {"alpha.md": 1, "beta.md": 2}


def test_embeddings_roundtrip_preserves_values(conn):
    add_document(conn, chunks=[("A", "first"), ("B", "second")])
    ids, matrix = db.load_embeddings(conn)
    assert len(ids) == 2
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)


def test_load_embeddings_on_empty_database(conn):
    ids, matrix = db.load_embeddings(conn)
    assert ids == []
    assert matrix.shape == (0, 0)


def test_fetch_chunks_preserves_requested_order(conn):
    add_document(conn, chunks=[("A", "first"), ("B", "second")])
    ids, _ = db.load_embeddings(conn)
    reversed_ids = list(reversed(ids))
    chunks = db.fetch_chunks(conn, reversed_ids)
    assert [chunk.id for chunk in chunks] == reversed_ids
    assert chunks[0].content == "second"


def test_fetch_chunks_ignores_unknown_ids(conn):
    add_document(conn)
    assert db.fetch_chunks(conn, [9999]) == []
    assert db.fetch_chunks(conn, []) == []


def test_citation_includes_heading(conn):
    add_document(conn, chunks=[("Cancellation", "text")])
    ids, _ = db.load_embeddings(conn)
    chunk = db.fetch_chunks(conn, ids)[0]
    assert chunk.citation() == "policy.pdf, Cancellation"


def test_citation_without_heading_is_filename_only(conn):
    add_document(conn, chunks=[("", "text")])
    ids, _ = db.load_embeddings(conn)
    chunk = db.fetch_chunks(conn, ids)[0]
    assert chunk.citation() == "policy.pdf"


def test_reingesting_same_filename_replaces_chunks(conn):
    add_document(conn, chunks=[("A", "first"), ("B", "second")])
    add_document(conn, chunks=[("C", "third")])
    assert db.count_documents(conn) == 1
    assert db.count_chunks(conn) == 1


def test_document_hash_lookup(conn):
    assert db.get_document_hash(conn, "policy.pdf") is None
    add_document(conn)
    assert db.get_document_hash(conn, "policy.pdf") == "hash-policy.pdf"


def test_insert_chunks_rejects_length_mismatch(conn):
    document_id = db.replace_document(conn, "a.pdf", "a.pdf", "h")
    with pytest.raises(ValueError):
        db.insert_chunks(conn, document_id, [("A", "x")], np.zeros((2, 4), np.float32))


def test_fts_finds_exact_term(conn):
    add_document(
        conn,
        chunks=[
            ("Cancellation", "You may cancel within 14 days."),
            ("Excess", "The deductible is 500 EUR per claim."),
        ],
    )
    results = db.search_fts(conn, "deductible", limit=5)
    assert len(results) == 1
    matched = db.fetch_chunks(conn, [results[0][0]])[0]
    assert "deductible" in matched.content


def test_fts_query_sanitises_operators(conn):
    add_document(conn, chunks=[("A", "cancel within 14 days")])
    # A bare quote and the NEAR operator would be a syntax error unquoted.
    assert db.search_fts(conn, 'what " is NEAR', limit=5) == []
    assert db.search_fts(conn, "cancel", limit=5) != []


def test_fts_ignores_single_character_terms(conn):
    add_document(conn)
    assert db.to_fts_query("a b c") == ""
    assert db.to_fts_query("cancel policy") == '"cancel" OR "policy"'


def test_deleting_document_cascades_to_chunks(conn):
    add_document(conn, chunks=[("A", "first"), ("B", "second")])
    conn.execute("DELETE FROM documents")
    assert db.count_chunks(conn) == 0


def test_delete_document_removes_it_and_its_chunks(conn):
    add_document(conn, filename="alpha.md", chunks=[("A", "first"), ("B", "second")])
    add_document(conn, filename="beta.md", chunks=[("C", "third")])
    assert db.delete_document(conn, "alpha.md") is True
    assert [entry["filename"] for entry in db.list_documents(conn)] == ["beta.md"]
    assert db.count_chunks(conn) == 1


def test_delete_document_reports_when_missing(conn):
    assert db.delete_document(conn, "missing.md") is False


def test_delete_all_documents_clears_everything_and_reports_count(conn):
    add_document(conn, filename="alpha.md")
    add_document(conn, filename="beta.md")
    assert db.delete_all_documents(conn) == 2
    assert db.count_documents(conn) == 0
    assert db.count_chunks(conn) == 0


def test_delete_all_documents_on_empty_database_reports_zero(conn):
    assert db.delete_all_documents(conn) == 0


def test_connection_can_be_opened_for_cross_thread_use(tmp_path):
    """Streamlit caches a connection across reruns that happen on other threads."""
    import threading

    connection = db.connect(tmp_path / "threaded.db", same_thread=False)
    add_document(connection)
    results: list[int] = []

    def read() -> None:
        results.append(db.count_chunks(connection))

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    connection.close()

    assert results == [1], "a cached connection must be readable from another thread"


def test_default_connection_stays_thread_bound(tmp_path):
    """The default keeps SQLite's own guard, which catches accidental sharing."""
    import threading

    connection = db.connect(tmp_path / "bound.db")
    errors: list[Exception] = []

    def read() -> None:
        try:
            db.count_chunks(connection)
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    connection.close()

    assert errors, "the default must still reject use from another thread"


# --- stopword filtering, added after the hard retrieval comparison -------


def test_stopwords_are_dropped_from_the_keyword_query():
    # Filler words carry BM25 weight without carrying meaning. Leaving them in
    # made a short unrelated clause outrank the correct one.
    built = db.to_fts_query("What administration fee applies when I cancel the policy?")
    assert '"administration"' in built
    assert '"fee"' in built
    assert '"cancel"' in built
    assert '"what"' not in built
    assert '"the"' not in built
    assert '"applies"' not in built


def test_a_question_of_only_stopwords_keeps_them():
    # Searching for nothing is worse than searching for common words.
    assert db.to_fts_query("what is the") == '"what" OR "is" OR "the"'


def test_stopword_filter_is_case_insensitive():
    assert db.to_fts_query("What The Excess") == '"Excess"'


def test_numbers_and_clause_references_survive_filtering():
    # Splitting on every non alphanumeric character turned 4.2 into two single
    # digits, which were then dropped as too short. Clause references are
    # exactly what the keyword half of the search exists to match.
    built = db.to_fts_query("what is clause 4.2 about the 500 EUR excess")
    assert '"clause"' in built
    assert '"4.2"' in built
    assert '"500"' in built
    assert '"EUR"' in built
    assert '"excess"' in built


def test_amounts_with_separators_stay_whole():
    assert '"1,000"' in db.to_fts_query("what is the 1,000 EUR excess")
    assert '"29.99"' in db.to_fts_query("is the charge 29.99 per month")


def test_single_letters_are_still_dropped():
    assert db.to_fts_query("a b excess") == '"excess"' 


def test_filtered_query_still_finds_the_right_passage(conn):
    add_document(
        conn,
        chunks=[
            ("5. Cancellation", "An administration fee of 25 EUR applies on cancellation."),
            ("3. Cancellation", "An administration fee of 50 EUR applies if agreed."),
        ],
    )
    results = db.search_fts(conn, "What administration fee applies?", limit=5)
    assert len(results) == 2, "both clauses mention an administration fee"
