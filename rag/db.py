"""SQLite storage for documents, chunks and their embeddings.

Embeddings are stored as raw float32 bytes rather than JSON, which keeps the
database roughly four times smaller and lets NumPy read them back without
parsing. A FTS5 virtual table mirrors chunk text so that BM25 keyword search
runs in the same database as the vectors, with no extra dependency.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    filename    TEXT    NOT NULL UNIQUE,
    title       TEXT    NOT NULL,
    sha256      TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    heading     TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    embedding   BLOB,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading,
    content,
    content='chunks',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

META_EMBEDDING_DIMENSION = "embedding_dimension"
META_EMBEDDING_MODEL = "embedding_model"
META_RERANK_THRESHOLD = "auto_rerank_threshold"


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, together with the heading it appeared under."""

    id: int
    document_id: int
    chunk_index: int
    heading: str
    content: str
    filename: str
    title: str

    def citation(self) -> str:
        """Human readable source reference, for example 'policy.pdf, Cancellation'."""
        if self.heading:
            return f"{self.filename}, {self.heading}"
        return self.filename


def connect(path: Path, same_thread: bool = True) -> sqlite3.Connection:
    """Open the database, creating parent directories and the schema if needed.

    Pass same_thread=False when the connection outlives the thread that opened
    it. Streamlit needs this: it caches the connection across reruns but runs
    each rerun on a different thread, and SQLite rejects that by default. It is
    safe here because the interface only reads, while every write happens in the
    separate ingestion process.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Store a single key/value pair, overwriting any previous value."""
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Read a metadata value, or None when the key was never written."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def delete_meta(conn: sqlite3.Connection, key: str) -> None:
    """Remove a metadata value, if any. A no-op when the key was never written."""
    conn.execute("DELETE FROM meta WHERE key = ?", (key,))


def get_document_hash(conn: sqlite3.Connection, filename: str) -> str | None:
    """Return the stored content hash for a filename, or None if unseen."""
    row = conn.execute(
        "SELECT sha256 FROM documents WHERE filename = ?", (filename,)
    ).fetchone()
    return row["sha256"] if row else None


def replace_document(
    conn: sqlite3.Connection, filename: str, title: str, sha256: str
) -> int:
    """Drop any rows for this filename and insert a fresh document, returning its id."""
    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    cursor = conn.execute(
        "INSERT INTO documents(filename, title, sha256, ingested_at) "
        "VALUES (?, ?, ?, ?)",
        (filename, title, sha256, datetime.now(timezone.utc).isoformat()),
    )
    return int(cursor.lastrowid)


def insert_chunks(
    conn: sqlite3.Connection,
    document_id: int,
    chunks: Sequence[tuple[str, str]],
    embeddings: np.ndarray,
) -> None:
    """Store (heading, content) pairs together with one embedding row each."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk and embedding counts differ: {len(chunks)} vs {len(embeddings)}"
        )
    rows = [
        (document_id, index, heading, content, vector.astype(np.float32).tobytes())
        for index, ((heading, content), vector) in enumerate(zip(chunks, embeddings))
    ]
    conn.executemany(
        "INSERT INTO chunks(document_id, chunk_index, heading, content, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Resynchronise the FTS5 index with the chunks table."""
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def count_chunks(conn: sqlite3.Connection) -> int:
    """Total number of stored chunks."""
    return int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])


def count_documents(conn: sqlite3.Connection) -> int:
    """Total number of ingested documents."""
    return int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    """Every ingested document with its title, filename and passage count."""
    rows = conn.execute(
        """
        SELECT d.title, d.filename, COUNT(c.id) AS passages
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.title
        """
    ).fetchall()
    return [
        {"title": row["title"], "filename": row["filename"], "passages": int(row["passages"])}
        for row in rows
    ]


def delete_document(conn: sqlite3.Connection, filename: str) -> bool:
    """Remove one document and its chunks (cascade). False if it did not exist."""
    cursor = conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    return cursor.rowcount > 0


def delete_all_documents(conn: sqlite3.Connection) -> int:
    """Remove every document and chunk. Returns how many documents were removed."""
    removed = count_documents(conn)
    conn.execute("DELETE FROM documents")
    return removed


def load_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Return chunk ids and their embedding matrix, ordered by chunk id.

    The matrix has shape (0, 0) when the database holds no embedded chunks.
    """
    rows = conn.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [int(row["id"]) for row in rows]
    matrix = np.vstack(
        [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
    )
    return ids, matrix


def fetch_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[int]) -> list[Chunk]:
    """Load chunks joined with their document, preserving the requested order."""
    ids = list(chunk_ids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, c.chunk_index, c.heading, c.content,
               d.filename, d.title
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    by_id = {
        int(row["id"]): Chunk(
            id=int(row["id"]),
            document_id=int(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            heading=row["heading"],
            content=row["content"],
            filename=row["filename"],
            title=row["title"],
        )
        for row in rows
    }
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def search_fts(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[int, float]]:
    """Rank chunks by BM25 for a keyword query.

    SQLite returns bm25() as a negative number where smaller means a better
    match, so the sign is flipped to make a larger score mean a better match.
    """
    match = to_fts_query(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # A malformed FTS expression means no keyword matches, not a crash.
        return []
    return [(int(row["rowid"]), -float(row["score"])) for row in rows]


# Words that carry no signal in a keyword search but do carry BM25 weight.
# Leaving them in measurably hurt retrieval: the question "what administration
# fee applies when I cancel the insurance policy" matched a short unrelated
# cancellation clause first, because the filler words raised its term density
# more than they raised the correct passage's.
STOPWORDS = frozenset(
    """
    a an the this that these those and or but if then than so
    is are was were be been being am do does did done doing
    have has had having can could will would shall should may might must
    i me my we our you your he she it its they them their
    of in on at to for from with without by as into about over under
    what which who whom whose when where why how
    any some all each every both no not nor only own same very
    there here whether while during before after above below
    apply applies applied get gets got give gives given
    """.split()
)


# Words of two or more letters, or numbers that may carry separators. Splitting
# naively on every non alphanumeric character destroyed exactly the tokens this
# search exists to match: a clause reference like 4.2 became the single digits
# "4" and "2", which were then dropped as too short.
TERM_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z][A-Za-z]+")


def to_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR query built from quoted terms.

    Quoting each term neutralises FTS5 operators that would otherwise turn an
    ordinary question into a syntax error, for example a stray quote or the bare
    word NEAR. Stopwords are dropped so that BM25 ranks on the words that
    actually identify a clause, and numbers are kept whole so that amounts and
    clause references stay searchable.

    If every term is a stopword the filter is skipped, since a keyword search on
    nothing is worse than one on common words.
    """
    terms = TERM_PATTERN.findall(query or "")
    meaningful = [term for term in terms if term.lower() not in STOPWORDS]
    return " OR ".join(f'"{term}"' for term in (meaningful or terms))
