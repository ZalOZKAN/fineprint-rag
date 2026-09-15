"""Build the searchable index from a directory of documents.

Ingestion is idempotent: each document is stored with the SHA-256 of its file
contents, and a document whose hash has not changed is skipped entirely. That
matters because embedding is the slow part of the pipeline, and during
development the same corpus is re-ingested many times.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import config
from rag import chunking, db, loaders
from rag.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """What one ingestion run did, for printing and for tests."""

    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunk_count: int = 0

    def summary(self) -> str:
        parts = [
            f"{len(self.ingested)} ingested",
            f"{len(self.skipped)} unchanged",
            f"{self.chunk_count} chunks total",
        ]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def ingest_document(
    conn: sqlite3.Connection,
    embedder: Embedder,
    document: loaders.LoadedDocument,
) -> int:
    """Chunk, embed and store one document. Returns the number of chunks stored."""
    chunks = chunking.chunk_document(document.markdown)
    if not chunks:
        raise ValueError(f"No chunks produced from {document.filename}")

    texts = [chunking.embedding_text(heading, content) for heading, content in chunks]
    vectors = embedder.embed_texts(texts)

    document_id = db.replace_document(
        conn, document.filename, document.title, document.sha256
    )
    db.insert_chunks(conn, document_id, chunks, vectors)
    db.set_meta(conn, db.META_EMBEDDING_DIMENSION, str(vectors.shape[1]))
    db.set_meta(conn, db.META_EMBEDDING_MODEL, embedder.model_alias)
    return len(chunks)


def ingest_directory(
    conn: sqlite3.Connection,
    embedder: Embedder,
    directory: Path | None = None,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> IngestReport:
    """Ingest every supported document in a directory.

    Documents whose contents are unchanged since the last run are skipped unless
    force is set. A document that cannot be read is recorded in the report and
    does not stop the run.

    Embedding a corpus on the CPU takes minutes, long enough that a caller with
    a screen needs to say so. on_progress is called with (done, total, filename)
    before each document, so an interface can draw a progress bar without this
    module knowing anything about one.
    """
    directory = directory or config.EVAL_CORPUS_DIR
    report = IngestReport()

    paths = list(loaders.discover_documents(directory))
    for position, path in enumerate(paths):
        if on_progress is not None:
            on_progress(position, len(paths), path.name)
        try:
            current_hash = loaders.file_hash(path)
            if not force and db.get_document_hash(conn, path.name) == current_hash:
                report.skipped.append(path.name)
                continue

            document = loaders.load_document(path)
            count = ingest_document(conn, embedder, document)
            conn.commit()
            report.ingested.append(path.name)
            logger.info("Ingested %s (%d chunks, %s)", path.name, count, document.loader)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
            conn.rollback()
            report.failed.append((path.name, str(error)))
            logger.warning("Failed to ingest %s: %s", path.name, error)

    if on_progress is not None:
        on_progress(len(paths), len(paths), "")

    db.rebuild_fts(conn)
    conn.commit()
    report.chunk_count = db.count_chunks(conn)
    return report


def main() -> None:
    """Command line entry point: python -m rag.ingest [directory]"""
    import argparse

    parser = argparse.ArgumentParser(description="Index documents for Fineprint.")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=config.EVAL_CORPUS_DIR,
        help="Folder holding the documents to index.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index documents even when their contents are unchanged.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    conn = db.connect(config.DATABASE_PATH)
    embedder = Embedder()
    try:
        report = ingest_directory(conn, embedder, arguments.directory, arguments.force)
    finally:
        embedder.unload()
        conn.close()

    print(report.summary())
    for name, error in report.failed:
        print(f"  failed: {name}: {error}")


if __name__ == "__main__":
    main()
