"""Turn source documents into markdown text.

MarkItDown is the primary loader because it preserves headings, and headings are
what the chunker splits on. PyMuPDF is kept as a fallback for PDFs that
MarkItDown cannot parse or that come back suspiciously empty, which happens with
some scanned or unusually encoded files. The fallback returns plain text with no
heading structure, so those documents chunk by size instead.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Below this many characters a load is treated as failed rather than as a very
# short document, which is what an unparsable PDF typically produces.
MINIMUM_USEFUL_CHARS = 200


@dataclass(frozen=True)
class LoadedDocument:
    """A document that has been read into markdown."""

    path: Path
    title: str
    markdown: str
    sha256: str
    loader: str

    @property
    def filename(self) -> str:
        return self.path.name


def file_hash(path: Path) -> str:
    """SHA-256 of the file contents, used to skip unchanged documents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_with_markitdown(path: Path) -> str:
    """Convert a document to markdown using MarkItDown."""
    from markitdown import MarkItDown

    converter = MarkItDown()
    result = converter.convert(str(path))
    return (result.text_content or "").strip()


def load_with_pymupdf(path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    import pymupdf

    with pymupdf.open(path) as document:
        pages = [page.get_text() for page in document]
    return "\n\n".join(page.strip() for page in pages if page.strip()).strip()


def load_document(path: Path) -> LoadedDocument:
    """Read one document, falling back to PyMuPDF when MarkItDown comes up short.

    Raises ValueError when neither loader produces usable text, so that a broken
    file is reported during ingestion instead of silently becoming an empty
    document that answers every question with nothing.
    """
    if path.suffix.lower() not in config.SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name}")

    loader = "markitdown"
    try:
        markdown = load_with_markitdown(path)
    except Exception as error:  # noqa: BLE001 - any parser failure is recoverable
        logger.warning("MarkItDown failed on %s: %s", path.name, error)
        markdown = ""

    if len(markdown) < MINIMUM_USEFUL_CHARS and path.suffix.lower() == ".pdf":
        logger.info("Falling back to PyMuPDF for %s", path.name)
        try:
            fallback = load_with_pymupdf(path)
        except Exception as error:  # noqa: BLE001
            logger.warning("PyMuPDF failed on %s: %s", path.name, error)
            fallback = ""
        if len(fallback) > len(markdown):
            markdown, loader = fallback, "pymupdf"

    # The length floor only guards PDF extraction, where a near empty result
    # means the parser failed rather than that the document is short. A plain
    # text or markdown file is allowed to be genuinely brief.
    if path.suffix.lower() == ".pdf" and len(markdown) < MINIMUM_USEFUL_CHARS:
        raise ValueError(
            f"Could not extract usable text from {path.name} "
            f"({len(markdown)} characters). The file may be scanned or encrypted."
        )
    if not markdown:
        raise ValueError(f"{path.name} is empty")

    return LoadedDocument(
        path=path,
        title=derive_title(markdown, path),
        markdown=markdown,
        sha256=file_hash(path),
        loader=loader,
    )


def derive_title(markdown: str, path: Path) -> str:
    """Use the first markdown heading as the title, else the file stem.

    Blockquote front matter (a `>` licence or provenance banner) and HTML
    comments are skipped rather than treated as the first line of body, so a
    document that opens with a banner still gets its real heading as the title.
    """
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((">", "<!--")):
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
        break
    return path.stem.replace("_", " ").replace("-", " ").strip()


# Files that live in the documents folder but describe the corpus rather than
# being part of it. Matched case-insensitively on the stem.
NON_CORPUS_STEMS = {"licenses", "license", "readme", "notice", "contributing"}


def discover_documents(directory: Path) -> list[Path]:
    """List documents to ingest, sorted for stable order.

    Metadata files such as LICENSES.md sit alongside the corpus but are not part
    of it, so they are skipped.
    """
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in config.SUPPORTED_SUFFIXES
        and path.stem.lower() not in NON_CORPUS_STEMS
    )
