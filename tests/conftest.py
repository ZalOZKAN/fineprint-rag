"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from rag import db


class FakeEmbedder:
    """Deterministic stand in for the Foundry Local embedder.

    Real embeddings need a multi gigabyte model and several seconds per call, so
    the tests use a hashing scheme instead. Texts sharing vocabulary land in the
    same buckets and therefore score as similar, which is enough to exercise the
    retrieval logic.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.model_alias = "fake-embedder"
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            cleaned = "".join(char for char in word if char.isalnum())
            if cleaned:
                vector[hash(cleaned) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_texts(self, texts) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        return self._vector(text)

    def unload(self) -> None:
        return None


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture()
def corpus(tmp_path):
    """A small document folder used only as test scaffolding.

    The shipped evaluation corpus is 31 public-domain federal documents in
    evaluation/corpus/. These tmp_path files exercise the loader, chunker and
    retrieval logic in isolation, with content simple enough to assert on.
    """
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "policy.md").write_text(
        "# Home Insurance Policy\n\n"
        "This document sets out the terms of your cover in full detail so that "
        "the reader can understand what is included and what is excluded.\n\n"
        "## Cancellation\n\n"
        "You may cancel this policy within 14 days of the start date without "
        "charge. After that period a cancellation fee of 25 EUR applies to every "
        "policy regardless of the reason given for the cancellation request.\n\n"
        "## Excess\n\n"
        "The excess payable is 500 EUR for each and every claim made under this "
        "policy, and it is deducted from the settlement amount before payment.\n",
        encoding="utf-8",
    )
    (directory / "lease.md").write_text(
        "# Residential Lease Agreement\n\n"
        "This agreement is made between the landlord and the tenant named above "
        "and covers the property described in the schedule attached to it.\n\n"
        "## Notice Period\n\n"
        "Either party may terminate this lease by giving 30 days written notice "
        "to the other party at the address stated in this agreement.\n\n"
        "## Subletting\n\n"
        "The tenant may not sublet the property without the prior written "
        "consent of the landlord, which shall not be unreasonably withheld.\n",
        encoding="utf-8",
    )
    return directory
