"""Investigate generations that return no text.

Two evaluation cases spent roughly 140 seconds generating and produced zero
characters. That is not sampling variance, it is a failure mode, and it needs
the raw stream to diagnose: the model may be emitting content on a field the
wrapper does not read, or emitting nothing at all.

    python scripts/probe_empty_answer.py "How long can the property be unoccupied before cover stops?"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag import db, prompts, retrieval  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.llm import ChatModel  # noqa: E402
from rag.retrieval import Index  # noqa: E402

DEFAULT_QUESTION = "How long can the property be unoccupied before cover stops?"


def describe_delta(delta: object) -> dict:
    """Every readable field on a streaming delta, so nothing is missed."""
    return {
        name: getattr(delta, name)
        for name in dir(delta)
        if not name.startswith("_") and not callable(getattr(delta, name, None))
    }


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    conn = db.connect(config.DATABASE_PATH)
    index = Index(conn)
    embedder = Embedder()
    found = retrieval.retrieve(conn, index, embedder, question)
    messages = prompts.build_messages(question, found.context_chunks())

    print(f"question: {question}")
    print(f"context passages: {len(found.results)}")
    print(f"prompt characters: {sum(len(m['content']) for m in messages)}")

    chat = ChatModel()
    chat._ensure_loaded()

    started = time.perf_counter()
    chunk_count = 0
    content_chars = 0
    other_fields: dict[str, int] = {}
    first_shape: dict | None = None

    for chunk in chat._client.complete_streaming_chat(messages):
        chunk_count += 1
        delta = chunk.choices[0].delta
        if first_shape is None:
            first_shape = describe_delta(delta)
        fields = describe_delta(delta)
        for name, value in fields.items():
            if isinstance(value, str) and value:
                other_fields[name] = other_fields.get(name, 0) + len(value)
        if getattr(delta, "content", None):
            content_chars += len(delta.content)
        if chunk_count <= 3:
            print(f"  chunk {chunk_count}: {fields}")

    elapsed = time.perf_counter() - started
    print(f"\nchunks: {chunk_count}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"content characters: {content_chars}")
    print(f"characters seen per field: {other_fields}")
    print(f"finish reason: {getattr(chunk.choices[0], 'finish_reason', 'unknown')}")

    embedder.unload()
    chat.unload()
    conn.close()


if __name__ == "__main__":
    main()
