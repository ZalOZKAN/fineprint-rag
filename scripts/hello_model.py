"""Smoke test for the Foundry Local runtime and SDK.

Downloads the embedding and chat models, loads them, and exercises both clients.
Run this once after installing Foundry Local to confirm the environment works and
to discover the exact client API surface.
"""

from __future__ import annotations

import inspect
import time

import foundry_local_sdk as fl

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen3-1.7b"


def describe(obj: object, label: str) -> None:
    """Print the public callable surface of an object."""
    print(f"\n--- {label}: {type(obj).__name__} ---")
    for name, member in inspect.getmembers(obj):
        if name.startswith("_") or not callable(member):
            continue
        try:
            print(f"    {name}{inspect.signature(member)}")
        except (TypeError, ValueError):
            print(f"    {name}(...)")


def acquire(manager: fl.FoundryLocalManager, alias: str):
    """Download if needed, then load a model. Returns the model handle."""
    print(f"\n=== {alias} ===")
    model = manager.catalog.get_model(alias)
    if model is None:
        raise SystemExit(f"Model not found in catalog: {alias}")
    print(f"cached={model.is_cached}")
    if not model.is_cached:
        model.download(lambda p: print(f"\r  downloading {p:.1f}%", end="", flush=True))
        print()
    started = time.perf_counter()
    model.load()
    print(f"loaded in {time.perf_counter() - started:.1f}s")
    return model


def main() -> None:
    fl.FoundryLocalManager.initialize(fl.Configuration(app_name="fineprint_hello"))
    manager = fl.FoundryLocalManager.instance

    emb_model = acquire(manager, EMBEDDING_MODEL)
    emb_client = emb_model.get_embedding_client()
    describe(emb_client, "embedding client")

    chat_model = acquire(manager, CHAT_MODEL)
    chat_client = chat_model.get_chat_client()
    describe(chat_client, "chat client")

    print("\n=== embedding probe ===")
    started = time.perf_counter()
    response = emb_client.generate_embeddings(["hello world", "insurance deductible"])
    vectors = [item.embedding for item in response.data]
    print(f"count={len(vectors)} dimension={len(vectors[0])} "
          f"elapsed={time.perf_counter() - started:.2f}s")

    print("\n=== chat probe ===")
    messages = [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What is a deductible in an insurance policy?"},
    ]
    started = time.perf_counter()
    pieces: list[str] = []
    for chunk in chat_client.complete_streaming_chat(messages):
        piece = chunk.choices[0].delta.content
        if piece:
            pieces.append(piece)
    answer = "".join(pieces)
    print(f"elapsed={time.perf_counter() - started:.2f}s")
    print(f"answer: {answer}")
    print(f"THINKING TAGS PRESENT: {'<think' in answer.lower()}")

    emb_model.unload()
    chat_model.unload()
    print("\nOK")


if __name__ == "__main__":
    main()
