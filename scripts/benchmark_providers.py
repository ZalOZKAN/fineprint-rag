"""Measure whether registering execution providers actually changes speed.

An earlier measurement showed generation dropping from 138 s to 25 s after
running check_hardware.py --install, and that was recorded as a 5.5 times
speedup from registration. The comparison was not controlled: the 138 s run was
also the first generation ever performed on this machine, immediately after the
model was downloaded. First run effects and registration were confounded.

This script separates them. It runs the same question several times in one
process and reports each timing, so warmup is visible rather than hidden in an
average, and it can be run with or without registering first.

    python scripts/benchmark_providers.py            without registering
    python scripts/benchmark_providers.py --register register, then measure
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import foundry_local_sdk as fl  # noqa: E402

import config  # noqa: E402
from rag import db, prompts, retrieval  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.llm import ChatModel  # noqa: E402
from rag.retrieval import Index  # noqa: E402

QUESTION = "What is the excess on the home insurance policy?"
REPEATS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Time generation, with or without EPs.")
    parser.add_argument("--register", action="store_true",
                        help="Register execution providers before measuring.")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    arguments = parser.parse_args()

    fl.FoundryLocalManager.initialize(fl.Configuration(app_name=config.APP_NAME))
    manager = fl.FoundryLocalManager.instance

    before = {ep.name: ep.is_registered for ep in manager.discover_eps()}
    print(f"providers at start: {before}")

    if arguments.register:
        started = time.perf_counter()
        result = manager.download_and_register_eps()
        print(f"registered in {time.perf_counter() - started:.1f}s: "
              f"success={result.success} eps={result.registered_eps}")
        after = {ep.name: ep.is_registered for ep in manager.discover_eps()}
        print(f"providers after registering: {after}")

    conn = db.connect(config.DATABASE_PATH)
    index = Index(conn)
    embedder = Embedder()
    found = retrieval.retrieve(conn, index, embedder, QUESTION)
    messages = prompts.build_messages(QUESTION, found.context_chunks())

    chat = ChatModel()
    started = time.perf_counter()
    chat._ensure_loaded()
    print(f"\nmodel loaded in {time.perf_counter() - started:.1f}s")
    print(f"selected variant: {chat._model.id}")

    timings: list[float] = []
    for run in range(1, arguments.repeats + 1):
        started = time.perf_counter()
        text = chat.complete(messages)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        print(f"  run {run}: {elapsed:6.1f}s  {len(text):4d} chars")

    print(f"\nfirst run   {timings[0]:.1f}s")
    print(f"median      {statistics.median(timings):.1f}s")
    if len(timings) > 1:
        warm = timings[1:]
        print(f"median warm {statistics.median(warm):.1f}s "
              f"(excludes the first run)")

    embedder.unload()
    chat.unload()
    conn.close()


if __name__ == "__main__":
    main()
