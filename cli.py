"""Command line interface for Fineprint.

Usage:
    python cli.py                      interactive session
    python cli.py "what is the excess" answer one question and exit
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
from rag import db, pipeline
from rag.embeddings import Embedder
from rag.llm import ChatModel
from rag.retrieval import Index, Mode


def render_footer(answer: pipeline.Answer) -> None:
    """Print the sources and timings that follow an answer."""
    if answer.sources:
        print("\nSources:")
        for scored in answer.sources:
            print(
                f"  {scored.chunk.citation()}"
                f"   (similarity {scored.dense_score:.2f})"
            )
    print(
        f"[{answer.outcome.value}] "
        f"retrieval {answer.retrieval_ms:.0f} ms, "
        f"generation {answer.generation_ms:.0f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in Mode],
        default=config.RETRIEVAL_MODE,
        help="Retrieval strategy. Defaults to config.RETRIEVAL_MODE "
             f"({config.RETRIEVAL_MODE}); see docs/adr/0002.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.RELEVANCE_THRESHOLD,
        help="Minimum similarity before the assistant will answer at all.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="After the answer, trace each sentence back to the passage it came "
             "from, with the retrieval scores.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking even when config enables it.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show library logs.")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(message)s",
    )

    conn = db.connect(config.DATABASE_PATH)
    if db.count_chunks(conn) == 0:
        print(
            "The library is empty. Add documents to "
            f"{config.EVAL_CORPUS_DIR} and run: python -m rag.ingest"
        )
        return 1

    print(
        f"Fineprint: {db.count_documents(conn)} documents, "
        f"{db.count_chunks(conn)} passages. Loading models..."
    )
    index = Index(conn)
    embedder = Embedder()
    chat_model = ChatModel()
    mode = Mode(arguments.mode)
    reranker = None
    if config.RERANK_ENABLED and not arguments.no_rerank:
        from rag.reranker import Reranker

        reranker = Reranker()

    def answer_one(question: str) -> None:
        answer, stream = pipeline.ask_streaming(
            conn, index, embedder, chat_model, question,
            mode=mode, threshold=arguments.threshold, reranker=reranker,
        )
        for piece in stream:
            print(piece, end="", flush=True)
        print()
        render_footer(answer)
        if arguments.audit and answer.answered and answer.sources:
            from rag.audit import audit_answer, format_audit

            print("\nHow this answer was built:")
            print(format_audit(audit_answer(answer.text, answer.sources, embedder)))
        pipeline.log_answer(answer, config.QUERY_LOG_PATH)

    try:
        if arguments.question:
            answer_one(arguments.question)
            return 0

        print("Type a question, or 'quit' to exit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                break
            answer_one(question)
            print()
    finally:
        embedder.unload()
        chat_model.unload()
        if reranker is not None:
            reranker.unload()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
