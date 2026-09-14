"""Local answer generation through Foundry Local.

The model is loaded once and reused, because loading costs several seconds while
a session asks many questions. Streaming is exposed separately from the blocking
call: the interface streams so the answer appears as it is written, while the
evaluation harness takes the whole string at once.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence

import config

logger = logging.getLogger(__name__)

# Some small models open with a stray newline or repeat the prompt framing.
# "cevap:" is the Turkish counterpart: the interface chats in Turkish (see
# CLAUDE.md), and the model answers in whatever language it was asked in.
LEADING_NOISE = ("assistant:", "answer:", "cevap:")

# The exact marker rag.prompts.format_context() puts in front of each context
# passage. A small model occasionally copies one or more of these blocks
# verbatim before writing its own answer, which puts the "sources" ahead of
# the answer instead of after it.
_SOURCE_ECHO_PREFIX = re.compile(r"^\[\d+\]\s*source\s*:", re.IGNORECASE)


def _drop_echoed_context(text: str) -> str:
    """Drop leading blank-line-separated blocks that echo a context passage.

    Blocks are dropped from the front for as long as they open with the
    "[n] Source: ..." marker; the first block that does not is where the
    model's real answer starts. At least one block is always kept, so a
    reply that is nothing but echoed context is not erased entirely.
    """
    blocks = text.split("\n\n")
    start = 0
    while start < len(blocks) - 1 and _SOURCE_ECHO_PREFIX.match(blocks[start].strip()):
        start += 1
    return "\n\n".join(blocks[start:]).strip()


# Fragments between repetition checks. Scanning the answer on every fragment is
# wasteful, but with a low character ceiling a loop has to be caught quickly or
# the ceiling cuts it first and hides that it was looping.
REPEAT_CHECK_EVERY = 5


def clean_answer(text: str) -> str:
    """Trim whitespace and the framing small models sometimes emit.

    Order matters: a copied context block is dropped first, since it can
    itself be followed by a leading "Answer:"/"Cevap:" label that only
    becomes the start of the string once the block ahead of it is gone.
    """
    cleaned = _drop_echoed_context((text or "").strip())
    lowered = cleaned.lower()
    for prefix in LEADING_NOISE:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
            break
    return cleaned


def is_repeating(
    text: str,
    window: int = config.REPEAT_WINDOW,
    limit: int = config.REPEAT_LIMIT,
) -> bool:
    """Whether the tail of the answer has already been emitted several times.

    Small models fall into loops where they restate the same sentence
    indefinitely. Sampling settings reduce the odds but do not remove them, so
    the loop is detected during generation and cut deterministically. Counting
    repeats of the trailing window catches a repeat of any length that is a
    divisor of it, which covers phrase level and sentence level loops alike.
    """
    if len(text) < window * limit:
        return False
    return text.count(text[-window:]) >= limit


class ChatModel:
    """Wraps the Foundry Local chat model behind a minimal interface."""

    def __init__(self, model_alias: str | None = None) -> None:
        self.model_alias = model_alias or config.CHAT_MODEL
        self._model = None
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return

        import foundry_local_sdk as fl

        try:
            fl.FoundryLocalManager.initialize(fl.Configuration(app_name=config.APP_NAME))
        except Exception:  # noqa: BLE001 - already initialized in this process
            logger.debug("Foundry Local manager was already initialized")

        manager = fl.FoundryLocalManager.instance
        model = manager.catalog.get_model(self.model_alias)
        if model is None:
            raise RuntimeError(f"Chat model not in catalog: {self.model_alias}")
        if not model.is_cached:
            logger.info("Downloading %s", self.model_alias)
            model.download()
        model.load()
        self._model = model
        self._client = model.get_chat_client()
        self._apply_settings()
        logger.info("Chat model ready: %s", self.model_alias)

    def _apply_settings(self) -> None:
        """Push optional native generation settings onto the chat client.

        Both are None by default: max_tokens behaved as a prompt+completion cap
        on this runtime and starved the answer, and forcing temperature 0 made
        generation fail outright. Generation is bounded in stream() instead.
        """
        if config.MAX_ANSWER_TOKENS is None and config.GENERATION_TEMPERATURE is None:
            return
        try:
            if config.MAX_ANSWER_TOKENS is not None:
                self._client.settings.max_tokens = config.MAX_ANSWER_TOKENS
            if config.GENERATION_TEMPERATURE is not None:
                self._client.settings.temperature = config.GENERATION_TEMPERATURE
        except AttributeError:  # noqa: BLE001 - settings surface is runtime-defined
            logger.warning("Chat client has no settings surface; using stream guards only")

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Yield answer fragments as the model produces them.

        The stream is cut when the answer grows past the configured ceiling or
        when the model starts repeating itself, since neither condition ever
        recovers on its own and both otherwise run until the model decides to
        stop, which for a loop is never.
        """
        self._ensure_loaded()
        produced: list[str] = []
        length = 0
        since_check = 0
        chunk_count = 0
        started = time.perf_counter()

        for chunk in self._client.complete_streaming_chat(list(messages)):
            chunk_count += 1

            # A single generation is not allowed to run forever. This is the only
            # guard that fires when the model streams empty chunks for minutes,
            # since there is then no text to measure or to detect a loop in.
            if time.perf_counter() - started > config.MAX_GENERATION_SECONDS:
                logger.warning(
                    "Stopped generation at the %ds time ceiling (%d chunks, %d chars)",
                    config.MAX_GENERATION_SECONDS, chunk_count, length,
                )
                return

            piece = chunk.choices[0].delta.content
            if not piece:
                continue
            produced.append(piece)
            length += len(piece)
            yield piece

            if length >= config.MAX_ANSWER_CHARS:
                logger.info("Stopped generation at the length ceiling")
                return

            # Scanning the whole answer on every fragment is wasteful, and a
            # loop takes many fragments to establish, so check periodically.
            since_check += 1
            if since_check >= REPEAT_CHECK_EVERY:
                since_check = 0
                if is_repeating("".join(produced)):
                    logger.info("Stopped generation after detecting a repetition loop")
                    return

        # Observed during evaluation: the model occasionally streams for over two
        # minutes and emits no text at all. It is intermittent, the same question
        # answers normally on a retry, and it is invisible unless recorded, so it
        # is logged with enough detail to tell a silent model from a fast one.
        if length == 0:
            logger.warning(
                "Model produced no text: %d chunks in %.1fs",
                chunk_count,
                time.perf_counter() - started,
            )

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the whole answer as one string."""
        return clean_answer("".join(self.stream(messages)))

    def unload(self) -> None:
        """Release the model. Safe to call when nothing was ever loaded."""
        if self._model is not None:
            self._model.unload()
            self._model = None
            self._client = None
