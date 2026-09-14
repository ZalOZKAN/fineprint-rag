"""Tests for the generation guards in rag/llm.py.

Small models fail in two ways this project has to survive: they ramble past any
useful length, and they fall into loops that never end. Both are cut in the
stream wrapper, so both are tested there with a fake client rather than by
hoping the real model misbehaves on cue.
"""

from __future__ import annotations

import config
from rag.llm import ChatModel, is_repeating


class FakeChunk:
    """Mimics one streaming chunk from the Foundry Local chat client."""

    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": content})})]


class FakeClient:
    """Replays a fixed list of fragments, recording how many were consumed."""

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.consumed = 0

    def complete_streaming_chat(self, messages, tools=None):
        for piece in self.pieces:
            self.consumed += 1
            yield FakeChunk(piece)


def model_with(pieces: list[str]) -> tuple[ChatModel, FakeClient]:
    """A ChatModel wired to a fake client, with loading bypassed."""
    model = ChatModel()
    client = FakeClient(pieces)
    model._client = client
    model._ensure_loaded = lambda: None  # type: ignore[method-assign]
    return model, client


# --- repetition detection ------------------------------------------------


def test_repetition_is_detected_after_enough_repeats():
    sentence = "The excess is 500 EUR for every claim under this policy. "
    padded = sentence.ljust(config.REPEAT_WINDOW)
    assert is_repeating(padded * config.REPEAT_LIMIT)


def test_varied_text_is_not_flagged_as_repeating():
    text = "".join(f"Clause {n} states a different rule entirely. " for n in range(40))
    assert not is_repeating(text)


def test_short_text_is_never_flagged():
    assert not is_repeating("short")
    assert not is_repeating("")


# --- stream guards -------------------------------------------------------


def test_normal_answer_streams_to_completion():
    model, client = model_with(["The ", "excess ", "is ", "500 EUR."])
    assert "".join(model.stream([])) == "The excess is 500 EUR."
    assert client.consumed == 4


def test_length_ceiling_cuts_a_rambling_answer():
    piece = "x" * 100
    model, client = model_with([piece] * 100)
    produced = "".join(model.stream([]))
    assert len(produced) <= config.MAX_ANSWER_CHARS + len(piece)
    assert client.consumed < 100, "the generator must stop being consumed"


def test_repetition_loop_is_cut_before_the_length_ceiling():
    # A loop short enough that the ceiling alone would not stop it quickly.
    loop = "The excess is 500 EUR. ".ljust(config.REPEAT_WINDOW)
    model, client = model_with([loop] * 200)
    produced = "".join(model.stream([]))
    assert client.consumed < 200
    assert len(produced) < config.MAX_ANSWER_CHARS


def test_time_ceiling_stops_a_generation_that_only_emits_empty_chunks(monkeypatch):
    # The observed failure: the model streams for minutes and yields no text, so
    # neither the length nor the repetition guard can fire. Time must bound it.
    monkeypatch.setattr(config, "MAX_GENERATION_SECONDS", 0)
    model, client = model_with([""] * 10_000)
    produced = "".join(model.stream([]))
    assert produced == ""
    assert client.consumed < 10, "the stream must be abandoned, not drained"


def test_empty_fragments_are_skipped():
    model, _ = model_with(["a", "", None, "b"])
    assert "".join(model.stream([])) == "ab"


def test_complete_joins_and_cleans_the_stream():
    model, _ = model_with(["Answer: ", "the ", "excess ", "is 500."])
    assert model.complete([]) == "the excess is 500."


def test_unload_is_safe_when_nothing_was_loaded():
    ChatModel().unload()
