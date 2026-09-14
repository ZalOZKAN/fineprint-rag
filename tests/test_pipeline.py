"""Tests for prompts, guards and the end to end pipeline."""

from __future__ import annotations

import json

import pytest

import config
from rag import ingest, pipeline, prompts
from rag.llm import clean_answer
from rag.pipeline import Outcome
from rag.retrieval import Index, Mode


class FakeChatModel:
    """Records the prompts it receives and replays a canned answer."""

    def __init__(self, reply: str = "The excess is 500 EUR, under Excess.") -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        self.messages = list(messages)
        for word in self.reply.split(" "):
            yield word + " "

    def complete(self, messages) -> str:
        return clean_answer("".join(self.stream(messages)))


@pytest.fixture()
def ready(conn, embedder, corpus):
    ingest.ingest_directory(conn, embedder, corpus)
    return conn, Index(conn), embedder, FakeChatModel()


# --- advice heuristic ----------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Should I cancel this policy?",
        "Can I sue my landlord?",
        "What would you do in my situation?",
        "Do you recommend accepting this offer?",
        "Is it legal to sublet without consent?",
        "I want your advice on this clause",
    ],
)
def test_advice_requests_are_detected(question):
    assert prompts.is_advice_request(question)


@pytest.mark.parametrize(
    "question",
    [
        # Added after the real-corpus eval let two advice questions through.
        "Is it worth cancelling this contract within the cooling-off period?",
        "Is the flood policy's deductible fair compared with private insurers?",
        "Is this a scam?",
        "Is that a good deal?",
    ],
)
def test_broadened_advice_phrasings_are_detected(question):
    assert prompts.is_advice_request(question)


@pytest.mark.parametrize(
    "question",
    [
        # "fair market value" and "fair proportion" are legal terms, not judgement.
        "How is the fair market value of the property determined?",
        "What does the fair proportion clause say?",
        "Is the proof of loss deadline 60 days?",
    ],
)
def test_legal_terms_are_not_mistaken_for_advice(question):
    assert not prompts.is_advice_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the excess?",
        "How much notice must I give?",
        "When can I cancel without a fee?",
        "Does the policy mention subletting?",
    ],
)
def test_factual_questions_are_not_advice(question):
    assert not prompts.is_advice_request(question)


def test_advice_detection_handles_empty_input():
    assert not prompts.is_advice_request("")


# --- prompt construction -------------------------------------------------


def test_context_block_numbers_and_cites_each_passage(ready):
    conn, index, embedder, _ = ready
    from rag import retrieval

    found = retrieval.retrieve(conn, index, embedder, "excess")
    block = prompts.format_context(found.context_chunks())
    assert "[1] Source:" in block
    assert any(chunk.citation() in block for chunk in found.context_chunks())


def test_messages_carry_the_system_rules_and_question(ready):
    conn, index, embedder, _ = ready
    from rag import retrieval

    found = retrieval.retrieve(conn, index, embedder, "excess")
    messages = prompts.build_messages("What is the excess?", found.context_chunks())
    assert messages[0]["role"] == "system"
    assert "only the context" in messages[0]["content"]
    assert "do not give legal" in messages[0]["content"].lower()
    assert messages[1] == {"role": "user", "content": "What is the excess?"}


def test_clean_answer_strips_model_prefixes():
    assert clean_answer("  Answer: the excess is 500  ") == "the excess is 500"
    assert clean_answer("Assistant: hello") == "hello"
    assert clean_answer("Cevap: 500 EUR") == "500 EUR"
    assert clean_answer(None) == ""


def test_clean_answer_drops_a_single_echoed_context_block():
    # A small model sometimes copies the "[n] Source: ..." block it was given
    # as context before writing its own answer, putting the sources first.
    text = (
        "[2] Source: policy.pdf, Cancellation\nYou may cancel within 14 days.\n\n"
        "You may cancel within 14 days of signing."
    )
    assert clean_answer(text) == "You may cancel within 14 days of signing."


def test_clean_answer_drops_several_echoed_context_blocks():
    text = (
        "[1] Source: a.pdf, A\nfirst passage\n\n"
        "[2] Source: b.pdf, B\nsecond passage\n\n"
        "Answer: the real answer"
    )
    assert clean_answer(text) == "the real answer"


def test_clean_answer_keeps_the_last_block_even_if_it_looks_like_context():
    # Never erase a reply down to nothing just because it echoes context.
    text = "[1] Source: a.pdf, A\nonly this"
    assert clean_answer(text) == "[1] Source: a.pdf, A\nonly this"


def test_clean_answer_leaves_ordinary_answers_untouched():
    assert clean_answer("You may cancel within 14 days.") == (
        "You may cancel within 14 days."
    )


# --- pipeline ------------------------------------------------------------


def test_grounded_question_is_answered(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.ANSWERED
    assert answer.answered
    assert answer.text
    assert answer.sources
    assert chat.calls == 1


def test_answer_reports_citations_without_duplicates(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                          threshold=-1.0)
    citations = answer.citations()
    assert citations
    assert len(citations) == len(set(citations))


def test_advice_question_never_reaches_the_model(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "Should I cancel this policy?")
    assert answer.outcome is Outcome.ADVICE_REFUSED
    assert not answer.answered
    assert chat.calls == 0, "the model must not be called for advice requests"
    assert answer.text == prompts.ADVICE_MESSAGE


def test_irrelevant_question_never_reaches_the_model(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "Who won the 1998 World Cup?",
                          threshold=2.0)
    assert answer.outcome is Outcome.NOT_IN_CORPUS
    assert chat.calls == 0, "the gate must run before generation"
    assert answer.sources == []


def test_empty_question_is_refused(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "   ")
    assert answer.outcome is Outcome.EMPTY_QUESTION
    assert chat.calls == 0


def test_timings_are_recorded(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                          threshold=-1.0)
    assert answer.retrieval_ms > 0
    assert answer.generation_ms > 0
    assert answer.total_ms == pytest.approx(
        answer.retrieval_ms + answer.generation_ms
    )


def test_mode_is_passed_through_to_retrieval(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "excess", mode=Mode.DENSE,
                          threshold=-1.0)
    assert answer.mode is Mode.DENSE
    assert all(scored.sparse_rank is None for scored in answer.sources)


def test_streaming_yields_pieces_and_fills_the_answer(ready):
    conn, index, embedder, chat = ready
    answer, stream = pipeline.ask_streaming(
        conn, index, embedder, chat, "What is the excess?", threshold=-1.0
    )
    pieces = list(stream)
    assert len(pieces) > 1
    assert answer.text == chat.reply
    assert answer.generation_ms > 0


def test_streaming_refusal_yields_the_refusal_text(ready):
    conn, index, embedder, chat = ready
    answer, stream = pipeline.ask_streaming(
        conn, index, embedder, chat, "Should I sue?"
    )
    assert list(stream) == [prompts.ADVICE_MESSAGE]
    assert answer.outcome is Outcome.ADVICE_REFUSED
    assert chat.calls == 0


def test_query_log_is_written_as_jsonl(ready, tmp_path):
    conn, index, embedder, chat = ready
    log_path = tmp_path / "queries.jsonl"
    pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                 threshold=-1.0, log_path=log_path)
    pipeline.ask(conn, index, embedder, chat, "Should I cancel?", log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["outcome"] == "answered"
    assert first["chunk_ids"]
    assert json.loads(lines[1])["outcome"] == "advice_refused"


def test_logging_failure_does_not_break_a_query(ready, tmp_path):
    conn, index, embedder, chat = ready
    # A directory where the log file should be makes the write fail.
    blocked = tmp_path / "queries.jsonl"
    blocked.mkdir()
    answer = pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                          threshold=-1.0, log_path=blocked)
    assert answer.answered


# --- empty generation, observed during evaluation ------------------------


def test_empty_model_output_is_reported_not_presented_as_an_answer(ready):
    # Two evaluation cases spent over two minutes generating and returned zero
    # characters. Passing an empty string through as an answer would hide that.
    conn, index, embedder, _ = ready
    silent = FakeChatModel(reply="")
    answer = pipeline.ask(conn, index, embedder, silent, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.GENERATION_FAILED
    assert not answer.answered
    assert answer.text == prompts.GENERATION_FAILED_MESSAGE


def test_whitespace_only_output_counts_as_a_failed_generation(ready):
    conn, index, embedder, _ = ready
    silent = FakeChatModel(reply="   \n  ")
    answer = pipeline.ask(conn, index, embedder, silent, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.GENERATION_FAILED


def test_streaming_also_reports_a_failed_generation(ready):
    conn, index, embedder, _ = ready
    silent = FakeChatModel(reply="")
    answer, stream = pipeline.ask_streaming(
        conn, index, embedder, silent, "What is the excess?", threshold=-1.0
    )
    list(stream)
    assert answer.outcome is Outcome.GENERATION_FAILED
    assert answer.text == prompts.GENERATION_FAILED_MESSAGE


def test_failed_generation_keeps_its_sources_for_debugging(ready):
    conn, index, embedder, _ = ready
    silent = FakeChatModel(reply="")
    answer = pipeline.ask(conn, index, embedder, silent, "What is the excess?",
                          threshold=-1.0)
    assert answer.sources, "the retrieved passages explain what the model was given"


# --- one retry on empty generation, follow-up to the earlier finding -----


class FlakyChatModel:
    """Returns empty on the first call, then a real answer."""

    def __init__(self, real="The excess is 500 EUR, under Excess."):
        self.real = real
        self.calls = 0

    def _reply(self):
        self.calls += 1
        return "" if self.calls == 1 else self.real

    def stream(self, messages):
        for word in self._reply().split(" "):
            if word:
                yield word + " "

    def complete(self, messages):
        return clean_answer("".join(self.stream(messages)))


def test_empty_generation_is_retried_once_and_recovers(ready):
    conn, index, embedder, _ = ready
    flaky = FlakyChatModel()
    answer = pipeline.ask(conn, index, embedder, flaky, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.ANSWERED
    assert answer.generation_retried is True
    assert flaky.calls == 2


def test_two_empty_generations_in_a_row_still_report_failure(ready):
    conn, index, embedder, _ = ready
    always_empty = FakeChatModel(reply="")
    answer = pipeline.ask(conn, index, embedder, always_empty, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.GENERATION_FAILED
    assert answer.generation_retried is True
    assert always_empty.calls == 2, "exactly one retry, not a loop"


def test_a_normal_answer_is_not_retried(ready):
    conn, index, embedder, chat = ready
    answer = pipeline.ask(conn, index, embedder, chat, "What is the excess?",
                          threshold=-1.0)
    assert answer.generation_retried is False
    assert chat.calls == 1


def test_streaming_retries_once_on_empty(ready):
    conn, index, embedder, _ = ready
    flaky = FlakyChatModel()
    answer, stream = pipeline.ask_streaming(
        conn, index, embedder, flaky, "What is the excess?", threshold=-1.0
    )
    list(stream)
    assert answer.outcome is Outcome.ANSWERED
    assert answer.generation_retried is True
    assert flaky.calls == 2


def test_retry_can_be_disabled_by_config(ready, monkeypatch):
    monkeypatch.setattr(config, "RETRY_ON_EMPTY_GENERATION", False)
    conn, index, embedder, _ = ready
    always_empty = FakeChatModel(reply="")
    answer = pipeline.ask(conn, index, embedder, always_empty, "What is the excess?",
                          threshold=-1.0)
    assert answer.outcome is Outcome.GENERATION_FAILED
    assert answer.generation_retried is False
    assert always_empty.calls == 1
