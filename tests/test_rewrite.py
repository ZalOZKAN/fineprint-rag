"""Query expansion: the rewrite is deterministic string work, so it is asserted
on directly rather than measured."""

from __future__ import annotations

import config
from rag import rewrite


def test_content_words_drop_the_interrogative_opening():
    words = rewrite.content_words(
        "Within how many hours must an employer report a workplace fatality?"
    )
    assert "hours" in words
    assert "employer" in words
    assert "fatality" in words
    # "within", "how", "many", "must", "an" carry no signal and are dropped.
    assert "how" not in words.split()
    assert "must" not in words.split()


def test_content_words_keep_numbers_and_drop_repeats():
    words = rewrite.content_words("Is the 30 day refund a 30 day refund?")
    assert words.split().count("30") == 1
    assert words.split().count("refund") == 1


def test_scope_reads_the_rule_the_question_names():
    assert (
        rewrite.scope("By when can a buyer cancel a sale under the Cooling-Off Rule?")
        == "Cooling-Off Rule"
    )
    assert (
        rewrite.scope("Under the Mail Order Rule, how must a refund be sent?")
        == "Mail Order Rule"
    )


def test_scope_falls_back_to_an_acronym():
    assert rewrite.scope("How many years must records be kept for OSHA?") == "OSHA"


def test_scope_is_empty_when_the_question_names_nothing():
    assert rewrite.scope("What is the special limit for jewellery?") == ""


def test_expansion_off_returns_only_the_question(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", False)
    question = "Under the Cooling-Off Rule, which days are not business days?"
    assert rewrite.expand(question) == [question]


def test_expansion_on_keeps_the_question_first_and_adds_variants(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", True)
    question = "Under the Cooling-Off Rule, which days are not business days?"
    variants = rewrite.expand(question)
    assert variants[0] == question
    assert 1 < len(variants) <= config.QUERY_VARIANTS
    # The scope-anchored variant repeats the rule next to the content words.
    assert any(v.startswith("Cooling-Off Rule") for v in variants[1:])


def test_expansion_never_exceeds_the_configured_variant_count(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPANSION_ENABLED", True)
    monkeypatch.setattr(config, "QUERY_VARIANTS", 2)
    variants = rewrite.expand(
        "Under the Dwelling Form, how many days after a flood loss is a proof of "
        "loss due to the insurer?"
    )
    assert len(variants) == 2


def test_expansion_of_an_empty_question_is_empty():
    assert rewrite.expand("   ") == []
