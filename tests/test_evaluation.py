"""Tests for the evaluation harness scoring rules.

The harness decides whether an answer counts as correct. If that logic is wrong
every number it reports is wrong too, so it is tested like any other module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

import run_eval  # noqa: E402

from rag.pipeline import Answer, Outcome  # noqa: E402


def make_answer(text="", outcome=Outcome.ANSWERED, citations=()):
    """An Answer with the citation list stubbed, since sources need a database."""
    answer = Answer(question="q", text=text, outcome=outcome)
    answer.citations = lambda: list(citations)  # type: ignore[method-assign]
    return answer


# --- golden set ----------------------------------------------------------


def test_golden_set_is_valid_and_covers_all_categories():
    cases = run_eval.load_cases()
    categories = {case["category"] for case in cases}
    assert categories == {"answerable", "not_in_corpus", "out_of_scope"}
    assert len(cases) >= 20


def test_golden_set_ids_are_unique():
    cases = run_eval.load_cases()
    identifiers = [case["id"] for case in cases]
    assert len(identifiers) == len(set(identifiers))


def test_answerable_cases_declare_expectations():
    for case in run_eval.load_cases():
        if case["category"] != "answerable":
            continue
        assert case.get("expect_source"), f"{case['id']} has no expected source"
        assert case.get("expect_contains"), f"{case['id']} has no expected fact"


def test_refusal_cases_declare_no_expectations():
    for case in run_eval.load_cases():
        if case["category"] == "answerable":
            continue
        assert "expect_source" not in case
        assert "expect_contains" not in case


# --- answerable scoring --------------------------------------------------


def test_answerable_passes_with_fact_and_source():
    case = {"expect_contains": ["500"], "expect_source": "policy.md"}
    answer = make_answer("The excess is 500 EUR.", citations=["policy.md, Excess"])
    passed, reason = run_eval.check_answerable(case, answer)
    assert passed
    assert reason == ""


def test_answerable_accepts_any_listed_spelling():
    case = {"expect_contains": ["1,000", "1000"], "expect_source": "policy.md"}
    answer = make_answer("It is 1000 EUR.", citations=["policy.md, Excess"])
    assert run_eval.check_answerable(case, answer)[0]


def test_answerable_fails_when_the_fact_is_missing():
    case = {"expect_contains": ["500"], "expect_source": "policy.md"}
    answer = make_answer("There is an excess.", citations=["policy.md, Excess"])
    passed, reason = run_eval.check_answerable(case, answer)
    assert not passed
    assert "missing expected fact" in reason


def test_answerable_fails_when_the_wrong_document_is_cited():
    case = {"expect_contains": ["500"], "expect_source": "policy.md"}
    answer = make_answer("The excess is 500 EUR.", citations=["lease.md, Rent"])
    passed, reason = run_eval.check_answerable(case, answer)
    assert not passed
    assert "instead of" in reason


def test_answerable_fails_when_the_assistant_refused():
    case = {"expect_contains": ["500"], "expect_source": "policy.md"}
    answer = make_answer("I could not find this.", outcome=Outcome.NOT_IN_CORPUS)
    passed, reason = run_eval.check_answerable(case, answer)
    assert not passed
    assert "refused" in reason


def test_fact_matching_is_case_insensitive():
    case = {"expect_contains": ["Consent"], "expect_source": "lease.md"}
    answer = make_answer("written consent is required", citations=["lease.md, Sublet"])
    assert run_eval.check_answerable(case, answer)[0]


# --- refusal scoring -----------------------------------------------------


def test_out_of_scope_passes_when_refused_as_advice():
    case = {"category": "out_of_scope"}
    answer = make_answer(outcome=Outcome.ADVICE_REFUSED)
    assert run_eval.check_refusal(case, answer)[0]


def test_not_in_corpus_passes_when_gated():
    case = {"category": "not_in_corpus"}
    answer = make_answer(outcome=Outcome.NOT_IN_CORPUS)
    assert run_eval.check_refusal(case, answer)[0]


def test_refusing_for_the_other_reason_still_counts_but_is_reported():
    case = {"category": "out_of_scope"}
    answer = make_answer(outcome=Outcome.NOT_IN_CORPUS)
    passed, reason = run_eval.check_refusal(case, answer)
    assert passed
    assert "rather than" in reason


def test_answering_a_question_that_should_be_refused_fails():
    case = {"category": "not_in_corpus"}
    answer = make_answer("Sure, here is an answer.", outcome=Outcome.ANSWERED)
    passed, reason = run_eval.check_refusal(case, answer)
    assert not passed
    assert "could not support" in reason


# --- retrieval hit -------------------------------------------------------


def test_retrieval_hit_is_true_when_the_source_is_cited():
    case = {"expect_source": "policy.md"}
    answer = make_answer(citations=["policy.md, Excess", "lease.md, Rent"])
    assert run_eval.retrieval_hit(case, answer) is True


def test_retrieval_hit_is_false_when_the_source_is_absent():
    case = {"expect_source": "policy.md"}
    answer = make_answer(citations=["lease.md, Rent"])
    assert run_eval.retrieval_hit(case, answer) is False


def test_retrieval_hit_is_undefined_without_an_expected_source():
    assert run_eval.retrieval_hit({}, make_answer()) is None


# --- metrics -------------------------------------------------------------


def test_metrics_on_empty_input_do_not_divide_by_zero():
    metrics = run_eval.Metrics(mode="hybrid")
    assert metrics.overall_accuracy == 0.0
    assert metrics.answer_accuracy == 0.0
    assert metrics.refusal_accuracy == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.percentile(0.5) == 0.0


def test_metrics_compute_expected_rates():
    metrics = run_eval.Metrics(
        mode="hybrid",
        total=10,
        passed=8,
        answerable_total=6,
        answerable_passed=5,
        refusal_total=4,
        refusal_passed=3,
        retrieval_total=6,
        retrieval_hits=6,
        latencies_ms=[100.0, 200.0, 300.0, 400.0],
    )
    assert metrics.overall_accuracy == pytest.approx(0.8)
    assert metrics.answer_accuracy == pytest.approx(5 / 6)
    assert metrics.refusal_accuracy == pytest.approx(0.75)
    assert metrics.hit_rate == pytest.approx(1.0)
    assert metrics.percentile(0.5) == 300.0
    assert metrics.percentile(0.95) == 400.0


# --- safety against gate, added after the first evaluation run -----------


def test_prose_refusal_is_recognised():
    assert run_eval.says_not_found("The context does not contain that information.")
    assert run_eval.says_not_found("That is not specified in the provided context.")
    assert run_eval.says_not_found("The answer cannot be determined from the context.")


def test_a_real_answer_is_not_read_as_a_refusal():
    assert not run_eval.says_not_found("The excess is 500 EUR for every claim.")
    assert not run_eval.says_not_found("")


def test_model_declining_in_prose_counts_as_safe():
    # The first evaluation run showed the model declining correctly in its own
    # words on every not_in_corpus question. Counting those as failures measured
    # the gate rather than safety.
    case = {"category": "not_in_corpus"}
    answer = make_answer(
        "The context provided does not contain information about mortgages.",
        outcome=Outcome.ANSWERED,
    )
    passed, reason = run_eval.check_refusal(case, answer)
    assert passed
    assert "gate did not fire" in reason


def test_fabricated_answer_still_fails():
    case = {"category": "not_in_corpus"}
    answer = make_answer("Your mortgage rate is 4.2 percent.", outcome=Outcome.ANSWERED)
    passed, reason = run_eval.check_refusal(case, answer)
    assert not passed
    assert "could not support" in reason


def test_gate_fired_distinguishes_the_two_paths():
    assert run_eval.gate_fired(make_answer(outcome=Outcome.NOT_IN_CORPUS))
    assert run_eval.gate_fired(make_answer(outcome=Outcome.ADVICE_REFUSED))
    assert not run_eval.gate_fired(
        make_answer("does not contain", outcome=Outcome.ANSWERED)
    )


def test_gate_rate_is_reported_separately_from_safety():
    metrics = run_eval.Metrics(
        mode="hybrid", refusal_total=8, refusal_passed=8, gate_total=8, gate_fired=3
    )
    assert metrics.refusal_accuracy == 1.0
    assert metrics.gate_rate == pytest.approx(3 / 8)


def test_expanded_markers_catch_the_phrasing_the_first_list_missed():
    # The first run had one case where the model declined with "not explicitly
    # covered", which the original marker list did not match.
    assert run_eval.says_not_found("The policy is not explicitly covered here.")
    assert run_eval.says_not_found("These documents do not address flight bookings.")


def test_markers_do_not_match_a_genuine_exclusion_answer():
    # "not covered" alone would wrongly match a real answer about exclusions,
    # which is why the markers describe the context rather than the policy.
    assert not run_eval.says_not_found("Damage caused by wear and tear is not covered.")
    assert not run_eval.says_not_found("Consumable parts are not covered by warranty.")
