"""Measure how well Fineprint answers, refuses and retrieves.

Unit tests prove the code does what it was written to do. They say nothing about
whether the assistant gives the right answer, which is a separate question and
the one that matters to a user. This harness answers it against a fixed set of
questions with known outcomes.

Three categories are measured separately, because they fail in different ways:

    answerable      the answer is in the corpus and must be produced, with the
                    right source cited
    not_in_corpus   the answer is absent and the assistant must say so
    out_of_scope    the question asks for advice and must be refused

Retrieval quality is measured on its own in compare_retrieval.py, at both the
document and the passage level. This harness measures what the model does with
whatever retrieval returns.

    python evaluation/run_eval.py                     the configured mode, one run
    python evaluation/run_eval.py --mode hybrid       force a different strategy
    python evaluation/run_eval.py --compare           dense and hybrid side by side
    python evaluation/run_eval.py --runs 3            repeat for reproducibility
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag import db, pipeline  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.llm import ChatModel  # noqa: E402
from rag.pipeline import Answer, Outcome  # noqa: E402
from rag.retrieval import Index, Mode  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVALUATION_DIR / "golden_set.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"

REFUSAL_OUTCOMES = {Outcome.NOT_IN_CORPUS, Outcome.ADVICE_REFUSED}

# Phrasings the model uses when it declines in its own words rather than being
# stopped by the gate. A question can be handled safely two ways: the gate
# refuses before generation, or the model reads the context and reports that the
# answer is not there. Both avoid fabrication, so both count as safe, but only
# the first is deterministic and only the first saves the generation time. The
# two are therefore measured separately.
NOT_FOUND_MARKERS = (
    "does not contain",
    "does not provide",
    "not specified",
    "not provided",
    "not mentioned",
    "cannot be determined",
    "could not find",
    "no information about",
    "is not included",
    # Added after the first run: the model also declines with phrasings the
    # original list missed. Each of these describes what the context lacks, not
    # what a policy excludes, so none of them can match a genuine answer about
    # exclusions such as "wear and tear is not covered".
    "not explicitly covered",
    "not explicitly detailed",
    "not explicitly mentioned",
    "not explicitly stated",
    "does not address",
    "do not address",
    "not addressed in",
    "outside the scope",
)


def says_not_found(text: str) -> bool:
    """Whether an answer declines in prose instead of asserting a fact.

    This is a phrase match and therefore approximate. It is used only to
    separate a safe outcome from a fabricated one, never to mark an answerable
    question as correct.
    """
    lowered = (text or "").lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


@dataclass
class CaseResult:
    """What happened for one evaluation question."""

    id: str
    category: str
    question: str
    outcome: str
    passed: bool
    retrieval_hit: bool | None
    answer_text: str
    citations: list[str]
    best_dense_score: float
    best_rerank_score: float | None
    retrieval_ms: float
    generation_ms: float
    reason: str = ""


@dataclass
class Metrics:
    """Aggregate numbers for one mode over one run."""

    mode: str
    total: int = 0
    passed: int = 0
    answerable_total: int = 0
    answerable_passed: int = 0
    refusal_total: int = 0
    refusal_passed: int = 0
    gate_total: int = 0
    gate_fired: int = 0
    retrieval_total: int = 0
    retrieval_hits: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def answer_accuracy(self) -> float:
        return (
            self.answerable_passed / self.answerable_total
            if self.answerable_total
            else 0.0
        )

    @property
    def refusal_accuracy(self) -> float:
        return self.refusal_passed / self.refusal_total if self.refusal_total else 0.0

    @property
    def gate_rate(self) -> float:
        """Share of refusable questions stopped before the model was called."""
        return self.gate_fired / self.gate_total if self.gate_total else 0.0

    @property
    def hit_rate(self) -> float:
        return self.retrieval_hits / self.retrieval_total if self.retrieval_total else 0.0

    def percentile(self, fraction: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        position = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[position]


def load_cases(path: Path | None = None) -> list[dict]:
    payload = json.loads((path or GOLDEN_SET_PATH).read_text(encoding="utf-8"))
    return payload["cases"]


def check_answerable(case: dict, answer: pipeline.Answer) -> tuple[bool, str]:
    """An answerable case passes when it was answered with the expected facts."""
    if answer.outcome is not Outcome.ANSWERED:
        return False, f"refused ({answer.outcome.value})"

    text = answer.text.lower()
    expected = [term.lower() for term in case.get("expect_contains", [])]
    # Alternative spellings of the same fact are listed together, so any match
    # counts. "1,000" and "1000" are the same number to a reader.
    if expected and not any(term in text for term in expected):
        return False, f"missing expected fact {case['expect_contains']}"

    source = case.get("expect_source")
    if source and not any(source in citation for citation in answer.citations()):
        return False, f"cited {answer.citations()} instead of {source}"

    return True, ""


def check_refusal(case: dict, answer: pipeline.Answer) -> tuple[bool, str]:
    """A refusal case passes when the assistant asserted nothing it cannot support.

    Passing means safe, not gated. An answer that reached the model and came back
    saying the context does not cover the question is still a correct outcome for
    the user, so it passes here. Whether the gate fired is tracked separately by
    gate_fired(), because that is a different property: determinism and saved
    generation time rather than safety.
    """
    if answer.outcome in REFUSAL_OUTCOMES:
        expected = (
            Outcome.ADVICE_REFUSED
            if case["category"] == "out_of_scope"
            else Outcome.NOT_IN_CORPUS
        )
        if answer.outcome is expected:
            return True, ""
        return True, f"refused as {answer.outcome.value} rather than {expected.value}"

    if says_not_found(answer.text):
        return True, "model declined in prose, the gate did not fire"

    return False, "asserted an answer it could not support"


def gate_fired(answer: pipeline.Answer) -> bool:
    """Whether a deterministic guard stopped the question before generation."""
    return answer.outcome in REFUSAL_OUTCOMES


def retrieval_hit(case: dict, answer: pipeline.Answer) -> bool | None:
    """Whether the document holding the answer reached the model context."""
    source = case.get("expect_source")
    if not source:
        return None
    return any(source in citation for citation in answer.citations())


def evaluate(
    conn, index, embedder, chat_model, cases: list[dict], mode: Mode, reranker=None
) -> tuple[Metrics, list[CaseResult]]:
    """Run every case once in the given retrieval mode."""
    metrics = Metrics(mode=mode.value)
    results: list[CaseResult] = []

    for case in cases:
        answer = pipeline.ask(
            conn, index, embedder, chat_model, case["question"], mode=mode,
            reranker=reranker,
        )

        if case["category"] == "answerable":
            passed, reason = check_answerable(case, answer)
            metrics.answerable_total += 1
            metrics.answerable_passed += int(passed)
            hit = retrieval_hit(case, answer)
            if hit is not None:
                metrics.retrieval_total += 1
                metrics.retrieval_hits += int(hit)
        else:
            passed, reason = check_refusal(case, answer)
            metrics.refusal_total += 1
            metrics.refusal_passed += int(passed)
            metrics.gate_total += 1
            metrics.gate_fired += int(gate_fired(answer))
            hit = None

        metrics.total += 1
        metrics.passed += int(passed)
        metrics.latencies_ms.append(answer.total_ms)

        results.append(
            CaseResult(
                id=case["id"],
                category=case["category"],
                question=case["question"],
                outcome=answer.outcome.value,
                passed=passed,
                retrieval_hit=hit,
                answer_text=answer.text,
                citations=answer.citations(),
                best_dense_score=round(answer.best_dense_score, 4),
                best_rerank_score=(
                    round(answer.best_rerank_score, 3)
                    if answer.best_rerank_score is not None
                    else None
                ),
                retrieval_ms=round(answer.retrieval_ms, 1),
                generation_ms=round(answer.generation_ms, 1),
                reason=reason,
            )
        )
        marker = "pass" if passed else "FAIL"
        print(f"  [{marker}] {case['id']}  {case['question'][:58]}")
        if not passed:
            print(f"         {reason}")

    return metrics, results


def print_report(metrics: Metrics) -> None:
    print(f"\n  mode                {metrics.mode}")
    print(
        f"  overall             {metrics.overall_accuracy:6.1%} "
        f"({metrics.passed}/{metrics.total})"
    )
    print(
        f"  answer accuracy     {metrics.answer_accuracy:6.1%} "
        f"({metrics.answerable_passed}/{metrics.answerable_total})"
    )
    print(
        f"  no fabrication      {metrics.refusal_accuracy:6.1%} "
        f"({metrics.refusal_passed}/{metrics.refusal_total})"
    )
    print(
        f"  gate fired first    {metrics.gate_rate:6.1%} "
        f"({metrics.gate_fired}/{metrics.gate_total})"
    )
    print(
        f"  retrieval hit@{config.CONTEXT_CHUNKS}      {metrics.hit_rate:6.1%} "
        f"({metrics.retrieval_hits}/{metrics.retrieval_total})"
    )
    print(f"  latency p50         {metrics.percentile(0.50) / 1000:6.1f} s")
    print(f"  latency p95         {metrics.percentile(0.95) / 1000:6.1f} s")


def rescore(path: Path = RESULTS_PATH) -> int:
    """Re-score saved answers under the current rules, without the model.

    Scoring rules change as the evaluation matures, and re-running 58 generations
    to see the effect of a changed marker list wastes half an hour. The stored
    answers are enough to recompute every metric except latency.
    """
    if not path.exists():
        print("No results.json to rescore. Run the evaluation first.")
        return 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    # Saved cases record what happened, not what was expected. The expectations
    # live in the golden set and have to be merged back in, otherwise every
    # check passes vacuously against absent criteria.
    expectations = {case["id"]: case for case in load_cases()}

    for mode, block in payload["modes"].items():
        metrics = Metrics(mode=mode)
        for stored in block["cases"]:
            case = {**expectations.get(stored["id"], {}), **stored}
            answer = Answer(
                question=case["question"],
                text=case["answer_text"],
                outcome=Outcome(case["outcome"]),
            )
            answer.citations = lambda cited=case["citations"]: list(cited)  # type: ignore[method-assign]

            if case["category"] == "answerable":
                passed, reason = check_answerable(case, answer)
                metrics.answerable_total += 1
                metrics.answerable_passed += int(passed)
                if case.get("expect_source"):
                    metrics.retrieval_total += 1
                    metrics.retrieval_hits += int(bool(retrieval_hit(case, answer)))
            else:
                passed, reason = check_refusal(case, answer)
                metrics.refusal_total += 1
                metrics.refusal_passed += int(passed)
                metrics.gate_total += 1
                metrics.gate_fired += int(gate_fired(answer))

            metrics.total += 1
            metrics.passed += int(passed)
            metrics.latencies_ms.append(case["retrieval_ms"] + case["generation_ms"])
            stored["passed"] = passed
            stored["reason"] = reason
            if not passed:
                print(f"  [FAIL] {stored['id']}  {reason}")

        print_report(metrics)
        block["runs"][0].update(
            {
                "overall_accuracy": metrics.overall_accuracy,
                "answer_accuracy": metrics.answer_accuracy,
                "refusal_accuracy": metrics.refusal_accuracy,
                "gate_rate": metrics.gate_rate,
                "hit_rate": metrics.hit_rate,
            }
        )

    payload["rescored"] = True
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRescored {RESULTS_PATH.name} in place")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Fineprint.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in Mode],
        default=config.RETRIEVAL_MODE,
        help="Retrieval strategy to evaluate.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run hybrid and dense and print both, for the retrieval experiment.",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Repeat to check reproducibility."
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Question set to evaluate. Use golden_set_hard.json for the "
             "retrieval questions whose vocabulary overlaps between documents.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Where to write the results, so one set does not overwrite another.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-apply the scoring rules to a saved results.json without calling "
             "the model. Use after changing a metric so the change is visible "
             "without paying for generation again.",
    )
    parser.add_argument(
        "--chat-model",
        default=config.CHAT_MODEL,
        help="Override the chat model, for the model size experiment.",
    )
    parser.add_argument(
        "--context-chunks",
        type=int,
        default=config.CONTEXT_CHUNKS,
        help="Override how many passages are sent to the model, for the "
             "context size experiment. Fewer passages generate faster.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking even when config enables it.",
    )
    expansion = parser.add_mutually_exclusive_group()
    expansion.add_argument(
        "--expand",
        action="store_true",
        help="Retrieve with the rewritten query variants (rag/rewrite.py).",
    )
    expansion.add_argument(
        "--no-expand",
        action="store_true",
        help="Retrieve with the question only, even when config enables expansion.",
    )
    arguments = parser.parse_args()

    if arguments.rescore:
        return rescore(arguments.results)

    # Applied for the whole run so retrieval.retrieve, which reads the constant
    # by default, uses the overridden value without threading a parameter
    # through every call site.
    config.CONTEXT_CHUNKS = arguments.context_chunks
    if arguments.expand or arguments.no_expand:
        config.QUERY_EXPANSION_ENABLED = arguments.expand

    conn = db.connect(config.DATABASE_PATH)
    if db.count_chunks(conn) == 0:
        print("The library is empty. Run: python -m rag.ingest")
        return 1

    cases = load_cases(arguments.golden_set)
    modes = [Mode.HYBRID, Mode.DENSE] if arguments.compare else [Mode(arguments.mode)]

    print(
        f"Evaluating {len(cases)} cases from {arguments.golden_set.name} over "
        f"{db.count_documents(conn)} documents with {arguments.chat_model}"
    )
    index = Index(conn)
    embedder = Embedder()
    chat_model = ChatModel(arguments.chat_model)
    reranker = None
    use_rerank = config.RERANK_ENABLED and not arguments.no_rerank
    if use_rerank:
        from rag.reranker import Reranker

        reranker = Reranker()

    started = time.perf_counter()
    payload: dict = {
        "chat_model": arguments.chat_model,
        "embedding_model": config.EMBEDDING_MODEL,
        "context_chunks": config.CONTEXT_CHUNKS,
        "threshold": config.RELEVANCE_THRESHOLD,
        "rerank": use_rerank,
        "query_expansion": config.QUERY_EXPANSION_ENABLED,
        "runs": arguments.runs,
        "modes": {},
    }

    try:
        for mode in modes:
            run_metrics: list[Metrics] = []
            last_results: list[CaseResult] = []
            for run in range(1, arguments.runs + 1):
                label = f"{mode.value} run {run}/{arguments.runs}"
                print(f"\n{label}")
                metrics, results = evaluate(
                    conn, index, embedder, chat_model, cases, mode, reranker
                )
                run_metrics.append(metrics)
                last_results = results
                print_report(metrics)

            accuracies = [m.overall_accuracy for m in run_metrics]
            payload["modes"][mode.value] = {
                "runs": [
                    {
                        "overall_accuracy": m.overall_accuracy,
                        "answer_accuracy": m.answer_accuracy,
                        "refusal_accuracy": m.refusal_accuracy,
                        "gate_rate": m.gate_rate,
                        "hit_rate": m.hit_rate,
                        "p50_ms": m.percentile(0.50),
                        "p95_ms": m.percentile(0.95),
                    }
                    for m in run_metrics
                ],
                "accuracy_spread": max(accuracies) - min(accuracies),
                "cases": [asdict(result) for result in last_results],
            }
            if arguments.runs > 1:
                print(
                    f"\n  reproducibility     spread {max(accuracies) - min(accuracies):.1%}"
                    f", mean {statistics.mean(accuracies):.1%}"
                )
    finally:
        embedder.unload()
        chat_model.unload()
        if reranker is not None:
            reranker.unload()
        conn.close()

    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    payload["context_chunks"] = config.CONTEXT_CHUNKS
    payload["golden_set"] = arguments.golden_set.name
    arguments.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {arguments.results}")

    if arguments.compare:
        hybrid = payload["modes"]["hybrid"]["runs"][0]
        dense = payload["modes"]["dense"]["runs"][0]
        print("\nExperiment A, retrieval strategy")
        print(f"  hit@{config.CONTEXT_CHUNKS}   dense {dense['hit_rate']:.1%}"
              f"   hybrid {hybrid['hit_rate']:.1%}")
        print(f"  overall  dense {dense['overall_accuracy']:.1%}"
              f"   hybrid {hybrid['overall_accuracy']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
