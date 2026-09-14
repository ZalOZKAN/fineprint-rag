"""Pick the relevance threshold from measured scores instead of by guessing.

`config.RELEVANCE_THRESHOLD` decides when Fineprint refuses to answer. Set it too
low and the assistant answers questions the corpus cannot support. Set it too
high and it refuses questions it could have answered. The initial value was a
guess, which is the wrong way to choose a number that controls a guardrail.

This script reads the per case scores that `run_eval.py` already records and
reports which threshold separates the two groups best:

    answerable      should score above the threshold
    not_in_corpus   should score below it

`out_of_scope` cases are excluded, since the advice heuristic refuses those
before retrieval runs and their scores say nothing about the gate.

    python evaluation/run_eval.py          produce results.json first
    python evaluation/calibrate_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def load_scores(
    mode: str | None = None, field: str = "best_dense_score"
) -> tuple[list[float], list[float]]:
    """Return `field` for answerable and not_in_corpus cases."""
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    available = list(payload["modes"])
    if mode is None:
        # Prefer the configured mode; otherwise the only one that was run.
        mode = config.RETRIEVAL_MODE if config.RETRIEVAL_MODE in available else available[0]
    if mode not in payload["modes"]:
        raise SystemExit(f"No results for mode {mode}. Available: {available}")

    answerable: list[float] = []
    absent: list[float] = []
    for case in payload["modes"][mode]["cases"]:
        value = case.get(field)
        if value is None:
            continue
        if case["category"] == "answerable":
            answerable.append(value)
        elif case["category"] == "not_in_corpus":
            absent.append(value)
    return answerable, absent


def score_threshold(
    threshold: float, answerable: list[float], absent: list[float]
) -> tuple[int, int, int]:
    """Return (correct, wrongly refused, wrongly answered) at this threshold."""
    wrongly_refused = sum(1 for score in answerable if score < threshold)
    wrongly_answered = sum(1 for score in absent if score >= threshold)
    total = len(answerable) + len(absent)
    return total - wrongly_refused - wrongly_answered, wrongly_refused, wrongly_answered


def main() -> int:
    if not RESULTS_PATH.exists():
        print("No results.json. Run: python evaluation/run_eval.py")
        return 1

    use_rerank = "--rerank" in sys.argv
    field = "best_rerank_score" if use_rerank else "best_dense_score"
    answerable, absent = load_scores(field=field)
    print(f"scores from mode: {config.RETRIEVAL_MODE}  field: {field}")
    if not answerable or not absent:
        print("Need both answerable and not_in_corpus cases to calibrate.")
        return 1

    print(f"answerable    n={len(answerable):2d}  "
          f"min {min(answerable):.3f}  max {max(answerable):.3f}")
    print(f"not_in_corpus n={len(absent):2d}  "
          f"min {min(absent):.3f}  max {max(absent):.3f}")

    overlap = min(answerable) <= max(absent)
    print(f"\nseparable cleanly: {'no, the two groups overlap' if overlap else 'yes'}")

    candidates = sorted({round(score, 3) for score in answerable + absent})
    best = None
    print("\n threshold   correct   refused wrongly   answered wrongly")
    for threshold in candidates:
        correct, refused, answered = score_threshold(threshold, answerable, absent)
        marker = ""
        if best is None or correct > best[1]:
            best, marker = (threshold, correct), "  <-"
        print(f"   {threshold:.3f}     {correct:3d}          {refused:3d}"
              f"                {answered:3d}{marker}")

    threshold, correct = best
    total = len(answerable) + len(absent)
    name = "RERANK_RELEVANCE_THRESHOLD" if use_rerank else "RELEVANCE_THRESHOLD"
    current = getattr(config, name)
    print(f"\nbest threshold {threshold:.3f}  ({correct}/{total} correct)")
    print(f"config.{name} is currently {current}")
    if abs(threshold - current) > 0.01:
        print(f"Consider setting {name} = {threshold:.2f} in config.py")

    if overlap:
        print(
            "\nThe groups overlap, so no threshold separates them perfectly. "
            "Some questions the corpus cannot answer look as close to it as "
            "questions it can, and no single number fixes that."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
