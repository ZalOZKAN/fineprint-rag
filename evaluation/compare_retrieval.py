"""Compare retrieval strategies without generating any answers.

Retrieval quality is a property of retrieval. Measuring it through the full
pipeline means every number carries the noise of a local model that answers the
same question differently on consecutive runs, and it costs tens of seconds per
question instead of milliseconds.

This script runs only the retrieval half, so the comparison is exact and
repeatable. It reports the same rank at two levels, because they fail
differently:

    doc hit@k    a passage from the right document reached the top k
    passage hit@k  the passage holding the answering clause reached the top k
    MRR          mean reciprocal rank of the right document

doc hit@k saturates: with a 220-passage document in the corpus, every strategy
puts the right file in the top 3 for every question. passage hit@k, matched
against each case's expect_phrase, is the one that discriminates, and with
depth = CONTEXT_CHUNKS it is the ceiling on answer accuracy.

    python evaluation/compare_retrieval.py
    python evaluation/compare_retrieval.py --golden-set evaluation/golden_set_hard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag import db, retrieval  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.retrieval import Index, Mode  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent


@dataclass
class Score:
    """Retrieval quality for one strategy over one question set.

    Two hit rates are tracked. The document hit asks whether a passage from the
    right file reached the top k. The passage hit asks whether the passage that
    actually contains the answering clause reached the top k. On a corpus with
    one 220-passage document, the first is easy and the second is the real test.
    """

    mode: str
    total: int = 0
    doc_hit_at_1: int = 0
    doc_hit_at_k: int = 0
    passage_hit_at_1: int = 0
    passage_hit_at_k: int = 0
    passage_total: int = 0
    reciprocal_ranks: float = 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_ranks / self.total if self.total else 0.0

    def rate(self, hits: int, of: int | None = None) -> float:
        denom = of if of is not None else self.total
        return hits / denom if denom else 0.0


def rank_of_source(results, expected: str) -> int | None:
    """1-based position of the first passage from the expected document."""
    for position, scored in enumerate(results, start=1):
        if expected in scored.chunk.filename:
            return position
    return None


def rank_of_phrase(results, phrase: str) -> int | None:
    """1-based position of the first passage that contains the answering clause."""
    needle = phrase.lower()
    for position, scored in enumerate(results, start=1):
        if needle in scored.chunk.content.lower():
            return position
    return None


def evaluate(
    conn, index, embedder, cases: list[dict], mode: Mode, depth: int,
    reranker=None, expand: bool = False,
) -> Score:
    label = mode.value
    if reranker is not None:
        label += "+rerank"
    if expand:
        label += "+expand"
    # Expansion is read from config inside retrieval, so it is toggled here for
    # the duration of this scoring pass and restored afterwards.
    previous = config.QUERY_EXPANSION_ENABLED
    config.QUERY_EXPANSION_ENABLED = expand
    try:
        return _score(conn, index, embedder, cases, mode, depth, reranker, label)
    finally:
        config.QUERY_EXPANSION_ENABLED = previous


def _score(
    conn, index, embedder, cases: list[dict], mode: Mode, depth: int,
    reranker, label: str,
) -> Score:
    score = Score(mode=label)
    for case in cases:
        expected = case.get("expect_source")
        if not expected:
            continue
        found = retrieval.retrieve(
            conn, index, embedder, case["question"], mode=mode,
            limit=depth, threshold=-1.0, reranker=reranker,
        )
        score.total += 1

        doc_pos = rank_of_source(found.results, expected)
        if doc_pos is not None:
            score.doc_hit_at_k += 1
            score.doc_hit_at_1 += int(doc_pos == 1)
            score.reciprocal_ranks += 1.0 / doc_pos

        phrase = case.get("expect_phrase")
        if phrase:
            score.passage_total += 1
            phrase_pos = rank_of_phrase(found.results, phrase)
            if phrase_pos is not None:
                score.passage_hit_at_k += 1
                score.passage_hit_at_1 += int(phrase_pos == 1)
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies.")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=EVALUATION_DIR / "golden_set.json",
        help="Question set to evaluate.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=config.CONTEXT_CHUNKS,
        help="How many passages count as retrieved.",
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Also score each mode with the cross-encoder reranker.",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Also score each variant with query expansion on (rag/rewrite.py).",
    )
    arguments = parser.parse_args()

    cases = json.loads(arguments.golden_set.read_text(encoding="utf-8"))["cases"]

    conn = db.connect(config.DATABASE_PATH)
    if db.count_chunks(conn) == 0:
        print("The library is empty. Run: python -m rag.ingest")
        return 1

    index = Index(conn)
    embedder = Embedder()

    print(f"{arguments.golden_set.name}: {len(cases)} cases, "
          f"{db.count_chunks(conn)} passages, depth {arguments.depth}\n")

    reranker = None
    if arguments.rerank:
        from rag.reranker import Reranker

        reranker = Reranker()

    scores: dict[str, Score] = {}
    k = arguments.depth
    try:
        print(f"  {'mode':22s}  {'doc hit@1':>10s} {'doc hit@'+str(k):>9s}"
              f" {'psg hit@1':>10s} {'psg hit@'+str(k):>9s}  {'MRR':>6s}")
        for mode in (Mode.DENSE, Mode.SPARSE, Mode.HYBRID):
            variants = [(mode, None, False)]
            if arguments.expand:
                variants.append((mode, None, True))
            if reranker is not None:
                variants.append((mode, reranker, False))
                if arguments.expand:
                    variants.append((mode, reranker, True))
            for mode_value, rr, expand in variants:
                s = evaluate(
                    conn, index, embedder, cases, mode_value, k, rr, expand
                )
                scores[s.mode] = s
                print(f"  {s.mode:22s}  "
                      f"{s.rate(s.doc_hit_at_1):9.1%} {s.rate(s.doc_hit_at_k):8.1%}"
                      f" {s.rate(s.passage_hit_at_1, s.passage_total):9.1%}"
                      f" {s.rate(s.passage_hit_at_k, s.passage_total):8.1%}"
                      f"  {s.mrr:.3f}")
    finally:
        embedder.unload()
        if reranker is not None:
            reranker.unload()
        conn.close()

    dense, hybrid = scores["dense"], scores["hybrid"]
    d_psg = dense.rate(dense.passage_hit_at_k, dense.passage_total)
    h_psg = hybrid.rate(hybrid.passage_hit_at_k, hybrid.passage_total)
    print(f"\n  hybrid minus dense:  passage hit@{k} {h_psg - d_psg:+.1%}   "
          f"MRR {hybrid.mrr - dense.mrr:+.3f}")
    if "dense+rerank" in scores:
        dr = scores["dense+rerank"]
        dr_psg = dr.rate(dr.passage_hit_at_k, dr.passage_total)
        print(f"  rerank on dense:     passage hit@{k} {dr_psg - d_psg:+.1%}   "
              f"MRR {dr.mrr - dense.mrr:+.3f}")

    if arguments.results:
        arguments.results.write_text(
            json.dumps(
                {
                    "golden_set": arguments.golden_set.name,
                    "depth": k,
                    "modes": {
                        name: {
                            "doc_hit_at_1": s.rate(s.doc_hit_at_1),
                            "doc_hit_at_k": s.rate(s.doc_hit_at_k),
                            "passage_hit_at_1": s.rate(s.passage_hit_at_1, s.passage_total),
                            "passage_hit_at_k": s.rate(s.passage_hit_at_k, s.passage_total),
                            "mrr": s.mrr,
                            "total": s.total,
                        }
                        for name, s in scores.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {arguments.results.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
