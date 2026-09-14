# 9. Query expansion was measured and left off

Status: rejected

## Context

[ADR 8](0008-cross-encoder-reranking.md) ends on a ceiling. Reranking roughly
doubles `passage hit@1`, but `passage hit@3` does not move at all: 68.8 % on the
main set and 53.3 % on the hard set, whether the reranked pool holds 20, 40 or
60 candidates, over dense or hybrid retrieval. A reranker reorders the pool it is
handed. It cannot add the passage that the first stage never surfaced, so roughly
a third of the answering clauses are out of reach no matter how well the second
stage works.

That leaves the first stage, and the first stage is built from a single embedding
of the question exactly as the user typed it. Two properties of that phrasing
look like they should hurt:

- A question opens with an interrogative that no clause ever contains. "Within
  how many hours must an employer report a work-related fatality" shares
  "within", "how many", "must an" with nothing in the regulation.
- A question names the rule it is scoped to ("under the Cooling-Off Rule") in
  words a near neighbour also uses. One of the three recorded answer misses is
  exactly this: asked which days are not business days under the Cooling-Off
  Rule, retrieval returned Regulation E passages that define "business day" for
  a different purpose.

The standard fix is query expansion: retrieve with several phrasings of the same
question and fuse the rankings. It is what the reference production RAG kits do,
and unlike most of their features it needs no network and no second model, so it
costs nothing that this project cares about.

## Decision

Implement it, measure it, and leave it off.

`rag/rewrite.py` expands one question into at most `QUERY_VARIANTS` (3) queries:

1. the question as asked, always first
2. the same question reduced to its content words, reusing the stopword list the
   keyword search already uses, so the interrogative opening is dropped
3. the question prefixed with the rule or document it names, so the scope is
   stated once more with positional prominence

Each variant retrieves independently and every ranking is fused by the same
Reciprocal Rank Fusion that already fuses dense and sparse. Two properties are
preserved deliberately so the experiment isolates one variable:

- The **original question stays the scored one.** The relevance gate and the
  cross-encoder both read the question as asked. Calibrated thresholds keep
  meaning what they meant.
- The reranked pool is **capped back to `RERANK_CANDIDATES`.** Expansion fuses
  more rankings and would otherwise hand the cross-encoder a larger pool,
  confounding this with the pool-size sweep ADR 8 already ran. The variable under
  test is *which* candidates get reranked, not *how many*.

`QUERY_EXPANSION_ENABLED` is `False`. `compare_retrieval.py --expand` and
`run_eval.py --expand` turn it on.

## Verification

`evaluation/compare_retrieval.py --rerank --expand`, retrieval only, no
generation, so no model noise. Main set, 31 documents, 7,675 passages, depth 3:

| | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- |
| dense | 31.2 % | **68.8 %** | 0.865 |
| dense + expand | 25.0 % | 56.2 % | 0.865 |
| sparse | 43.8 % | **56.2 %** | 0.927 |
| sparse + expand | 37.5 % | 43.8 % | 0.927 |
| hybrid | 31.2 % | **62.5 %** | 0.896 |
| hybrid + expand | 37.5 % | 50.0 % | 0.896 |
| dense + rerank | 56.2 % | 68.8 % | 0.833 |
| **dense + rerank + expand** | **56.2 %** | **68.8 %** | **0.833** |
| sparse + rerank + expand | 56.2 % | 68.8 % | 0.833 |
| hybrid + rerank + expand | 56.2 % | 68.8 % | 0.833 |

Hard set, 15 colliding-deadline questions, same command:

| | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- |
| dense | 20.0 % | **53.3 %** | 0.733 |
| dense + expand | 20.0 % | 46.7 % | 0.733 |
| sparse + expand | 26.7 % | 53.3 % | 0.900 (from 0.933) |
| hybrid + expand | 20.0 % | 46.7 % | 0.900 |
| dense + rerank | 53.3 % | 53.3 % | 0.811 |
| **dense + rerank + expand** | **53.3 %** | **53.3 %** | **0.811** |
| sparse + rerank + expand | 46.7 % | 53.3 % | 0.856 |
| hybrid + rerank + expand | 46.7 % | 60.0 % (from 53.3) | 0.833 (from 0.822) |

Two results, and the second is the one that decides it.

**Without a reranker, expansion costs about 12 points of `passage hit@3`**, and
it costs the same 12 points in all three retrieval modes (68.8 to 56.2, 56.2 to
43.8, 62.5 to 50.0). That consistency is the tell: this is not noise on a
particular strategy, it is the fusion doing what it is designed to do. RRF scores
a passage by agreement across rankings. Two of the three rankings now come from
degraded queries, and a passage those two both like outranks the correct passage
that only the original found. Consensus among worse queries is consensus on the
wrong clause. `psg hit@1` moves both ways (down on dense and sparse, up on
hybrid), which is what an effect near the noise floor looks like next to a
consistent one.

**With the reranker on, which is the shipping configuration, expansion changes
nothing at all.** Not "not much": `56.2 / 68.8 / 0.833` before and after, in
every mode, to three decimals. The cross-encoder reads each question and passage
together and its judgement overrides whatever order the first stage produced, so
a differently shuffled pool of the same candidates comes out the same. This also
reproduces ADR 8's finding a third time: once reranking runs, all three retrieval
modes converge to identical numbers.

The hard set says the same thing more quietly, on a smaller sample. Without a
reranker, `psg hit@3` falls on dense and hybrid (53.3 to 46.7) and holds on
sparse. With the reranker, dense and sparse are again identical to three
decimals.

One cell disagrees and is reported rather than buried: `hybrid + rerank + expand`
scores 60.0 % against 53.3 %. On a 15-question set that is one question, 6.7
points, in a mode that is not the default, against a hard-set run-to-run variance
this project has already measured at about 3.4 points. One case moving in one of
twelve cells is what noise looks like, and it would take a larger set to claim
otherwise. It is not enough to turn on a feature that is flat or negative
everywhere else, including in the configuration that actually ships.

So the honest summary is that expansion either hurts or does nothing, depending
on whether the reranker is on, and it is on.

## Consequences

`QUERY_EXPANSION_ENABLED` stays `False`. The code stays, behind the flag and
under test, because the measurement is the deliverable: the next person to reach
for query expansion on this corpus can run one command and see the number rather
than repeat the work.

The `passage hit@3` ceiling is therefore still open, and now with one more fix
crossed off. Widening the rerank pool did not move it (ADR 8), changing the
fusion strategy did not move it (ADR 2), and rewriting the query does not move it
either. What all three have in common is that they rearrange or re-weight the
same candidate set. The remaining hypothesis is that the answering sentence is
not independently retrievable at all, because a 700-character chunk that holds it
is dominated by the other sentences around it. That points at the chunking and
the embedding model, not at the search: finer units (sentence-window retrieval),
or a retriever that encodes a long clause better. Neither is a flag, both are the
honest next experiment.

Good:

- One more plausible fix is now measured rather than assumed, and the surprise
  (exactly zero change under the default configuration) is a sharper statement
  about where the ceiling lives than the negative result on its own.
- The rewrite is deterministic string work with no model call, so the experiment
  cost microseconds per query and can be rerun on any future corpus in minutes.

Bad:

- Dead-ish code behind a flag that the default never takes. It is small, it is
  tested, and it is the evidence for this record, but it is not carrying weight
  at runtime.
- The result is corpus-specific. On a corpus of conversational questions rather
  than clause lookups, or without a cross-encoder second stage, expansion may
  well earn its place. The flag is there for that case.
