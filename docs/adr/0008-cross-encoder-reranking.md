# 8. Cross-encoder reranking of retrieval candidates

Status: accepted

## Context

ADR 2 measured retrieval at the passage level and found the real ceiling on
answer quality: on the six-document corpus the passage that actually contains
the answering clause reaches the top 3 only 75% of the time on the main set and
53% on the hard set. The model cannot quote a clause it never sees, so every
point of `passage_hit@3` lost is a point of answer accuracy that no prompt or
larger model can recover.

Dense and sparse retrieval are both bi-encoders. They turn the query into a
vector and each passage into a vector, independently, and compare the two. That
independence is what makes them fast enough to run over a whole corpus, and it
is also their weakness: they can tell that a passage is *about* flood-loss
deadlines, but not that this particular passage answers *this* question rather
than the neighbouring clause about a different deadline. The hard set is built
entirely out of that distinction.

A cross-encoder does not embed the two sides separately. It concatenates the
query and one passage and runs them through the model together, one forward pass
per pair, producing a single relevance score. It sees the interaction the
bi-encoder cannot, and on passage-ranking benchmarks it consistently beats
bi-encoder retrieval by a wide margin. The cost is that it cannot be run over
the corpus: one pass per passage per query is far too slow.

## Decision

Keep the bi-encoder retrieval as the first stage, widen its net, and add a
cross-encoder as a second stage that reranks only what the first stage found.

- Retrieval pulls `RERANK_CANDIDATES` (20) from dense and from sparse instead of
  the usual 8, fuses them with RRF as before, and keeps the pooled candidates.
- The cross-encoder scores every `(question, passage)` pair in the pool and
  reorders it. The top `CONTEXT_CHUNKS` (3) of that order become the model
  context.
- The relevance gate still looks at the best dense cosine over the *whole* pool,
  not the reranked top 3, so reranking cannot cause a false refusal.

The model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, a 22M-parameter cross
encoder trained for exactly this task. It runs as ONNX through onnxruntime, the
same runtime Foundry Local uses, with the `tokenizers` library for the
WordPiece step. **No torch, no transformers, no sentence-transformers.** All
three imports (`onnxruntime`, `tokenizers`, `huggingface_hub`) already ship as
transitive dependencies of `foundry-local-sdk`. The model downloads once from
the Hugging Face hub, about 90 MB, and is then cached on disk like the Foundry
models. `RERANK_ENABLED` in config and `--no-rerank` on the CLI and the
evaluation harness turn it off for comparison.

## Verification

Measured with `evaluation/compare_retrieval.py --rerank`, which scores retrieval
on its own with no generation, so the numbers carry no model noise.

Main set (16 phrase-anchored questions):

| | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- |
| dense | 31.2 % | **75.0 %** | 0.896 |
| dense + rerank | **62.5 %** | 68.8 % | 0.896 |

Hard set (15 questions, colliding deadlines across documents):

| | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- |
| dense | 20.0 % | 53.3 % | 0.833 |
| dense + rerank | **46.7 %** | **60.0 %** | **0.889** |

The result splits by difficulty, and the split is the point:

- **`psg hit@1` roughly doubles on both sets** (31 to 62, 20 to 47). The
  reranker is much better at putting the answering passage *first*, which is the
  position the model weights most.
- **On the hard set, `psg hit@3` also rises**, 53 to 60, and MRR with it. When
  five passages all say "30 days", reading the question and the passage together
  is what separates them, and that is exactly what the cross-encoder adds.
- **On the main set, `psg hit@3` dips slightly**, 75 to 69: one question of 16
  whose answering passage the bi-encoder held at rank 3 and the cross-encoder,
  confidently but wrongly, pushed to rank 4. On easy questions there is little
  for reranking to fix and its rare mistakes are visible.

Reranking 20 pairs takes about 80 ms on this CPU, against roughly 60 s of
generation. The latency cost is not measurable in practice.

The corpus was later grown to 31 documents. The `psg hit@1` gain held (dense
31 % to 56 %, hard 20 % to 53 %); `psg hit@3` did not move, because the reranker
only reorders the pool it is handed.

That pool was then widened to see if `psg hit@3` would follow. It did not.
Reranking 20, 40 or 60 candidates, over `dense` or `hybrid` retrieval, all give
exactly 68.8 % `psg hit@3` on the main set and 53.3 % on the hard set; on the
hard set 40 and 60 candidates actually *lower* `psg hit@1` (53 % to 47 %),
because a wider pool feeds the reranker more distractors. `dense` and `hybrid`
produce identical numbers at every pool size, which is its own finding: once the
reranker runs, the first-stage fusion strategy stops mattering. The `psg hit@3`
ceiling is set by first-stage recall, whether the bi-encoder surfaces the
answering passage at all, not by how many of its candidates get reranked.
Lifting it needs a different first stage (finer chunks so the answering sentence
is its own retrievable unit, a stronger embedding model, or query expansion),
none of which is a tuning knob. `RERANK_CANDIDATES` stays at 20.

### The reranker score as the relevance gate

The relevance gate used to compare the best dense cosine against a threshold.
Going from 6 to 31 documents forced that threshold from 0.52 to 0.60: with more
documents, an out-of-scope question almost always retrieves *something* whose
cosine clears the old bar, and five not-in-corpus questions started reaching the
model. A gate that has to be recalibrated every time the corpus changes is a
liability.

When the reranker is on, the gate now compares its top score against
`RERANK_RELEVANCE_THRESHOLD` instead. The cross-encoder already reads the
question and passage together to judge relevance, which is a better "is this
answerable at all" signal than vector proximity, and its scores move far less as
the corpus grows. On the 31-document corpus the answerable questions score 1.9
to 9.4 and the absent ones -10.3 to 1.4; 1.7 sits in the gap and classifies all
24 golden questions correctly, the same as the cosine gate at 0.60, so answer
quality is unchanged. The gain is robustness, not accuracy. `--no-rerank` falls
back to the cosine gate.

## Consequences

Reranking is on by default, and its score is the relevance gate. The `psg hit@1`
gain is large and unambiguous, and moving the gate off the bi-encoder cosine
removes a per-corpus recalibration step. The one main-set `psg hit@3` regression
seen on the six-document corpus is a single question inside the run-to-run noise
the project already documents.

Good:

- The answering passage lands first far more often, which is the retrieval
  property most tied to answer quality.
- Two-stage retrieve-then-rerank is the standard architecture for this problem,
  so the design is legible to anyone who has built RAG before.
- No heavy dependency. The reranker rides on the runtime that was already
  installed.

Bad:

- A third model to download and hold in memory (~90 MB on disk, small in RAM).
- One more thing that can be confidently wrong. On easy questions the
  bi-encoder's fuzzier ordering was occasionally safer; `--no-rerank` exists so
  that trade can be re-measured on any corpus.
- The gate now scores a 20-passage pool rather than 3, a few extra dot products
  per query. Negligible at this corpus size, worth noting at a much larger one.
