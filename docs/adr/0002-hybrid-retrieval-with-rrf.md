# 2. Hybrid retrieval fused with Reciprocal Rank Fusion

Status: accepted, then revised. Equal-weight RRF was the default. On the
six-document corpus a passage-level metric showed it puts the answering clause
in the model's context *less* often than dense search alone, so the default is
now `dense`. The history below is kept because the reasoning still applies to a
larger corpus and the flip should be re-checked when one exists.

## Context

The retrieval step decides everything downstream. If the passage holding the
answer never reaches the model, no prompt and no model size will recover it. So
the question is which search finds the right passage.

Dense retrieval, comparing embeddings, is the default in most RAG systems and it
is good at meaning. A question asking "can I let a room to someone else" finds a
clause about subletting even though the two share no words.

It is weak in exactly the place this corpus lives. Consumer contracts are dense
with tokens that must match exactly:

    an excess of 500 EUR          a specific amount
    within 14 days                a deadline
    clause 4.2                    a reference
    the escape of water           a defined term with a precise meaning

An embedding compresses a passage into 1024 numbers. That representation is good
at topic and bad at exact tokens: "500 EUR" and "1,000 EUR" land close together
because they mean nearly the same kind of thing, which is precisely the
distinction a reader cares about. BM25 has the opposite profile. It cannot tell
that subletting and letting a room are related, and it matches "500" perfectly.

Neither search is sufficient. The interesting question is not whether to combine
them but how, and whether the combination actually helps on this corpus. Most
projects assert that hybrid retrieval is better and do not check.

## Decision

Run both searches and fuse their results with Reciprocal Rank Fusion.

Dense search returns the top 8 passages by cosine similarity. Sparse search
returns the top 8 by BM25 through SQLite FTS5. Each passage is then scored as

    score(passage) = sum over lists of  1 / (k + rank in that list),  k = 60

and the best 3 become the model context.

Fusing on rank rather than on score is the point. A cosine similarity of 0.58 and
a BM25 score of 12.4 are not comparable quantities, and normalising them would
require assumptions about their distributions that neither search guarantees.
Ranks are always comparable. The constant k flattens the difference between the
top few positions, so a passage that both searches rank highly beats a passage
that only one search puts first, which is the behaviour worth having: two
independent methods agreeing is stronger evidence than one method being
confident.

RRF also has no weight to tune. A weighted score blend would need a coefficient
fitted on data this project does not have.

## Verification

The claim that hybrid beats dense is tested rather than assumed. `Mode.DENSE`
exists in the code purely so the baseline can be run, and the evaluation harness
takes a `--compare` flag that runs both over the same questions:

    python evaluation/run_eval.py --compare

The metric is hit@3: for each question whose answer is known to live in a
specific document, whether that document appears among the 3 passages sent to the
model. It is measured separately from answer accuracy so that retrieval quality
is not confounded with how well the chat model writes.

**The comparison found no difference.** Over 15 answerable questions on the
sample corpus:

| Mode | hit@3 | Overall |
| --- | --- | --- |
| dense only | 100 % (15/15) | 72.4 % |
| hybrid + RRF | 100 % (15/15) | 72.4 % |

Dense retrieval alone already places the right document in the top 3 for every
question. There is no headroom for hybrid search to recover, so the two are
indistinguishable here.

This is a limitation of the test, not evidence that hybrid retrieval is useless.
The sample corpus is 5 documents and 42 passages covering five unrelated
subjects: insurance, tenancy, a warranty, a phone plan, a gym. Embeddings
separate those topics trivially. A question about the excess cannot be confused
with a question about roaming charges, so the exact term matching that BM25
contributes is never the deciding factor.

### The harder test, and what it found

A test that could distinguish the two needs vocabulary that overlaps between
documents, so `evaluation/golden_set_hard.json` was written: 15 questions whose
answer appears verbatim in the wrong document, or whose terms appear in several.
Two documents charge an "administration fee", of 25 and 50 EUR. Two promise a
response in "5 working days". Two use the number 60, for months and for days.

Retrieval is also now compared without generating anything, in
`evaluation/compare_retrieval.py`. Measuring retrieval through the full pipeline
buries it in the noise of a model that answers differently on consecutive runs
and costs 15 seconds per question rather than milliseconds. The script reports
hit@1 and MRR alongside hit@3, because hit@3 saturates and stops discriminating.

The first result was not the expected one:

| | hit@1 | MRR |
| --- | --- | --- |
| dense only | 100 % | 1.000 |
| sparse only | 93.3 % | 0.967 |
| **hybrid + RRF** | **93.3 %** | **0.967** |

**Hybrid was worse than dense.** It lost a question dense answered correctly.

The failing question was "what administration fee applies when I cancel the
insurance policy". Dense ranked the insurance cancellation clause first at 0.588.
BM25 ranked the *gym* cancellation clause first, and RRF, which weights both
lists equally, promoted it over a passage dense was confident about.

The cause was in the query builder, not in the idea. Every word longer than one
character was being sent to FTS5, including "what", "the", "when" and "applies".
Those carry BM25 weight without carrying meaning, and they raise the score of a
short passage more than a long one, so a brief unrelated clause outranked the
right one. Filtering stopwords restored parity:

| | hit@1 | MRR |
| --- | --- | --- |
| dense only | 100 % | 1.000 |
| sparse only | 100 % | 1.000 |
| hybrid + RRF | 100 % | 1.000 |

Writing those tests exposed a second defect in the same function. Splitting the
question on every non alphanumeric character turned a clause reference like
`4.2` into the single digits `4` and `2`, which were then dropped as too short.
Clause references are precisely what the keyword half of this design exists to
match, so the search was failing at its stated purpose. The tokenizer now keeps
numbers whole, including `1,000` and `29.99`.

### What the experiment actually established

Hybrid does not beat dense on the synthetic corpus. Dense alone scores 100 % on
both the easy and the hard set, so there is nothing left to recover. What the
harder test did find was two real bugs in the sparse half, neither of which the
easy test could surface, and both of which would have degraded retrieval on a
corpus large enough for the keyword search to matter.

### The corpus then changed, and the experiment went silent

The project later replaced the five synthetic documents with two real ones: the
NFIP flood policy and the FTC cooling-off rule. On two documents about unrelated
subjects, "which document" is trivial for dense, sparse and hybrid alike, and
`compare_retrieval.py` scores all three at 100 % hit@1 and MRR 1.000 on both the
easy and the hard set. The metric is document-level, and the colliding clauses
the hard set targets are inside one document; the regulation's section headings
are too coarse to score at clause level.

So the experiment now measures nothing at all. The synthetic corpus was better
for it, precisely because five unrelated subjects put the collisions between
documents where a document-level metric can see them. That was the reason it was
built that way, and the reason its retrieval numbers are kept here even though
the shipped corpus is different.

Results are in `evaluation/retrieval_core.json` and `evaluation/retrieval_hard.json`.

### The six-document corpus, and the metric that finally discriminated

The corpus was then rebuilt a third time, deliberately to make retrieval hard.
Six real federal documents replaced the two: three NFIP flood forms (policy,
general property, condominium), and three FTC and OSHA rules (cooling-off, mail
order, injury recordkeeping). The three flood forms share most of their
vocabulary. Deadlines collide across all six: OSHA alone has 8 hours, 24 hours,
7 days and 5 years, against 30 days for a mail-order refund and 30 days for a
flood renewal.

The other change was to the metric. `doc_hit_at_k` asks whether the right
*document* reached the top 3. With a 223-chunk document in the corpus that
question is nearly free, and it was hiding the one that matters: whether the
specific clause that answers the question reached the model. `compare_retrieval.py`
now also reports `passage_hit_at_k`, using an `expect_phrase` field in the golden
set that pins the verbatim answering sentence, and finds the rank of the first
retrieved chunk that contains it.

With `CONTEXT_CHUNKS = 3`, `passage_hit_at_k` at k=3 is exactly "did the model
get the sentence it needed". It is an upper bound on answer accuracy: the model
cannot quote a clause it never saw.

| core set (16) | doc hit@1 | doc hit@3 | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- | --- | --- |
| dense | 81 % | 100 % | 31 % | **75 %** | 0.896 |
| sparse | 94 % | 100 % | 56 % | 62 % | 0.958 |
| hybrid + RRF | 81 % | 100 % | 44 % | 62 % | 0.906 |

| hard set (15) | doc hit@1 | doc hit@3 | psg hit@1 | psg hit@3 | MRR |
| --- | --- | --- | --- | --- | --- |
| dense | 73 % | 100 % | 20 % | **53 %** | 0.833 |
| sparse | 80 % | 100 % | 13 % | 47 % | 0.889 |
| hybrid + RRF | 80 % | 100 % | 20 % | **33 %** | 0.889 |

Dense puts the answering clause in the context most often on both sets. Hybrid is
the worst of the three on `psg_hit@3`, and on the hard set it is far worse: 33 %
against dense's 53 %.

The reason is the same equal weighting that RRF was chosen for. When one
retriever ranks the right clause 2nd and the other ranks it 9th, RRF with equal
weight averages them down to roughly 5th, past the k=3 cut. Dense on its own
would have kept it. Fusing two lists helps when both are roughly as good; it
hurts when one is clearly better for the query and the fusion has no way to know
which. On this corpus, for these questions, dense is usually the better list, and
blending it 1:1 with sparse throws away that advantage.

`doc_hit@1` and MRR still favour sparse and tie dense with hybrid, which is why
the earlier document-level runs saw nothing wrong. Those metrics were measuring
the wrong thing.

## Consequences

The default is now `dense` (`config.RETRIEVAL_MODE`). It is the mode that most
often gets the answering clause into the model's context on the shipped corpus,
which is the only retrieval property that bounds answer quality.

Hybrid and sparse stay in the code as `config.RETRIEVAL_MODE` values and as
`run_eval.py --mode`, so the comparison re-runs on demand. The sparse half keeps
the two fixes the synthetic experiment forced (stopword filtering, whole clause
numbers); it is a worse default here but not a broken one, and `psg_hit@1` shows
it finds the exact clause first more often than dense does, it just recovers
worse by k=3.

What is explicitly *not* done: weighted fusion. A blend that trusted dense more
than sparse would very likely beat both, but the weight has to be fitted, and
this corpus is too small to fit it without overfitting the golden set. That is
the real fix and it is left as future work, stated as a gap rather than papered
over.

The judgement from the two-document era, that a real household's messier and
larger paperwork is where exact-term matching starts to pay off, still stands and
is still a judgement. If that corpus ever exists, re-run `--compare` and revisit
this.

Good:

- The FTS5 index and both search paths are already built and tested, so trying
  hybrid or sparse again is a one-line config change, not a rebuild.
- No new dependency. BM25 comes from the SQLite already storing the text.
- The mode is a config value and a `run_eval.py` flag, so the comparison re-runs
  at any time and stays honest as the corpus changes.
- The passage-level metric now exists. Whatever the corpus, `psg_hit@k` reports
  the number that actually bounds answer quality.

Bad:

- The default carries a dependency it does not use at query time. `dense` mode
  never touches FTS5, but the index is still built and maintained at ingest.
- Equal-weight RRF is kept in the code as a mode despite being measurably the
  worst of the three here. It is retained only for the larger-corpus hypothesis,
  which is unverified.
- The FTS5 query still has to be built defensively for the non-default modes. A
  user question containing a quote or the bare word NEAR is a syntax error in
  FTS5, so terms are extracted and quoted individually and a malformed
  expression degrades to no keyword matches rather than raising.
- A passage found only by BM25 has no dense rank, so the relevance gate has to
  score it separately, otherwise the gate would reject exactly the exact term
  matches sparse and hybrid mode exist to catch. See ADR 4.
