# 4. Refusal is decided in code, not by the model

Status: accepted

## Context

An assistant that answers questions about contracts has two ways to be wrong,
and the second is worse than the first:

1. It fails to find an answer that is in the documents.
2. It invents an answer that is not in the documents.

The second failure is what makes a tool like this untrustworthy. Someone
checking whether their policy covers water damage cannot tell a fabricated
clause from a real one, which is precisely why they are asking.

The standard mitigation is a line in the system prompt: "if the context does not
contain the answer, say you do not know". Fineprint runs a small local model
(4B parameters, see [ADR 7](0007-chat-model-size.md)). Instructions like that
are followed inconsistently at that size, and there is no way to test
compliance short of asking many times and counting.

There is a second category to handle. Some questions are not about what a
document says at all: "should I cancel this policy", "can I sue my landlord".
Answering those is giving legal advice, which this project explicitly does not do.

## Decision

Refuse in code, before the model is called. Two independent guards run in front
of generation, and both are ordinary Python that a test can assert on.

**Relevance gate.** After retrieval, if the best relevance score across the
retrieved passages is below a threshold, the question is answered with a fixed
message and the chat model is never invoked.

The score is the cross-encoder reranker's top score when reranking is on (the
default), and the best dense cosine otherwise. Both are deterministic Python a
test can assert on. The move to the reranker score is [ADR 8](0008-cross-encoder-reranking.md):
a bi-encoder cosine threshold had to be recalibrated every time the corpus
changed, and a relevance judgement carries across corpora better.

The dense score is taken across the fused result set rather than from the dense
candidate list alone. A passage found only by BM25 has no dense rank, and gating
on the dense list would have rejected exactly the exact term matches that hybrid
retrieval was added to catch.

**Advice heuristic.** A regular expression over the question detects phrasings
that ask for a recommendation rather than for a document fact: "should I", "can I
sue", "do you recommend", "is it legal", and similar. Matching questions are
refused before retrieval even runs.

The system prompt still carries the same instructions. It is the third layer, not
the first.

## Consequences

Good:

- Refusal is deterministic. The same question refuses the same way every time,
  which is what makes `refusal_accuracy` a meaningful number in the evaluation
  set rather than a sample from a distribution.
- Refusing before generation is also the fastest possible path, since generation
  dominates response time.
- Both guards are unit tested directly, with a fake chat model that records
  whether it was called at all.

Bad:

- The advice heuristic is pattern matching. It will miss advice requests phrased
  in a way the patterns do not cover, and it can misfire on a factual question
  that happens to contain a matching phrase. This is measured rather than
  claimed: `out_of_scope` is its own evaluation category.
- The relevance threshold is a single global number. A well phrased question
  about an obscure clause and a vague question about a common one are judged by
  the same bar.
- A threshold set too high refuses answerable questions. The evaluation set
  measures both directions, so the cost of moving it is visible.
