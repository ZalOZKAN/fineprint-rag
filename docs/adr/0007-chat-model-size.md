# 7. Choosing the chat model by measurement

Status: accepted

## Context

In a RAG system the model's job is narrower than in open conversation. It is
handed three passages that already contain the answer and asked to state what
they say. That is closer to reading comprehension than to recall, and it is
reasonable to hope a small model is enough, because a small model is much
faster and speed is the whole difference between a tool someone uses and one
they do not.

The reference plan for this project makes exactly that bet: smaller models
respond faster, and it prioritises speed so that feedback is quick.

Hoping is not measuring. Three sizes from one family were run over the same 29
question evaluation set, so that only parameter count varies. Staying inside the
Qwen3 family keeps the training recipe constant; comparing across vendors would
have confounded size with everything else that differs between them.

## Measurement

    python evaluation/run_eval.py --chat-model qwen3-0.6b --results evaluation/results_qwen3-0.6b.json
    python evaluation/run_eval.py --chat-model qwen3-1.7b --results evaluation/results_qwen3-1.7b.json
    python evaluation/run_eval.py --chat-model qwen3-4b   --results evaluation/results_qwen3-4b.json

Each was run twice. The table shows the range across both runs.

| | qwen3-0.6b | qwen3-1.7b | qwen3-4b |
| --- | --- | --- | --- |
| Overall | 89.7 % (both) | 93 to 97 % | 100 % (both) |
| Answer accuracy | 86.7 % (both) | 87 to 93 % | 100 % (both) |
| Retrieval hit@3 | 100 % | 100 % | 100 % |
| Latency p50 | 7 to 8 s | 14 to 16 s | 28 to 30 s |
| Latency p95 | 16 to 28 s | 35 to 43 s | 54 to 205 s |

Latency increases with size and roughly doubles at each step. hit@3 is flat by
construction and serves as a control: it does not depend on the chat model, so
if it had moved the comparison would have been measuring something else.

## How much of this is real

The reproducibility spread on this set, measured separately across three
consecutive runs of `qwen3-1.7b`, is 3.4 percentage points. Against that:

- **0.6b lands on 89.7 % both times. 4b lands on 100 % both times.** Those two
  are stable.
- **1.7b swings from 93 % to 97 %**, across most of the spread. Its single
  numbers are not reliable.
- 0.6b to 1.7b: the accuracy gap overlaps the noise on 1.7b's low run. **Not
  established.**
- 1.7b to 4b and 0.6b to 4b: 4b answers every question correctly on both runs
  while neither smaller model does. **Real.**

The latency differences are far outside any noise. One number is worth
singling out: 4b's p95 hit 205 s on the second run, a single empty generation
that triggered the retry added in ADR 6. The retry doubles the worst case, and
4b's worst case is already the slowest, so 4b has a punishing tail even though
its median is only twice 1.7b's.

## Decision

Use `qwen3-4b` as the default.

The table above was measured against an earlier synthetic corpus, where 1.7b was
close enough to 4b that the latency argument won and 1.7b was the default. That
changed when the corpus was replaced with two real federal documents, the NFIP
flood policy and the FTC cooling-off rule. On regulation prose, 1.7b stopped
extracting figures and started hedging: asked for the door-to-door cancellation
window it said "the exact duration is specified in the document" without giving
the three days, and a flood-deductible question ran for over a minute and
produced a vague paragraph. 4b answers those directly.

The corpus later grew again, first to six overlapping federal documents and then
to 31. The size sweep has not been re-run, because the finding it produced (only
4b extracts figures reliably from regulation prose) held on two real documents
and there is no reason a harder corpus would reverse it. 4b on the 31-document
set scores about 81 % answer accuracy on the main golden set; its remaining
misses are retrieval failures, the answering clause never reaching the model,
not comprehension failures, so a larger model would not recover them. See the
README Results section.

So the trade reversed. On clean synthetic clauses the speed of 1.7b was worth
its small accuracy cost. On real regulation the accuracy cost is not small, and a
fast wrong answer about a legal deadline is worse than a slow right one.

`--chat-model` on the evaluation harness and `CHAT_MODEL` in `config.py` make the
choice a one line change for anyone who would rather have 1.7b's speed and accept
its hedging.

## Consequences

Good:

- The reference plan's assumption, that a small model suffices because retrieval
  does the work, is tested rather than inherited. It holds only partly: the
  retrieval half is genuinely model independent, but reading three passages and
  reporting them accurately is not free, and 0.6b is measurably worse at it.
- Someone who values speed over the accuracy on hard clauses has a one line
  change back to 1.7b and a table telling them what it costs.

Bad:

- Two runs per model, not more. Enough to see that 0.6b and 4b are stable and
  1.7b is not, but not enough to put a confidence interval on any single figure.
  The table gives ranges rather than points for that reason.
- The 1 to 3 second target in the reference plan is out of reach on this machine
  at any of these sizes. The fastest configuration measured is 7 s at the lowest
  accuracy. Closing that gap needs hardware acceleration, which
  [ADR 5](0005-register-execution-providers.md) establishes is unavailable
  through this catalog.
