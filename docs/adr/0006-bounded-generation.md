# 6. Generation is bounded in the stream wrapper

Status: accepted

## Context

The Foundry Local chat client offers two methods:

    complete_chat(messages, tools)
    complete_streaming_chat(messages, tools)

Neither takes a maximum token count. There is no parameter that says "stop after
400 tokens". The model generates until it decides to stop, and the application
consumes whatever arrives.

That is fine when the model behaves. A small local model does not reliably
behave. Two failure modes matter here:

**Rambling.** Asked for a clause, the model sometimes restates the whole context,
adds caveats, and offers a summary of the summary. The answer is not wrong, it is
just far longer than the two sentences the question deserved, and every extra
token is time the user waits.

**Repetition loops.** Small models fall into cycles where they emit the same
sentence over and over. This is a well known failure at this size and it does not
resolve on its own: the loop continues until something external stops it. With no
token limit and a blocking `"".join(stream)`, a single looping question would
hang the interface indefinitely.

An early configuration constant named `MAX_ANSWER_TOKENS` existed in this project
and was never referenced by anything, because there was nowhere to pass it. Dead
configuration that looks like a working control is worse than no control at all.

## Decision

Bound generation where the stream is consumed, in `ChatModel.stream()`.

**Length ceiling.** Stop yielding once the answer passes `MAX_ANSWER_CHARS`
(2000). Characters rather than tokens, because characters are what the wrapper
can actually count without a tokenizer.

**Loop detection.** Every 20 fragments, check whether the trailing
`REPEAT_WINDOW` (80) characters have already appeared `REPEAT_LIMIT` (3) times in
the answer so far. If they have, stop.

    def is_repeating(text, window, limit):
        if len(text) < window * limit:
            return False
        return text.count(text[-window:]) >= limit

Counting occurrences of the trailing window catches repeats of any length that
divides it, so a looping phrase and a looping sentence are both detected without
knowing the period in advance.

Both cuts are deterministic. Neither depends on sampling temperature, on a
prompt instruction, or on the model choosing to stop.

## Consequences

Good:

- A looping question ends in bounded time instead of hanging the interface.
- The guards are unit tested against a fake streaming client, so the behaviour is
  verified without waiting for the real model to misbehave on cue.
- Because the cut happens in the generator, the fragments after the cut are never
  requested from the model, so the time is genuinely saved rather than discarded.

Bad:

- Checking every 20 fragments means a loop runs for up to 20 fragments past the
  point it became detectable. Checking on every fragment would be exact and
  wasteful; this is the trade.
- The length ceiling can truncate a legitimately long answer mid sentence. For
  the questions this project targets, a clause lookup, 2000 characters is
  generous, but a question that genuinely warrants a long answer will be cut.
- `text.count(text[-window:])` scans the whole answer. The answer is bounded by
  the ceiling, so the cost is bounded too, but it is not free.
