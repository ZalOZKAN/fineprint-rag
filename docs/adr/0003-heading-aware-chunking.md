# 3. Heading aware chunking

Status: accepted

## Context

Retrieval works on passages, so a document has to be cut up before it can be
indexed. The default approach in most RAG tutorials is a sliding window of a
fixed character count with some overlap.

That default is a poor fit for the documents Fineprint targets. Consumer
contracts are not flowing prose. They are a list of numbered, titled clauses:

    2. Excess
    An excess of 500 EUR applies to every claim. For claims arising from
    escape of water the excess is 1,000 EUR.

    3. What is not covered
    ...

A fixed window cuts across those boundaries. A passage that begins halfway
through clause 2 and ends halfway through clause 3 mixes two unrelated rules, and
worse, it no longer records which clause it came from. Citing "policy.pdf, page
4" is much less useful to someone trying to check a claim than citing
"policy.pdf, Excess".

## Decision

Split on structure first, size second.

1. Walk the markdown and open a new section at every heading. Markdown headings
   are matched directly. A numbered clause opener such as `4.2 Termination` is
   also treated as a heading, because MarkItDown does not always emit those as
   markdown headers.
2. Pack the paragraphs inside a section up to a target size. Only when a single
   section exceeds the maximum is it split further, first on sentence
   boundaries, then on character count as a last resort.
3. Carry the heading with every chunk produced from that section.

The heading is used twice. It is prepended to the text sent to the embedding
model, and it is shown to the user as the citation.

MarkItDown was chosen as the primary loader specifically because it preserves
heading structure. PyMuPDF, the fallback, returns flat text; documents that fall
back therefore chunk by size alone, which is a known and accepted degradation.

## Consequences

Good:

- Citations name a clause rather than a page, which is what someone checking a
  contract actually wants.
- Prepending the heading means a question phrased around "cancellation" matches
  the cancellation clause even when the clause body never repeats the word.
- Chunks are semantically coherent: one clause, one rule.

Bad:

- The result depends on the loader producing headings. A scanned PDF that falls
  back to PyMuPDF loses this entirely.
- The numbered clause pattern is a heuristic. Ingesting the real NFIP flood
  policy showed it misfiring on ordinary list items: `1 Pay its own appraiser;
  and` was read as a heading titled "Pay its own appraiser; and". The pattern was
  tightened to require a real separator after the number and to reject titles
  that end in a conjunction, but a heuristic on regulation text will still miss.
- Sections vary in length, so chunks do too, which makes the number of passages
  sent to the model a less precise proxy for the number of tokens.

One bug found while testing this is worth recording: the overlap added between
consecutive chunks could push a chunk past the configured maximum, because the
overlap was appended after the size check. Overlap is now only applied when the
result still fits, so the maximum is a real bound rather than an approximate one.
