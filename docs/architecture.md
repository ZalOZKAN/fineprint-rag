# Architecture

Fineprint is a single process application. There is no server, no API key and no
network call at query time. Everything below runs on the machine holding the
documents.

## Layers

```
  cli.py                app.py (Streamlit)
      |                       |
      +-----------+-----------+
                  |
            rag/pipeline.py          orchestration, guards, query log
                  |
      +-----------+-----------+-----------------+
      |           |           |                 |
 rag/prompts  rag/retrieval  rag/llm      rag/ingest
      |           |           |                 |
      |      rag/db.py   Foundry Local   rag/loaders, rag/chunking,
      |      (SQLite)     (on device)     rag/embeddings
      |           |           |                 |
      +-----------+-----------+-----------------+
                  |
            data/fineprint.db
```

The rule that keeps this testable: `rag/` never imports Streamlit or CLI code.
Both interfaces call `pipeline.ask()` or `pipeline.ask_streaming()` and render
whatever comes back. The evaluation harness calls the same function, so what is
measured is what a user gets.

## Indexing, once per document

```
  PDF / DOCX / MD
        |
        |  rag/loaders.py        MarkItDown, PyMuPDF as fallback
        v
  markdown text
        |
        |  rag/chunking.py       split on headings, then on size
        v
  (heading, content) passages
        |
        |  rag/embeddings.py     Foundry Local, then L2 normalize
        v
  1024 dimension unit vectors
        |
        |  rag/db.py             float32 BLOB + FTS5 index
        v
  data/fineprint.db
```

A document is skipped entirely when the SHA-256 of its file contents matches what
is already stored, so re-running ingestion on an unchanged corpus costs one hash
per file and no embedding calls.

## Answering, once per question

```
  question
     |
     |  advice heuristic          refuse -> stop, model never called
     v
  expand query               the question plus rewritten variants, off by
     |                        default and measured off, see ADR 9
     v
  embed query
     |
     +--> dense search      cosine over the embedding matrix, top 8
     |
     +--> sparse search     SQLite FTS5 bm25(), top 8   (hybrid / sparse modes only)
     |
     v
  Reciprocal Rank Fusion    fuse by rank position, not by raw score
     |
     v
  cross-encoder rerank      score each (question, passage) pair, keep top 3
     |
     v
  relevance gate            best reranker score below threshold -> stop
                             (dense cosine instead, if reranking is off)
     |
     v
  build prompt              3 passages, each labelled with its citation
     |
     v
  local chat model          answer grounded in those passages
     |
     v
  answer + citations + scores + timings
```

Two of the three stages can end the request. That is deliberate: refusing is a
correct outcome, and two of the three evaluation categories test for it.

The retrieval strategy is set in `config.RETRIEVAL_MODE`. It defaults to `dense`,
which skips the sparse search and the fusion step; `hybrid` and `sparse` run the
path above. The default was chosen by passage-level measurement, see ADR 2.

`rag/rewrite.py` turns one question into several phrasings so that each can be
retrieved with separately. It is disabled by default: measured, it costs about
12 points of passage hit@3 without reranking and changes nothing at all with it.
The module stays so the experiment can be rerun on another corpus, see ADR 9.

## Why each piece is the way it is

| Decision | Record |
| --- | --- |
| One SQLite file instead of a vector database | [ADR 1](adr/0001-sqlite-as-the-vector-store.md) |
| Retrieval strategy, measured at passage level: dense is the default | [ADR 2](adr/0002-hybrid-retrieval-with-rrf.md) |
| Split on headings before size | [ADR 3](adr/0003-heading-aware-chunking.md) |
| Refusal decided in code, not by the model | [ADR 4](adr/0004-refusal-decided-in-code.md) |
| Execution providers, and a retracted speedup claim | [ADR 5](adr/0005-register-execution-providers.md) |
| Generation bounded in the stream wrapper | [ADR 6](adr/0006-bounded-generation.md) |
| Chat model chosen by measurement | [ADR 7](adr/0007-chat-model-size.md) |
| Cross-encoder reranks retrieval candidates | [ADR 8](adr/0008-cross-encoder-reranking.md) |
| Query expansion, measured and rejected | [ADR 9](adr/0009-query-expansion.md) |

## Data that leaves the machine

None, at query time.

One setup step uses the network: downloading the models on first run. After that the application answers
questions with the network disconnected. Document text, questions and answers are
never transmitted.

The query log at `logs/queries.jsonl` stays local and is git ignored. It records
the question, the outcome, retrieval scores and timings, which is what the
evaluation harness and any later debugging need.
