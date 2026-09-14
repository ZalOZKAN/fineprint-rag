# 1. SQLite as the vector store

Status: accepted

## Context

Fineprint has to store document passages, their embeddings, and a keyword index,
and search all three from one process on a laptop with no network access.

The usual answer is a dedicated vector database. Chroma, Qdrant, FAISS and
LanceDB all solve this well at scale. They also all add either a server process,
a native dependency, or a second storage format sitting next to the text.

The corpus this project targets is a person's own paperwork: a policy, a lease, a
warranty, a phone contract. That is tens of documents and low thousands of
passages, not millions.

## Decision

Store everything in one SQLite file.

- Passages and their metadata go in ordinary tables.
- Embeddings are stored as raw `float32` bytes in a `BLOB` column.
- Keyword search uses SQLite's built in FTS5 extension with its `bm25()` ranking
  function.
- Dense search loads the embedding matrix into NumPy once and scores every
  passage with a single matrix multiplication.

Vectors are L2 normalized when they are written, so cosine similarity reduces to
a dot product and the whole search is one `matrix @ query` call.

## Consequences

Good:

- Zero extra dependencies. FTS5 and `bm25()` were verified present in the Python
  bundled SQLite (3.50.4) before this was chosen.
- One file holds the text, the vectors and the keyword index, so a backup is a
  file copy and there is no way for the three to drift apart.
- No server process, which matters for an application whose entire claim is that
  nothing leaves the machine.
- Storing vectors as `float32` bytes rather than JSON keeps the database roughly
  four times smaller and removes parsing from the read path.

Bad:

- Dense search is a linear scan. At the target size (42 passages in the sample
  corpus) this is microseconds, but there is no index to fall back on, so the
  cost grows linearly with the corpus.
- The whole embedding matrix is held in memory. At 1024 dimensions and float32
  that is 4 KB per passage, so 10,000 passages would be 40 MB. Acceptable here,
  not acceptable at a million.

The crossover point where a real vector index earns its complexity is somewhere
in the low hundreds of thousands of passages. If Fineprint ever indexes a
company's document store rather than one person's paperwork, this decision should
be revisited. For its stated scope it is not close.
