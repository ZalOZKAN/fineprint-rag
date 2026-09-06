# Fineprint

A question answering assistant for your own paperwork, running entirely on your machine.

Insurance policies, rental agreements and warranties run to dozens of pages, and nobody reads
them until something goes wrong. Fineprint indexes those documents locally and answers
questions about them, quoting the clause the answer came from.

Nothing is uploaded. These documents carry names, addresses, ID and account numbers, so every
step runs on device: parsing, embedding, retrieval and answer generation. No API keys, no
network calls at query time.

## How it works

1. Documents are parsed to text and split into chunks that follow the document headings, so
   each chunk knows which section it belongs to
2. Every chunk is embedded locally and stored in SQLite next to a full text search index
3. A question triggers hybrid retrieval: vector similarity for meaning, BM25 for exact terms
   like clause numbers, dates and amounts, combined with Reciprocal Rank Fusion
4. The top passages go to a local language model, which answers using only that text and cites
   the section
5. If nothing clears the relevance threshold, the model is never called and the assistant says
   the answer is not in the documents

Fineprint reports what a document says. It does not give legal advice.

## Stack

Python, Microsoft Foundry Local for on device inference, SQLite with FTS5, NumPy, Streamlit.

## Status

In development.

## License

MIT, see [LICENSE](LICENSE).
