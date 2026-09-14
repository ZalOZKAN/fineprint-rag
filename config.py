"""Central configuration for Fineprint.

Every tunable constant lives here. Modules must not hardcode values that belong
in this file, so that experiments only require editing one place.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
# The internship's evaluation corpus: 31 public-domain federal documents the
# golden sets, ADRs and README numbers are all measured against. Not where a
# user's own documents go - that is LIBRARY_DIR below. Lives under evaluation/
# rather than data/ so its name alone says what it is for.
EVAL_CORPUS_DIR = PROJECT_ROOT / "evaluation" / "corpus"
DATABASE_PATH = DATA_DIR / "fineprint.db"
LOG_DIR = PROJECT_ROOT / "logs"
QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"

# --- Foundry Local models ------------------------------------------------

APP_NAME = "fineprint"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen3-4b"

# --- Document loading ----------------------------------------------------

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")

# --- Personal library (Streamlit app) -------------------------------------
# The interactive app does not read or write EVAL_CORPUS_DIR/DATABASE_PATH
# above. Those hold the sample corpus the internship evaluation is measured
# against, and the CLI and evaluation scripts keep using them so those numbers
# stay reproducible. The app instead keeps its own folder and database, so
# cloning the repository and running the app opens an empty personal
# workspace: Fineprint is a tool for the documents you load into it, not a
# viewer for this project's own test corpus. The sample corpus is one
# explicit button away in the app (see app.py), never loaded automatically.
LIBRARY_DIR = DATA_DIR / "library"
LIBRARY_DATABASE_PATH = DATA_DIR / "library.db"

# --- Chunking ------------------------------------------------------------
# Chunks follow document headings. A section longer than CHUNK_MAX_CHARS is
# split further, repeating CHUNK_OVERLAP_RATIO of the previous chunk so that a
# sentence spanning the boundary stays retrievable.

CHUNK_TARGET_CHARS = 700
CHUNK_MAX_CHARS = 1200
CHUNK_MIN_CHARS = 80
CHUNK_OVERLAP_RATIO = 0.15

# --- Retrieval -----------------------------------------------------------
# Dense search finds passages by meaning, sparse (BM25) search finds exact terms
# such as clause numbers and amounts. Both candidate lists are fused with
# Reciprocal Rank Fusion and the best CONTEXT_CHUNKS passages become context.

# Retrieval strategy: "dense" (vector), "sparse" (BM25), or "hybrid" (both,
# fused with RRF). Measured on the six-document corpus, dense alone put the
# answering passage in the top 3 more often than equal-weight hybrid; see
# docs/adr/0002. Change here to compare.
RETRIEVAL_MODE = "dense"

DENSE_CANDIDATES = 8
SPARSE_CANDIDATES = 8
CONTEXT_CHUNKS = 3
RRF_K = 60

# Query expansion. One question is retrieved with more than one phrasing (the
# original, the same question scoped to the rule it names, and its content words
# alone) and the rankings are fused by the same RRF that fuses dense and sparse.
# It was built to attack the first-stage recall ceiling in docs/adr/0008, and it
# does not: measured off. Without a reranker it costs about 12 points of passage
# hit@3 in every retrieval mode, because RRF rewards agreement and two of the
# three rankings now come from worse queries. With the reranker on, which is the
# default, it changes nothing at all, to three decimals. Kept behind this flag
# with the measurement in docs/adr/0009 rather than deleted, so the experiment
# can be rerun on another corpus with one command.
QUERY_EXPANSION_ENABLED = False
QUERY_VARIANTS = 3

# Cross-encoder reranking. Dense and sparse retrieval score the query and a
# passage independently; a cross-encoder reads the pair together and is much
# better at telling the passage that answers the question from one that only
# shares its topic. It is too slow to run over the whole corpus, so it reruns
# only the RERANK_CANDIDATES that retrieval already surfaced and reorders them.
# Cost is tens of milliseconds on CPU, negligible next to generation. The model
# is an ONNX cross-encoder run through onnxruntime (no torch); it downloads once
# from the Hugging Face hub, like the Foundry models. See docs/adr/0008.
RERANK_ENABLED = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

# When the reranker runs, its top score is the relevance gate instead of the
# dense cosine below. The cross-encoder is trained on relevance judgements and
# its scores barely move when the corpus grows, where the cosine gate had to be
# recalibrated from 0.52 to 0.60 going from 6 to 31 documents. This is a raw
# ms-marco logit (roughly -11 to +11). On the 31-document corpus the golden-set
# answerable questions score 1.9 to 9.4 and the absent ones -10.3 to 1.4, so 1.7
# sits in the gap and classifies all 24 correctly, the same as the cosine gate
# at 0.60. The point is not a tighter split on this corpus but a more stable
# one: a cross-encoder judges relevance directly, so its cutoff should carry
# across corpora with less retuning than a bi-encoder cosine.
RERANK_RELEVANCE_THRESHOLD = 1.7

# Gatekeeper: when the best dense cosine similarity falls below this value the
# question is treated as unanswerable and the chat model is never called.
# Cosine similarity of L2 normalized vectors ranges from -1 to 1.
# Recalibrated on the 31-document corpus (evaluation/calibrate_threshold.py):
# answerable questions score 0.611 to 0.850, absent ones 0.250 to 0.585. The
# larger corpus pushed the absent scores up (it now holds passages loosely
# related to almost any consumer question), so the gate had to move up with
# them: at the old 0.52 it let five not-in-corpus questions through to the model.
RELEVANCE_THRESHOLD = 0.60

# --- Generation ----------------------------------------------------------
# ChatClientSettings.max_tokens was tried as a native generation cap. On this
# runtime a low value (320) collapsed answer accuracy from ~81% to ~56%: the
# limit appears to bound prompt + completion together, and the prompt alone is
# already larger, so the answer was cut off before it began. Forcing
# temperature 0 made it worse still. Both are left unset; generation is bounded
# in the stream wrapper instead (below).
MAX_ANSWER_TOKENS: int | None = None
GENERATION_TEMPERATURE: float | None = None

# Hard ceiling on answer length. The prompt asks for two or three sentences; a
# clause lookup never needs more, and a lower ceiling is a lower worst-case wait
# on CPU. Past this the model is rambling, not answering.
MAX_ANSWER_CHARS = 700

# Small models fall into repetition loops, emitting the same sentence forever.
# Generation stops when the trailing REPEAT_WINDOW characters have already
# appeared REPEAT_LIMIT times in the answer so far.
REPEAT_WINDOW = 80
REPEAT_LIMIT = 3

# Wall-clock ceiling on a single generation. The character and repetition guards
# do nothing when the model streams empty chunks for minutes (an observed
# failure mode), so time is bounded directly. A well-behaved hard answer on CPU
# finishes well inside this (measured p50 ~45s, p95 ~115s); past it the request
# is not going to recover.
MAX_GENERATION_SECONDS = 150

# The model occasionally streams for minutes and returns nothing. It is
# intermittent, and the same request usually succeeds on a second attempt, so
# one retry is made before the failure is reported. Kept at one: a second
# retry would double the worst case wait for a case that is already rare.
RETRY_ON_EMPTY_GENERATION = True

# ...but only retry when the empty attempt was quick. A fast empty return is a
# transient glitch a retry fixes; an empty attempt that ran into the time
# ceiling is a stuck generation, and retrying it just spends the budget twice.
RETRY_EMPTY_MAX_ELAPSED_SECONDS = 30
