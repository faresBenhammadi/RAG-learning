# RAG Learning Repo

A hands-on playground for learning how Retrieval-Augmented Generation (RAG) systems are built, from a basic ingestion pipeline up to hybrid search with reranking. Each notebook tackles one layer of the stack, going from "get *something* working" to "make retrieval actually good."

## What this repo demonstrates

By working through this repo, I built a working understanding of the full RAG pipeline: ingestion → chunking → embedding → storage → retrieval → reranking → generation, and of the tradeoffs at each stage.

### 1. Ingestion pipelines
- **Plain text ingestion** (`rag_tuto.ipynb`): loading `.txt` files with `DirectoryLoader` + `TextLoader`, splitting with `CharacterTextSplitter` (fixed `chunk_size`, `chunk_overlap`).
- **Multimodal PDF ingestion** (`multi_rag_model.ipynb`): parsing PDFs with `unstructured`'s `partition_pdf` (hi-res strategy), extracting tables (with inferred structure) and images as separate element types instead of flattening everything to raw text.

### 2. Chunking strategies
- Naive fixed-size character chunking (`CharacterTextSplitter`).
- **Title-based / semantic chunking** with `chunk_by_title` (from `unstructured`), which groups content under section headers instead of cutting mid-topic, with controls for `max_characters`, `new_after_n_chars`, and `combine_text_under_n_chars`.

### 3. Multimodal chunk enrichment
- Separating each chunk into text / tables / images.
- Using a vision-capable LLM (via Groq) to generate a **searchable text description** for chunks that contain tables or images, so retrieval isn't blind to non-text content.
- Keeping the original raw content (text, HTML tables, base64 images) in metadata, so the enriched summary is used for *search* while the original content is passed to the LLM at *generation* time.

### 4. Embeddings & vector storage
- Local embeddings with `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`) — no external embedding API dependency.
- Persisted vector storage with **Chroma**, using cosine similarity (`hnsw:space: cosine`).
- Building the store both in one shot (`Chroma.from_documents`) and incrementally (`Chroma(...) + add_documents`).

### 5. Query handling
- **Conversational query reformulation**: before retrieving, passing the chat history + latest question to an LLM to rewrite the query so it's self-contained (resolves pronouns/context from earlier turns) before it hits the retriever.

### 6. Retrieval methods
- **Vector (semantic) retrieval** — dense embedding similarity search via Chroma's retriever interface.
- **Keyword retrieval (BM25)** — exact/lexical matching with `BM25Retriever`, useful for queries with specific terms, numbers, or names that embeddings can blur.
- **Hybrid search** — combining vector + BM25 with `EnsembleRetriever`, weighting each retriever's contribution (e.g. 0.5 / 0.5) so exact-match strength and semantic recall reinforce each other instead of relying on one alone.
- **Multi-query retrieval** — generating multiple reformulations of the same query (via an LLM) and retrieving for each, then merging results, to reduce sensitivity to how the original query happens to be phrased.
- **Reciprocal Rank Fusion (RRF)** — merging ranked result lists from multiple retrievers (e.g. vector + BM25, or multiple query variants) by combining their rank positions rather than raw scores, which avoids the problem of different retrievers producing scores on incomparable scales.

### 7. Reranking
- Using a **cross-encoder reranker** (Cohere `rerank-english-v3.0`) as a second-pass "quality inspector" over the hybrid retriever's candidates.
- Understanding *why* this matters: bi-encoder embedding search scores a query and a chunk independently and compares vectors (fast but coarser), while a cross-encoder scores the (query, chunk) pair jointly (slower, but much more precise) — so reranking a small candidate set catches relevance that embedding similarity alone misses.

### 8. Generation
- Grounding the LLM's answer strictly in retrieved context, with an explicit instruction to say when the answer isn't in the documents (reduces hallucination).
- Passing images back into the LLM prompt (multimodal generation) alongside text/table context for the multimodal pipeline.
- Using Groq-hosted models (`openai/gpt-oss-120b`, `qwen/...`) as the generation backend.

## Repo structure

| Notebook | Focus |
|---|---|
| `rag_tuto.ipynb` | End-to-end baseline RAG: ingestion → chunking → embeddings → Chroma → retrieval → conversational query rewriting → generation loop |
| `multi_rag_model.ipynb` | Multimodal RAG over PDFs: `unstructured` parsing, title-based chunking, table/image extraction, AI-generated searchable summaries for non-text chunks |
| `hybrid_search___reranker.ipynb` | Retrieval quality: vector vs. BM25 vs. hybrid (`EnsembleRetriever`), plus cross-encoder reranking with Cohere |

## Stack

- **Orchestration:** LangChain (`langchain-community`, `langchain-classic`, `langchain-core`)
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5` (local, via `langchain-huggingface`)
- **Vector store:** Chroma
- **Keyword retrieval:** BM25 (`langchain-community`)
- **Reranking:** Cohere Rerank (`rerank-english-v3.0`)
- **PDF/document parsing:** `unstructured` (hi-res strategy, table + image extraction)
- **LLMs:** Groq-hosted models (`openai/gpt-oss-120b`, `qwen/...`) for query rewriting, summarization, and generation

## What I can now build

Starting from a folder of raw documents (text or PDF, including tables/images), I can now:
- Design a chunking strategy appropriate to the content (fixed-size vs. title/semantic-aware).
- Make non-text content (tables, images) searchable, not just displayable.
- Stand up a persistent vector store and query it.
- Combine semantic and lexical retrieval (hybrid search) instead of relying on embeddings alone.
- Diversify a single query into multiple retrieval passes (multi-query) and merge results consistently (RRF).
- Add a reranking pass to sharpen precision on the final context sent to the LLM.
- Handle multi-turn conversations by reformulating queries with history before retrieval.
- Constrain generation to retrieved context to reduce hallucination.


