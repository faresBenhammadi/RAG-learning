# RAG Learning Repo

A hands-on playground for learning how Retrieval-Augmented Generation (RAG) systems are built. The centerpiece is `entire_rag_pipeline.ipynb` — a single, self-contained notebook that combines everything below into one working pipeline, from raw documents to a grounded, multimodal answer.

## Repo structure

```
.
├── entire_rag_pipeline.ipynb        # the main pipeline — everything learned, combined
└── improving_rag_techniques/
    ├── rag_tuto.ipynb                       # baseline RAG walkthrough
    └── hybrid_search___reranker.ipynb       # vector vs BM25 vs hybrid, multi-query, reranking
```

`improving_rag_techniques/` holds the exploratory notebooks — the "how does each piece work in isolation" step. `entire_rag_pipeline.ipynb` is the "put it all together" step, and is the file to read to see the current state of the pipeline end to end.

## What `entire_rag_pipeline.ipynb` does

One notebook, run top to bottom, that:

1. **Ingests two kinds of sources** — plain `.txt` files and a PDF (`attention-is-all-you-need`) — into a single unified document set.
2. **Chunks each source appropriately** — fixed-size recursive splitting for text, title-aware chunking for the PDF so tables and images aren't wrenched out of context.
3. **Enriches non-text content** — tables and images extracted from the PDF get a vision-LLM-generated searchable summary, so retrieval isn't blind to anything that isn't plain text. The original raw text/tables/images are kept in metadata so the *final* LLM call still sees the real content, not just the summary.
4. **Embeds and stores everything** in a persistent Chroma vector store (local HuggingFace embeddings, cosine similarity).
5. **Builds three retrievers**: a dense vector retriever, a sparse BM25 keyword retriever, and a hybrid `EnsembleRetriever` combining both (weighted 0.7 / 0.3 toward vector).
6. **Expands the query** — an LLM generates 3 alternative phrasings of the user's question (structured JSON output), and the original question is kept too, so retrieval runs across 4 query variants instead of just one.
7. **Retrieves per variant, then fuses** — each query variant is run through the hybrid retriever independently, and results are merged with **Reciprocal Rank Fusion (RRF)**, which combines rank positions across lists rather than raw (and often incomparable) similarity scores.
8. **Reranks** the fused candidates with a Cohere cross-encoder reranker (`rerank-english-v3.0`) — a second, more precise pass over a small candidate set, since bi-encoder embedding similarity alone tends to under- or over-rate genuine relevance.
9. **Generates the final answer**, feeding the top reranked chunks — text, HTML tables, and images — back into a vision-capable LLM, explicitly instructed to say when the retrieved context doesn't contain the answer (to reduce hallucination).

## What I can now build

Starting from a mix of plain text and PDF sources (including tables and images), I can now put together an end-to-end pipeline that:
- Chunks content appropriately per source type instead of one-size-fits-all splitting.
- Makes non-text content (tables, images) searchable, not just displayable.
- Combines semantic and lexical retrieval instead of relying on embeddings alone.
- Diversifies a single query into multiple retrieval passes and merges the results consistently with RRF.
- Adds a reranking pass to sharpen precision on what actually reaches the LLM.
- Feeds retrieved text, tables, *and* images back into a multimodal LLM for grounded generation.

## Stack

- **Orchestration:** LangChain (`langchain-community`, `langchain-classic`, `langchain-core`)
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5` (local)
- **Vector store:** Chroma
- **Keyword retrieval:** BM25 (`langchain-community`)
- **Reranking:** Cohere Rerank (`rerank-english-v3.0`)
- **PDF/document parsing:** `unstructured` (hi-res strategy, table + image extraction)
- **LLMs:** Groq-hosted models — `openai/gpt-oss-120b` for query expansion, `qwen/qwen3.6-27b` (vision-capable) for table/image summarization and final answer generation


