# RAG Learning — Chunking Strategies

This repository is my learning journey through Retrieval-Augmented Generation (RAG).

This document is a personal reference for the different **chunking strategies** I learn.

---

# 1. What is Chunking?

Before a RAG system can retrieve useful information, large documents are usually divided into smaller pieces called **chunks**.

For example:

```text
Large document
      ↓
   Chunking
      ↓
┌──────────┐
│ Chunk 1  │
├──────────┤
│ Chunk 2  │
├──────────┤
│ Chunk 3  │
├──────────┤
│   ...    │
└──────────┘
```

Each chunk can then be converted into an embedding and stored in a vector database.

The basic RAG pipeline is:

```text
Documents
    ↓
Chunking
    ↓
Chunks
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retrieval
    ↓
Relevant Chunks
    ↓
LLM
    ↓
Answer
```

## Why is chunking important?

Chunking has a major impact on retrieval quality.

If chunks are too large:

- They may contain too much irrelevant information.
- The retrieved context can become noisy.
- More tokens are sent to the LLM.

If chunks are too small:

- Important context can be lost.
- A single idea may be split across multiple chunks.
- Retrieval may return incomplete information.

Therefore, **there is no universally perfect chunk size**.

The best strategy depends on the type and structure of the data.

---

# 2. Character Text Splitting

## Idea

`CharacterTextSplitter` is one of the simplest chunking strategies.

It splits a document based on a specified separator and tries to respect a maximum chunk size.

Example:

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(documents)
```

The main parameters are:

### `chunk_size`

The approximate maximum size of a chunk.

```python
chunk_size=800
```

This does **not necessarily mean every chunk will contain exactly 800 characters**.

### `chunk_overlap`

Controls how much content is shared between consecutive chunks.

For example:

```text
Chunk 1:
A B C D E F G H

Chunk 2:
                G H I J K L M N
                ↑
             overlap
```

If:

```python
chunk_overlap=2
```

then the last 2 units of one chunk may be reused at the beginning of the next chunk.

---

## Advantages

- Very simple.
- Easy to understand.
- Fast.
- Good for learning the basics of chunking.

## Disadvantages

- Does not understand the semantic structure of the document.
- Can split sentences or ideas in undesirable places.
- Not always appropriate for complex documents.

---

# 3. Recursive Character Text Splitting

## Idea

`RecursiveCharacterTextSplitter` tries to split text more intelligently by using a hierarchy of separators.

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
```

Instead of simply cutting text wherever necessary, it tries different separators.

Conceptually:

```text
Paragraph
    ↓
Sentence
    ↓
Word
    ↓
Character
```

The idea is:

> Keep larger meaningful units together whenever possible, and only split further when necessary.

---

## Example

Suppose we have:

```text
Paragraph 1.

Sentence A.
Sentence B.
Sentence C.

Paragraph 2.
```

A recursive splitter may try:

```text
1. Split by paragraphs
2. If chunks are still too large:
       split by sentences
3. If still too large:
       split by smaller separators
```

This usually produces more natural chunks than a basic character splitter.

---

## Advantages

- Better preservation of document structure.
- Usually better than simple character splitting for normal text.
- Very common in RAG systems.
- Easy to configure.

## Disadvantages

- Still fundamentally based on text structure rather than meaning.
- Does not truly understand the semantic meaning of the text.
- Chunk quality still depends on the chosen separators and chunk size.

---

# 4. Semantic Chunking

## Idea

Semantic chunking tries to divide a document based on **meaning**, rather than simply character count or separators.

Instead of asking:

> "Where should I cut the text?"

it tries to determine:

> "Where does the meaning of the text change?"

This usually involves embeddings.

Conceptually:

```text
Sentence 1 ────────┐
Sentence 2         │ Similar meaning
Sentence 3 ────────┘
                    ↓
                  Chunk 1

Sentence 4 ────────┐
Sentence 5         │ Different topic
                    ↓
                  Chunk 2
```

A semantic chunking system can compare the semantic similarity between neighboring pieces of text.

When the semantic difference becomes large enough, a new chunk can be created.

---

## Typical pipeline

```text
Document
   ↓
Split into smaller units
   ↓
Generate embeddings
   ↓
Compare neighboring units
   ↓
Detect semantic boundaries
   ↓
Create chunks
```

---

## Advantages

- Chunks can better represent complete ideas.
- Can work well when document structure is inconsistent.
- More meaning-aware than fixed-size splitting.

## Disadvantages

- More computationally expensive.
- Requires an embedding model.
- More parameters and complexity.
- Not automatically better for every dataset.

---

# 5. Comparison

| Strategy | Based on | Complexity | Semantic awareness | Typical use |
|---|---|---:|---:|---|
| CharacterTextSplitter | Characters/separators | Low | Low | Simple text |
| RecursiveCharacterTextSplitter | Hierarchical separators | Low/Medium | Low/Medium | General RAG |
| Semantic Chunking | Meaning/embeddings | Higher | High | Complex or unstructured text |

A useful mental model:

```text
                    Semantic awareness
                           ↑
                           │       Semantic
                           │       Chunking
                           │
                           │
                           │ Recursive
                           │
                           │
                           │ Character
                           └──────────────────→
                              Complexity
```

---

# 6. Chunk Overlap

Chunk overlap is independent of the chunking strategy.

It is used to preserve context between neighboring chunks.

Without overlap:

```text
Chunk 1:
"The patient should take the medication after"

Chunk 2:
"eating breakfast every morning."
```

The sentence is split between two chunks.

With overlap:

```text
Chunk 1:
"The patient should take the medication after eating breakfast"

Chunk 2:
"after eating breakfast every morning."
```

The overlap gives the retrieval system more context.

However, too much overlap can:

- Increase storage requirements.
- Create redundant information.
- Increase the amount of context sent to the LLM.

So overlap should be chosen carefully.

---

# 7. Chunking is a Retrieval Problem

A common mistake is to think:

> "The goal of chunking is simply to create chunks of a certain size."

The real objective is:

> **Create chunks that are useful units of information for retrieval.**

For example, imagine a document containing:

```text
Company information
Financial information
Employee information
Technical information
```

A chunk containing all four topics might technically be valid, but it may be bad for retrieval because it contains a lot of irrelevant information.

A good chunk should ideally contain a coherent piece of information.

---

# 8. There is No Universal Best Chunking Strategy

The best strategy depends on the data.

For example:

### Simple text

```text
CharacterTextSplitter
```

may be enough.

### Normal documents

```text
RecursiveCharacterTextSplitter
```

is often a strong baseline.

### Complex / poorly structured documents

```text
Semantic Chunking
```

may be useful.

Other approaches can also be appropriate depending on the data:

- Markdown-aware chunking
- HTML-aware chunking
- Code-aware chunking
- Table-aware chunking
- Document-structure-based chunking
- Parent-child chunking

These will be studied later.

---

# 9. Important Parameters

When experimenting with chunking, pay attention to:

### Chunk size

```python
chunk_size=800
```

Controls how much text can be placed in a chunk.

### Chunk overlap

```python
chunk_overlap=100
```

Controls how much neighboring chunks can share.

### Separators

For recursive splitting, separators determine where the splitter tries to create boundaries.

### Embedding model

For semantic chunking, the embedding model affects how semantic similarity is measured.

---

# 10. How to Evaluate a Chunking Strategy

A chunking strategy should not be judged only by looking at the chunks.

The real question is:

> **Does it improve retrieval?**

A useful experiment is:

```text
Same documents
       ↓
 ┌─────┴─────┐
 ↓           ↓
Strategy A   Strategy B
 ↓           ↓
Retrieval    Retrieval
 ↓           ↓
Compare results
```

For example:

```text
Question:
"What are the side effects of medication X?"

Strategy A → retrieves unrelated chunks
Strategy B → retrieves the correct section
```

Strategy B is probably better for this dataset.

Useful evaluation ideas include:

- Are the relevant chunks retrieved?
- Are irrelevant chunks retrieved?
- Is the answer supported by the retrieved context?
- How much context is required?
- How many tokens are sent to the LLM?

---

