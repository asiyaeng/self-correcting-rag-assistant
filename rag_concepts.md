# Retrieval-Augmented Generation (RAG) — Concepts & Patterns

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant external documents at inference time and including them as context. This grounds the LLM's answers in real data, reducing hallucinations and enabling knowledge beyond the training cutoff.

## Core RAG Pipeline

### 1. Indexing (Offline)
```
Documents → Chunking → Embedding → Vector Store
```
- Split documents into semantically coherent chunks
- Generate embedding vectors for each chunk
- Store vectors with metadata in a vector database

### 2. Retrieval (Online)
```
Query → Embed Query → Similarity Search → Top-K Chunks
```
- Convert the user's question to an embedding
- Find the most similar chunks using cosine/dot-product similarity
- Return the top-k results with scores

### 3. Generation (Online)
```
Question + Context Chunks → LLM → Answer with Citations
```
- Inject retrieved chunks into the prompt as context
- Instruct the LLM to answer using only the provided context
- Include citations pointing back to source documents

## Chunking Strategies

### Fixed-size Chunking
Split by character count with overlap. Simple but can break sentences.
```
chunk_size=512, chunk_overlap=64
```

### Recursive Character Splitting
Tries to split on natural boundaries: paragraphs → sentences → words.
Best general-purpose strategy for prose and technical docs.

### Semantic Chunking
Groups sentences by semantic similarity. More expensive but produces coherent chunks.

### Markdown/Code-aware Splitting
Splits on headers (`## Section`) for Markdown or function/class boundaries for code.

### Choosing Chunk Size
- **Small (128–256 tokens)**: High precision, less context per chunk
- **Medium (512–1024 tokens)**: Best for most technical documentation
- **Large (2048+ tokens)**: More context but noisier retrieval

## Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `text-embedding-3-small` (OpenAI) | 1536 | Fast, cheap, good quality |
| `text-embedding-3-large` (OpenAI) | 3072 | Best quality, more expensive |
| `all-MiniLM-L6-v2` (HuggingFace) | 384 | Free, runs locally, surprisingly good |
| `bge-large-en-v1.5` (BAAI) | 1024 | Top open-source model |

## Vector Stores

### ChromaDB
- Open-source, runs locally
- Persistent storage to disk
- Python-native, easy to use
- Good for development and small-to-medium corpora

### FAISS (Facebook AI Similarity Search)
- Extremely fast, production-ready
- Flat (exact) or quantized (approximate) indexes
- No built-in persistence (must save/load manually)
- Best for large-scale retrieval

### Pinecone / Weaviate / Qdrant
- Managed cloud services
- Built-in scalability and replication
- Suitable for production deployments

## Advanced RAG Patterns

### Corrective RAG (CRAG)
Adds a document grading step after retrieval:
1. Retrieve documents
2. Grade each document as relevant/irrelevant
3. If all irrelevant → rewrite query and re-retrieve
4. If some relevant → filter and generate
5. Optional: supplement with web search if knowledge is insufficient

### Self-RAG
The LLM itself decides whether to retrieve, what to retrieve, and checks its own output:
- **Retrieve token**: Should I retrieve for this query?
- **ISREL**: Is this document relevant?
- **ISSUP**: Is the generated sentence supported by the document?
- **ISUSE**: Is the response useful?

### Adaptive RAG
Routes queries to different retrieval strategies based on query type:
- Simple factual queries → direct answer (no retrieval)
- Technical how-to → retrieve from docs
- Current events → web search
- Complex multi-hop → agent-style iterative retrieval

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, embed it, then use it for retrieval instead of the raw question. Often improves recall for complex queries.

```python
hyp_answer = llm.invoke(f"Write a short document that answers: {question}")
docs = vectorstore.similarity_search(hyp_answer)
```

### Multi-Query RAG
Generate multiple rephrased versions of the query, retrieve for each, and deduplicate:
```python
queries = llm.invoke(f"Generate 3 search queries for: {question}")
all_docs = [retrieve(q) for q in queries]
unique_docs = deduplicate(all_docs)
```

## Evaluation Metrics

### Retrieval Quality
- **Recall@K**: Fraction of relevant docs in top-K results
- **Precision@K**: Fraction of top-K results that are relevant
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result

### Generation Quality
- **Faithfulness**: Is the answer supported by the context? (no hallucinations)
- **Answer Relevance**: Does the answer address the question?
- **Context Relevance**: Are the retrieved chunks actually relevant?

### RAG-specific Frameworks
- **RAGAS**: Automated evaluation using LLM-as-a-judge for faithfulness and relevance
- **TruLens**: Evaluation and monitoring for LLM apps
- **DeepEval**: Unit-test-style framework for LLM evaluation

## Hallucination Detection
After generating an answer, verify it against the source context:

```python
verification_prompt = """
Given this answer and the source context, check if every factual claim 
in the answer is supported by the context.
Answer: {answer}
Context: {context}
Return JSON: {"grounded": true/false, "unsupported_claims": []}
"""
```

## Common Pitfalls
1. **Chunk boundaries break context**: Use overlap and natural-boundary splitters
2. **Query-document mismatch**: Questions are short; documents are long → use HyDE
3. **Irrelevant retrieval**: Add document grading and rewrite loop
4. **Hallucination on edge cases**: Add hallucination check node
5. **Outdated knowledge**: Implement web search fallback for current events
6. **No citation tracking**: Always store source metadata with each chunk
