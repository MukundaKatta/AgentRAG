# AgentRAG

AgentRAG is an early-stage project for building modular retrieval-augmented generation pipelines for AI agents, with a focus on chunking, embedding, retrieval, reranking, and pluggable vector-store workflows.

RAG systems often start simple and become messy quickly. Chunking strategies, embedding providers, retrievers, rerankers, and vector stores all evolve at different speeds, especially once they are used inside agent workflows rather than a single chatbot. AgentRAG exists to make that stack easier to reason about and easier to compose.

## Why AgentRAG

The goal is to make RAG pipelines more modular and more practical for agent systems.

That means:

- separating retrieval stages cleanly
- supporting experimentation without forcing a full framework
- making it easier to compare chunking, embedding, and retrieval choices
- building toward reusable patterns for agent-oriented retrieval workflows

## Current Status

AgentRAG is now at the first runnable implementation stage.

The repository includes a small Python package with core RAG pipeline interfaces plus an in-memory reference pipeline for indexing and retrieval. The goal is still to keep the surface area small while making the architecture concrete enough to extend.

## Project Direction

AgentRAG is intended to explore and eventually support:

- pluggable chunking strategies
- embedding-provider abstraction
- interchangeable vector stores
- reranking and retrieval-stage composition
- retrieval patterns designed for agentic workflows instead of one-shot search

## What I Want This Repo To Become

Over time, AgentRAG should become a practical home for:

- reusable retrieval pipeline building blocks
- experiments that compare retrieval strategies side by side
- examples for agent-aware context assembly
- cleaner interfaces between indexing, retrieval, and reranking
- developer-friendly patterns for production RAG systems

## Why This Matters

RAG is no longer just about search quality. In agent systems, retrieval affects planning, tool use, context quality, and overall reliability. A modular retrieval layer makes it easier to improve those systems without rewriting everything around them.

## Roadmap

Near-term priorities include:

- documenting integration patterns for agent runtimes
- building toward more realistic retrieval experiments

## Reference Implementation

The first working implementation now includes:

- `Document`, `Chunk`, `IndexedChunk`, and `SearchResult` models
- protocol-style interfaces for `Chunker`, `Embedder`, `VectorStore`, and `Retriever`
- `SimpleWordChunker` for local chunking experiments
- `BagOfWordsEmbedder` for deterministic local embeddings
- `InMemoryVectorStore` plus `InMemoryRAGPipeline` for end-to-end indexing and retrieval

### Example

```python
from agentrag import Document, InMemoryRAGPipeline

pipeline = InMemoryRAGPipeline()
pipeline.index_documents(
    [
        Document(
            id="doc-1",
            text="Ganapathi homam is performed before major life events to remove obstacles.",
        )
    ]
)

response = pipeline.retrieve("Which ritual helps remove obstacles?", top_k=1)
for result in response.results:
    print(result.chunk.document_id, result.score, result.chunk.text)
```

## Contributing

Contributions, ideas, and feedback are welcome, especially around:

- chunking strategies
- embeddings and retrieval abstractions
- reranking patterns
- vector-store integration design
- documentation and examples

## Project Structure

```text
AgentRAG/
├── README.md
├── pyproject.toml
├── src/agentrag/
└── tests/
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```
