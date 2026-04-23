"""Reference in-memory indexing and retrieval pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from agentrag.interfaces import Chunk, Chunker, Document, Embedder, IndexedChunk, SearchResult, VectorStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class SimpleWordChunker:
    """Chunk documents into fixed-size word windows."""

    def __init__(self, chunk_size: int = 80, overlap: int = 10) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.chunk_size]
            if not window:
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{index}",
                    document_id=document.id,
                    text=" ".join(window),
                    metadata={
                        **document.metadata,
                        "chunk_index": str(index),
                        "start_word": str(start),
                    },
                )
            )
        return chunks


class BagOfWordsEmbedder:
    """Tiny deterministic embedder for local tests and reference workflows."""

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            for token in _tokenize(text):
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)

        vectors: list[list[float]] = []
        size = len(self._vocabulary)
        for text in texts:
            vector = [0.0] * size
            counts = Counter(_tokenize(text))
            for token, count in counts.items():
                vector[self._vocabulary[token]] = float(count)
            vectors.append(vector)
        return vectors


class InMemoryVectorStore:
    """Naive cosine-similarity store for reference retrieval workflows."""

    def __init__(self) -> None:
        self._chunks: list[IndexedChunk] = []

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        by_id = {chunk.id: chunk for chunk in self._chunks}
        for chunk in chunks:
            by_id[chunk.id] = chunk
        self._chunks = list(by_id.values())

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        scored = [
            SearchResult(
                chunk=Chunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                ),
                score=_cosine_similarity(query_embedding, chunk.embedding),
            )
            for chunk in self._chunks
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


@dataclass(slots=True)
class RetrievalResponse:
    query: str
    results: list[SearchResult]


class InMemoryRAGPipeline:
    """Reference RAG pipeline that keeps indexing and retrieval interfaces separate."""

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.chunker = chunker or SimpleWordChunker()
        self.embedder = embedder or BagOfWordsEmbedder()
        self.vector_store = vector_store or InMemoryVectorStore()

    def index_documents(self, documents: list[Document]) -> list[IndexedChunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunker.chunk(document))

        embeddings = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        indexed = [
            IndexedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.vector_store.upsert(indexed)
        return indexed

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        query_embedding = self.embedder.embed([query])[0]
        return RetrievalResponse(query=query, results=self.vector_store.search(query_embedding, top_k))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    size = max(len(left), len(right))
    if size == 0:
        return 0.0

    padded_left = left + [0.0] * (size - len(left))
    padded_right = right + [0.0] * (size - len(right))

    numerator = sum(a * b for a, b in zip(padded_left, padded_right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in padded_left))
    right_norm = math.sqrt(sum(b * b for b in padded_right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
