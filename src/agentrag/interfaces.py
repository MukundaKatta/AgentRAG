"""Core RAG pipeline interfaces and shared models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class IndexedChunk(Chunk):
    embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings."""


class VectorStore(Protocol):
    def upsert(self, chunks: list[IndexedChunk]) -> None:
        """Persist indexed chunks."""

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Return the nearest chunks for the query embedding."""


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve the most relevant chunks for a query."""
