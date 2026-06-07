import pytest

from agentrag import Document, InMemoryRAGPipeline
from agentrag.pipeline import BagOfWordsEmbedder, SimpleWordChunker, _cosine_similarity


def test_index_documents_creates_chunked_reference_pipeline():
    pipeline = InMemoryRAGPipeline(chunker=SimpleWordChunker(chunk_size=4, overlap=1))

    indexed = pipeline.index_documents(
        [
            Document(
                id="doc-1",
                text="Ganapathi homam supports obstacle removal and auspicious beginnings for devotees.",
                metadata={"source": "ritual-guide"},
            )
        ]
    )

    assert len(indexed) >= 2
    assert indexed[0].document_id == "doc-1"
    assert indexed[0].metadata["source"] == "ritual-guide"
    assert indexed[0].embedding


def test_retrieve_returns_relevant_chunks_from_in_memory_store():
    pipeline = InMemoryRAGPipeline()
    pipeline.index_documents(
        [
            Document(
                id="doc-1",
                text="Ganapathi homam is performed before major life events to remove obstacles.",
            ),
            Document(
                id="doc-2",
                text="Satyanarayana vratam centers on gratitude, devotion, and family blessings.",
            ),
        ]
    )

    response = pipeline.retrieve("Which ritual helps remove obstacles?", top_k=1)

    assert response.query == "Which ritual helps remove obstacles?"
    assert len(response.results) == 1
    assert response.results[0].chunk.document_id == "doc-1"
    assert response.results[0].score > 0


def test_embedder_grows_vocabulary_without_breaking_previous_vectors():
    embedder = BagOfWordsEmbedder()
    first = embedder.embed(["alpha beta"])[0]
    second = embedder.embed(["beta gamma"])[0]

    assert len(second) >= len(first)
    assert first[0] > 0
    assert second[1] > 0


def test_chunker_does_not_emit_redundant_tail_chunk():
    # With chunk_size=4, overlap=1 (step=3) over 10 words the previous
    # implementation produced a trailing chunk holding only the final word,
    # which is fully contained in the prior window. Every chunk must add at
    # least one word not covered by its predecessor.
    chunker = SimpleWordChunker(chunk_size=4, overlap=1)
    words = " ".join(f"w{i}" for i in range(10))
    chunks = chunker.chunk(Document(id="doc-1", text=words))

    starts = [int(chunk.metadata["start_word"]) for chunk in chunks]
    for previous, current in zip(starts, starts[1:]):
        assert current + chunker.chunk_size > previous + chunker.chunk_size

    # The union of all chunks must still cover every word in the document.
    covered: set[int] = set()
    for start in starts:
        covered.update(range(start, min(start + chunker.chunk_size, 10)))
    assert covered == set(range(10))


def test_chunker_handles_short_and_empty_documents():
    chunker = SimpleWordChunker(chunk_size=4, overlap=1)
    assert chunker.chunk(Document(id="empty", text="   ")) == []

    short = chunker.chunk(Document(id="short", text="one two"))
    assert len(short) == 1
    assert short[0].text == "one two"


def test_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SimpleWordChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SimpleWordChunker(chunk_size=4, overlap=4)
    with pytest.raises(ValueError):
        SimpleWordChunker(chunk_size=4, overlap=-1)


def test_cosine_similarity_edge_cases():
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    # Different-length vectors are zero-padded before comparison.
    assert _cosine_similarity([1.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
