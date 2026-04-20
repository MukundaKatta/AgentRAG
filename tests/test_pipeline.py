from agentrag import Document, InMemoryRAGPipeline
from agentrag.pipeline import BagOfWordsEmbedder, SimpleWordChunker


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
