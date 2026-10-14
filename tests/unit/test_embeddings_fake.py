import math

from vectorist.embeddings.fake import FakeEmbeddingProvider

EXPECTED_DIM = 384


def test_same_text_produces_same_document_vector():
    provider = FakeEmbeddingProvider()
    text = "vectorist chunks markdown for retrieval"
    first = provider.embed_documents([text])[0]
    second = provider.embed_documents([text])[0]
    assert first == second


def test_document_vectors_have_expected_dimension():
    provider = FakeEmbeddingProvider()
    vectors = provider.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == EXPECTED_DIM


def test_vectors_are_unit_length():
    provider = FakeEmbeddingProvider()
    vector = provider.embed_documents(["normalize me"])[0]
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_different_texts_produce_different_vectors():
    provider = FakeEmbeddingProvider()
    a, b = provider.embed_documents(["alpha", "beta"])
    assert a != b


def test_embed_query_returns_unit_vector_of_expected_dimension():
    provider = FakeEmbeddingProvider()
    vector = provider.embed_query("what is vectorist?")
    assert len(vector) == EXPECTED_DIM
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_embed_query_is_deterministic():
    provider = FakeEmbeddingProvider()
    text = "search query"
    assert provider.embed_query(text) == provider.embed_query(text)


def test_provider_reports_dimensions_and_model_id():
    provider = FakeEmbeddingProvider()
    assert provider.dimensions == EXPECTED_DIM
    assert provider.model_id == "fake/BAAI/bge-small-en-v1.5"
