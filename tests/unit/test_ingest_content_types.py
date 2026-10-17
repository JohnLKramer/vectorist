from vectorist.ir.models import IRDocument, ParagraphNode
from vectorist.pipeline.ingest import IngestService
from vectorist.cache.sqlite_cache import SqliteCache
from vectorist.embeddings.fake import FakeEmbeddingProvider
from vectorist.storage.sqlite_vec_store import SqliteVecStore


class _RecordingAdapter:
    def __init__(self):
        self.calls = []

    def parse(self, source_uri, content):
        self.calls.append((source_uri, content))
        node = ParagraphNode(
            node_id="n1", parent_id=None, children_ids=[], breadcrumbs=["x"],
            line_range=(1, 1), raw_content=content, metadata={},
        )
        return IRDocument(source_uri=source_uri, root_id="n1", nodes={"n1": node})


async def _service(tmp_path, **kwargs):
    cache = SqliteCache(tmp_path / "c.db")
    store = SqliteVecStore(tmp_path / "v.db")
    await cache.initialize()
    await store.initialize()
    return IngestService(cache, store, FakeEmbeddingProvider(), **kwargs)


async def test_unknown_type_rejected(tmp_path):
    service = await _service(tmp_path)
    result = await service.ingest_document("a", "x", content_type="application/pdf")
    assert result.status == "failed"
    assert "unsupported content type" in result.error["message"]


async def test_html_dispatches_to_html_adapter_entry(tmp_path):
    fake = _RecordingAdapter()
    service = await _service(tmp_path, adapters={"text/html": fake})
    result = await service.ingest_document("u", "<p>hi</p>", content_type="text/html")
    assert result.status == "indexed" and fake.calls == [("u", "<p>hi</p>")]


async def test_markdown_still_default(tmp_path):
    service = await _service(tmp_path)
    result = await service.ingest_document("a.md", "# T\n\ntext\n")
    assert result.status == "indexed"
