import pytest

from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.enrichment.apply import apply_enrichments
from vectorist.ir.models import EnrichmentStatus, IRDocument, MediaNode

_CONTENT = "# T\n\n![Diagram](a.png)\n\n![x](b.png)\n\n![](c.png)"


def _document(content: str = _CONTENT) -> IRDocument:
    doc = MarkdownAdapter().parse("u", content)
    for node in doc.nodes.values():
        if isinstance(node, MediaNode) and node.metadata["ref"] == "c.png":
            node.enrichment_status = EnrichmentStatus.SKIPPED
            node.metadata["skip_reason"] = "manual"
    return doc


def _media(doc: IRDocument, ref: str) -> list[MediaNode]:
    return [
        node
        for node in doc.nodes.values()
        if isinstance(node, MediaNode) and node.metadata["ref"] == ref
    ]


def _one(doc: IRDocument, ref: str) -> MediaNode:
    nodes = _media(doc, ref)
    assert len(nodes) == 1
    return nodes[0]


def test_linked_text_becomes_caption_and_raw_content():
    result = apply_enrichments(
        _document(), {"a.png": "h1"}, {"h1": "boxes and arrows"}, {}
    )

    node = _one(result, "a.png")
    assert node.caption == "boxes and arrows"
    assert node.raw_content == "Diagram\nboxes and arrows"
    assert node.enrichment_status == "DONE"
    assert node.asset_uri_or_hash == "sha256:h1"


def test_no_alt_uses_text_only():
    doc = MarkdownAdapter().parse("u", "# T\n\n![](d.png)")

    result = apply_enrichments(doc, {"d.png": "h2"}, {"h2": "a chart"}, {})

    node = _one(result, "d.png")
    assert node.alt_text == ""
    assert node.raw_content == "a chart"
    assert node.enrichment_status == "DONE"


def test_linked_without_text_stays_pending_unless_failed():
    failed = apply_enrichments(_document(), {"a.png": "h1"}, {}, {"h1": "FAILED"})
    pending = apply_enrichments(_document(), {"a.png": "h1"}, {}, {})

    assert _one(failed, "a.png").enrichment_status == "FAILED"
    assert _one(failed, "a.png").asset_uri_or_hash == "sha256:h1"
    pending_node = _one(pending, "a.png")
    assert pending_node.enrichment_status == "PENDING"
    assert pending_node.asset_uri_or_hash == "sha256:h1"


def test_unlinked_and_skipped_nodes_untouched():
    doc = _document()
    original_b = _one(doc, "b.png").model_copy(deep=True)
    original_c = _one(doc, "c.png").model_copy(deep=True)

    result = apply_enrichments(
        doc,
        {"a.png": "h1", "c.png": "h3"},
        {"h1": "boxes and arrows", "h3": "should not appear"},
        {},
    )

    b_node = _one(result, "b.png")
    assert b_node.enrichment_status == "PENDING"
    assert b_node == original_b
    c_node = _one(result, "c.png")
    assert c_node.enrichment_status == "SKIPPED"
    assert c_node == original_c


def test_apply_is_idempotent_and_does_not_mutate_input():
    doc = _document()
    before = doc.model_copy(deep=True)
    links = {"a.png": "h1", "b.png": "h2"}
    texts = {"h1": "boxes and arrows"}
    statuses = {"h2": "FAILED"}

    once = apply_enrichments(doc, links, texts, statuses)
    twice = apply_enrichments(once, links, texts, statuses)

    assert twice == once
    assert doc == before
    assert once != before


def test_duplicate_src_nodes_all_update():
    doc = MarkdownAdapter().parse("u", "# T\n\n![one](x.png)\n\n![two](x.png)")

    result = apply_enrichments(doc, {"x.png": "h"}, {"h": "a red circle"}, {})

    nodes = _media(result, "x.png")
    assert len(nodes) == 2
    assert [node.raw_content for node in nodes] == [
        "one\na red circle",
        "two\na red circle",
    ]
    assert all(node.caption == "a red circle" for node in nodes)
    assert all(node.enrichment_status == "DONE" for node in nodes)


def test_linked_without_text_resets_stale_text():
    doc =MarkdownAdapter().parse("u", "# T\n\n![Diagram](a.png)\n\n![](d.png)")
    done = apply_enrichments(
        doc, {"a.png": "h1", "d.png": "h2"}, {"h1": "old", "h2": "old chart"}, {}
    )
    assert _one(done, "a.png").caption == "old"

    pending = apply_enrichments(done, {"a.png": "h1", "d.png": "h2"}, {}, {})
    failed = apply_enrichments(
        done, {"a.png": "h1", "d.png": "h2"}, {}, {"h1": "FAILED", "h2": "FAILED"}
    )

    for result, status in ((pending, "PENDING"), (failed, "FAILED")):
        a_node = _one(result, "a.png")
        assert a_node.caption == ""
        assert a_node.raw_content == "Diagram"
        assert a_node.enrichment_status == status
        d_node = _one(result, "d.png")
        assert d_node.caption == ""
        assert d_node.raw_content == "d.png"
        assert d_node.enrichment_status == status


def test_legacy_node_without_ref_raises_explicit_error():
    doc = MarkdownAdapter().parse("u", "# T\n\n![a](x.png)")
    for node in doc.nodes.values():
        if isinstance(node, MediaNode):
            node.metadata.pop("ref")

    with pytest.raises(ValueError, match="predates image refs"):
        apply_enrichments(doc, {"x.png": "h"}, {"h": "t"}, {})
