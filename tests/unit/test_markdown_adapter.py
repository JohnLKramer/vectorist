from pathlib import Path

import pytest

from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.ir.models import CodeBlockNode, MediaNode, NodeType, SectionNode, TableNode


@pytest.fixture
def pooling_md(fixtures_dir: Path) -> str:
    return (fixtures_dir / "pooling.md").read_text(encoding="utf-8")


@pytest.fixture
def untagged_code_md(fixtures_dir: Path) -> str:
    return (fixtures_dir / "untagged_code.md").read_text(encoding="utf-8")


def test_section_hierarchy_reflects_headings(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.level == 1
    assert root.title == "Connection Pooling"

    child_sections = [
        doc.nodes[cid]
        for cid in root.children_ids
        if doc.nodes[cid].node_type == NodeType.SECTION
    ]
    assert len(child_sections) == 1
    overview = child_sections[0]
    assert isinstance(overview, SectionNode)
    assert overview.level == 2
    assert overview.title == "Overview"
    assert overview.parent_id == root.node_id


def test_fenced_python_block_language_from_fence(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    code_nodes = [n for n in doc.nodes.values() if n.node_type == NodeType.CODE_BLOCK]
    assert len(code_nodes) == 1
    block = code_nodes[0]
    assert isinstance(block, CodeBlockNode)
    assert block.language == "python"
    assert block.detection_source == "fence"


def test_untagged_fence_is_unknown_language(untagged_code_md: str):
    doc = MarkdownAdapter().parse("samples/untagged_code.md", untagged_code_md)
    code_nodes = [n for n in doc.nodes.values() if n.node_type == NodeType.CODE_BLOCK]
    assert len(code_nodes) == 1
    block = code_nodes[0]
    assert isinstance(block, CodeBlockNode)
    assert block.language == "unknown"


def test_table_headers_and_rows(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    tables = [n for n in doc.nodes.values() if n.node_type == NodeType.TABLE]
    assert len(tables) == 1
    table = tables[0]
    assert isinstance(table, TableNode)
    assert table.headers == ["Strategy", "Description"]
    assert table.rows == [
        ["eager", "Pre-warm connections at startup"],
        ["lazy", "Create connections on demand"],
    ]


def test_image_becomes_media_node_with_pending_enrichment(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    media_nodes = [n for n in doc.nodes.values() if n.node_type == NodeType.MEDIA]
    assert len(media_nodes) == 1
    media = media_nodes[0]
    assert isinstance(media, MediaNode)
    assert media.alt_text == "Pooling diagram"
    assert media.enrichment_status == "PENDING"


def test_nodes_have_line_range(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    for node in doc.nodes.values():
        start, end = node.line_range
        assert isinstance(start, int) and isinstance(end, int)
        assert start >= 1
        assert end >= start
