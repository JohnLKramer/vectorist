from pathlib import Path

import pytest

from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.ir.models import CodeBlockNode, MediaNode, NodeType, ParagraphNode, SectionNode, TableNode


def _reachable_ids(doc) -> set[str]:
    seen: set[str] = set()
    stack = [doc.root_id]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(doc.nodes[nid].children_ids)
    return seen


def assert_all_nodes_reachable(doc) -> None:
    assert _reachable_ids(doc) == set(doc.nodes.keys())


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


def test_pooling_all_nodes_reachable_from_root(pooling_md: str):
    doc = MarkdownAdapter().parse("docs/pooling.md", pooling_md)
    assert_all_nodes_reachable(doc)


def test_two_top_level_h2_sections_both_reachable():
    content = "## Installation\n\nSteps.\n\n## Usage\n\nExamples.\n"
    doc = MarkdownAdapter().parse("guide.md", content)
    assert_all_nodes_reachable(doc)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.title == "guide.md"
    h2_titles = [
        doc.nodes[cid].title
        for cid in root.children_ids
        if doc.nodes[cid].node_type == NodeType.SECTION
    ]
    assert h2_titles == ["Installation", "Usage"]


def test_h3_then_h2_promotes_and_stays_reachable():
    content = "### Deep\n\nDetail.\n\n## Higher\n\nOverview.\n"
    doc = MarkdownAdapter().parse("nested.md", content)
    assert_all_nodes_reachable(doc)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.title == "nested.md"
    top_sections = [
        doc.nodes[cid]
        for cid in root.children_ids
        if doc.nodes[cid].node_type == NodeType.SECTION
    ]
    assert {s.title for s in top_sections} == {"Deep", "Higher"}


def test_two_top_level_h1_sections_both_reachable():
    content = "# First\n\nIntro.\n\n# Second\n\nMore.\n"
    doc = MarkdownAdapter().parse("two-h1.md", content)
    assert_all_nodes_reachable(doc)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.title == "two-h1.md"
    h1_titles = [
        doc.nodes[cid].title
        for cid in root.children_ids
        if doc.nodes[cid].node_type == NodeType.SECTION
    ]
    assert h1_titles == ["First", "Second"]


def test_paragraph_before_first_heading_under_document_root():
    content = "Preamble paragraph.\n\n# Title\n\nBody.\n"
    doc = MarkdownAdapter().parse("preamble.md", content)
    assert_all_nodes_reachable(doc)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.title == "preamble.md"
    child_types = [doc.nodes[cid].node_type for cid in root.children_ids]
    assert NodeType.PARAGRAPH in child_types
    assert NodeType.SECTION in child_types
    preamble = next(
        doc.nodes[cid]
        for cid in root.children_ids
        if doc.nodes[cid].node_type == NodeType.PARAGRAPH
    )
    assert isinstance(preamble, ParagraphNode)
    assert "Preamble" in preamble.raw_content


def test_headingless_document_all_blocks_under_synthetic_root():
    content = "Only prose.\n\n```\ncode\n```\n"
    doc = MarkdownAdapter().parse("plain.md", content)
    assert_all_nodes_reachable(doc)
    root = doc.nodes[doc.root_id]
    assert isinstance(root, SectionNode)
    assert root.title == "plain.md"
    assert len(root.children_ids) >= 2


def test_mixed_paragraph_splits_image_to_media_node():
    content = "See ![alt text](images/diagram.png) here.\n"
    doc = MarkdownAdapter().parse("mixed.md", content)
    assert_all_nodes_reachable(doc)
    media = [n for n in doc.nodes.values() if n.node_type == NodeType.MEDIA]
    assert len(media) == 1
    assert isinstance(media[0], MediaNode)
    assert media[0].alt_text == "alt text"
    assert media[0].enrichment_status == "PENDING"
    paragraphs = [n for n in doc.nodes.values() if n.node_type == NodeType.PARAGRAPH]
    assert len(paragraphs) == 2
    texts = sorted(p.raw_content.strip() for p in paragraphs)
    assert texts == ["See", "here."]
