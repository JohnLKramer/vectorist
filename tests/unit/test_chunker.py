from __future__ import annotations

import pytest

from vectorist.chunking.chunker import Chunk, Chunker, _context_header
from vectorist.chunking.tokenizer import estimate_tokens
from vectorist.ir.models import (
    CodeBlockNode,
    IRDocument,
    NodeType,
    ParagraphNode,
    SectionNode,
)


def _section(
    node_id: str,
    *,
    parent_id: str | None,
    children_ids: list[str],
    breadcrumbs: list[str],
    level: int,
    title: str,
    line_range: tuple[int, int] = (1, 1),
) -> SectionNode:
    return SectionNode(
        node_id=node_id,
        node_type=NodeType.SECTION,
        parent_id=parent_id,
        children_ids=children_ids,
        breadcrumbs=breadcrumbs,
        line_range=line_range,
        raw_content=title,
        metadata={},
        level=level,
        title=title,
    )


def _paragraph(
    node_id: str,
    *,
    parent_id: str,
    text: str,
    breadcrumbs: list[str],
    line_range: tuple[int, int],
) -> ParagraphNode:
    return ParagraphNode(
        node_id=node_id,
        node_type=NodeType.PARAGRAPH,
        parent_id=parent_id,
        children_ids=[],
        breadcrumbs=breadcrumbs,
        line_range=line_range,
        raw_content=text,
        metadata={},
    )


def _code(
    node_id: str,
    *,
    parent_id: str,
    text: str,
    breadcrumbs: list[str],
    line_range: tuple[int, int],
) -> CodeBlockNode:
    return CodeBlockNode(
        node_id=node_id,
        node_type=NodeType.CODE_BLOCK,
        parent_id=parent_id,
        children_ids=[],
        breadcrumbs=breadcrumbs,
        line_range=line_range,
        raw_content=text,
        metadata={},
        language="python",
        detection_source="fence",
    )


def test_estimate_tokens_whitespace_word_approximation():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n\t  ") == 0
    assert estimate_tokens("one two three") == 3
    assert estimate_tokens("  padded   words  ") == 2


def test_chunks_never_merge_across_sibling_h2_sections():
    crumbs_a = ["doc.md", "Alpha"]
    crumbs_b = ["doc.md", "Beta"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["sec-a", "sec-b"],
        breadcrumbs=["doc.md"],
        level=1,
        title="doc.md",
    )
    sec_a = _section(
        "sec-a",
        parent_id="root",
        children_ids=["p-a"],
        breadcrumbs=crumbs_a,
        level=2,
        title="Alpha",
        line_range=(2, 5),
    )
    sec_b = _section(
        "sec-b",
        parent_id="root",
        children_ids=["p-b"],
        breadcrumbs=crumbs_b,
        level=2,
        title="Beta",
        line_range=(6, 9),
    )
    p_a = _paragraph(
        "p-a",
        parent_id="sec-a",
        text="Alpha paragraph.",
        breadcrumbs=crumbs_a,
        line_range=(3, 3),
    )
    p_b = _paragraph(
        "p-b",
        parent_id="sec-b",
        text="Beta paragraph.",
        breadcrumbs=crumbs_b,
        line_range=(7, 7),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={
            "root": root,
            "sec-a": sec_a,
            "sec-b": sec_b,
            "p-a": p_a,
            "p-b": p_b,
        },
    )

    chunks = Chunker(max_tokens=512).chunk_document(doc)
    assert len(chunks) == 2
    texts = [c.content_text for c in chunks]
    assert any("Alpha paragraph." in t and "Beta paragraph." not in t for t in texts)
    assert any("Beta paragraph." in t and "Alpha paragraph." not in t for t in texts)
    for chunk in chunks:
        assert chunk.content_text.startswith("[Context: ")


def test_packs_paragraph_and_following_code_in_same_h2_section():
    crumbs = ["guide.md", "Setup"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["sec"],
        breadcrumbs=["guide.md"],
        level=1,
        title="guide.md",
    )
    sec = _section(
        "sec",
        parent_id="root",
        children_ids=["p1", "code"],
        breadcrumbs=crumbs,
        level=2,
        title="Setup",
        line_range=(2, 10),
    )
    p1 = _paragraph(
        "p1",
        parent_id="sec",
        text="Run the installer.",
        breadcrumbs=crumbs,
        line_range=(3, 3),
    )
    code = _code(
        "code",
        parent_id="sec",
        text="pip install vectorist",
        breadcrumbs=crumbs,
        line_range=(5, 5),
    )
    doc = IRDocument(
        source_uri="guide.md",
        root_id="root",
        nodes={"root": root, "sec": sec, "p1": p1, "code": code},
    )

    chunks = Chunker(max_tokens=512).chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content_text.startswith(_context_header(crumbs))
    assert "Run the installer." in chunk.content_text
    assert "pip install vectorist" in chunk.content_text
    assert chunk.node_ids == ["p1", "code"]
    assert chunk.node_types == [NodeType.PARAGRAPH.value, NodeType.CODE_BLOCK.value]
    assert chunk.breadcrumbs == crumbs


def test_oversized_code_block_splits_into_multiple_chunks():
    crumbs = ["doc.md"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["code"],
        breadcrumbs=crumbs,
        level=1,
        title="doc.md",
    )
    big_code = "word " * 200
    code = _code(
        "code",
        parent_id="root",
        text=big_code,
        breadcrumbs=crumbs,
        line_range=(1, 1),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "code": code},
    )

    chunks = Chunker(max_tokens=50).chunk_document(doc)
    assert len(chunks) >= 2
    joined = " ".join(c.content_text for c in chunks)
    assert "word" in joined
    for chunk in chunks:
        assert chunk.content_text.startswith("[Context: ")


def test_oversized_code_single_token_emits_without_assertion_error():
    crumbs = ["doc.md"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["code"],
        breadcrumbs=crumbs,
        level=1,
        title="doc.md",
    )
    token = "x" * 500
    code = _code(
        "code",
        parent_id="root",
        text=token,
        breadcrumbs=crumbs,
        line_range=(1, 1),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "code": code},
    )

    header_budget = estimate_tokens(_context_header(crumbs))
    assert estimate_tokens(token) == 1
    assert header_budget >= estimate_tokens(token)
    chunks = Chunker(max_tokens=header_budget).chunk_document(doc)
    assert len(chunks) == 1
    assert token in chunks[0].content_text
    assert estimate_tokens(chunks[0].content_text) > header_budget


def test_oversized_paragraph_splits_on_blank_lines():
    crumbs = ["doc.md"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["p"],
        breadcrumbs=crumbs,
        level=1,
        title="doc.md",
    )
    body = "First block sentence.\n\nSecond block sentence."
    p = _paragraph(
        "p",
        parent_id="root",
        text=body,
        breadcrumbs=crumbs,
        line_range=(1, 3),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "p": p},
    )

    chunks = Chunker(max_tokens=6).chunk_document(doc)
    assert len(chunks) >= 2
    joined = "\n".join(c.content_text for c in chunks)
    assert "First block sentence." in joined
    assert "Second block sentence." in joined
    for chunk in chunks:
        assert chunk.content_text.startswith("[Context: ")


def test_chunk_indices_are_sequential():
    crumbs_a = ["doc.md", "A"]
    crumbs_b = ["doc.md", "B"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["sa", "sb"],
        breadcrumbs=["doc.md"],
        level=1,
        title="doc.md",
    )
    sa = _section(
        "sa",
        parent_id="root",
        children_ids=["pa"],
        breadcrumbs=crumbs_a,
        level=2,
        title="A",
    )
    sb = _section(
        "sb",
        parent_id="root",
        children_ids=["pb"],
        breadcrumbs=crumbs_b,
        level=2,
        title="B",
    )
    pa = _paragraph("pa", parent_id="sa", text="A text.", breadcrumbs=crumbs_a, line_range=(1, 1))
    pb = _paragraph("pb", parent_id="sb", text="B text.", breadcrumbs=crumbs_b, line_range=(2, 2))
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "sa": sa, "sb": sb, "pa": pa, "pb": pb},
    )

    chunks = Chunker(max_tokens=512).chunk_document(doc)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_sibling_h1_sections_are_isolated():
    crumbs_first = ["two.md", "First"]
    crumbs_second = ["two.md", "Second"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["h1-first", "h1-second"],
        breadcrumbs=["two.md"],
        level=1,
        title="two.md",
    )
    h1_first = _section(
        "h1-first",
        parent_id="root",
        children_ids=["p-first"],
        breadcrumbs=crumbs_first,
        level=1,
        title="First",
    )
    h1_second = _section(
        "h1-second",
        parent_id="root",
        children_ids=["p-second"],
        breadcrumbs=crumbs_second,
        level=1,
        title="Second",
    )
    p_first = _paragraph(
        "p-first",
        parent_id="h1-first",
        text="First intro.",
        breadcrumbs=crumbs_first,
        line_range=(2, 2),
    )
    p_second = _paragraph(
        "p-second",
        parent_id="h1-second",
        text="Second intro.",
        breadcrumbs=crumbs_second,
        line_range=(5, 5),
    )
    doc = IRDocument(
        source_uri="two.md",
        root_id="root",
        nodes={
            "root": root,
            "h1-first": h1_first,
            "h1-second": h1_second,
            "p-first": p_first,
            "p-second": p_second,
        },
    )

    chunks = Chunker(max_tokens=512).chunk_document(doc)
    assert len(chunks) == 2
    for chunk in chunks:
        assert not (
            "First intro." in chunk.content_text and "Second intro." in chunk.content_text
        )


def test_h3_siblings_under_h2_use_own_breadcrumb_headers():
    crumbs_h2 = ["doc.md", "Chapter"]
    crumbs_a = ["doc.md", "Chapter", "Part A"]
    crumbs_b = ["doc.md", "Chapter", "Part B"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["h2"],
        breadcrumbs=["doc.md"],
        level=1,
        title="doc.md",
    )
    h2 = _section(
        "h2",
        parent_id="root",
        children_ids=["h3-a", "h3-b"],
        breadcrumbs=crumbs_h2,
        level=2,
        title="Chapter",
    )
    h3_a = _section(
        "h3-a",
        parent_id="h2",
        children_ids=["pa"],
        breadcrumbs=crumbs_a,
        level=3,
        title="Part A",
    )
    h3_b = _section(
        "h3-b",
        parent_id="h2",
        children_ids=["pb"],
        breadcrumbs=crumbs_b,
        level=3,
        title="Part B",
    )
    pa = _paragraph(
        "pa",
        parent_id="h3-a",
        text="Alpha detail.",
        breadcrumbs=crumbs_a,
        line_range=(4, 4),
    )
    pb = _paragraph(
        "pb",
        parent_id="h3-b",
        text="Beta detail.",
        breadcrumbs=crumbs_b,
        line_range=(8, 8),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={
            "root": root,
            "h2": h2,
            "h3-a": h3_a,
            "h3-b": h3_b,
            "pa": pa,
            "pb": pb,
        },
    )

    chunks = Chunker(max_tokens=512).chunk_document(doc)
    assert len(chunks) == 2
    alpha_chunk = next(c for c in chunks if "Alpha detail." in c.content_text)
    beta_chunk = next(c for c in chunks if "Beta detail." in c.content_text)
    assert alpha_chunk.content_text.startswith(_context_header(crumbs_a))
    assert beta_chunk.content_text.startswith(_context_header(crumbs_b))
    assert alpha_chunk.breadcrumbs == crumbs_a
    assert beta_chunk.breadcrumbs == crumbs_b
    assert "Beta detail." not in alpha_chunk.content_text
    assert "Alpha detail." not in beta_chunk.content_text


def test_oversized_multi_block_paragraph_sentence_splits_first_block():
    crumbs = ["doc.md"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["p"],
        breadcrumbs=crumbs,
        level=1,
        title="doc.md",
    )
    body = "Sentence one. Sentence two.\n\nTail."
    p = _paragraph(
        "p",
        parent_id="root",
        text=body,
        breadcrumbs=crumbs,
        line_range=(1, 3),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "p": p},
    )

    chunks = Chunker(max_tokens=4).chunk_document(doc)
    bodies = [c.content_text.split("\n\n", 1)[1] for c in chunks]
    assert "Sentence one." in bodies
    assert "Sentence two." in bodies
    assert "Tail." in bodies
    assert not any("Sentence one. Sentence two." in b for b in bodies)


def test_split_paragraph_pieces_in_one_chunk_list_node_once():
    crumbs = ["doc.md"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["p"],
        breadcrumbs=crumbs,
        level=1,
        title="doc.md",
    )
    body = "Alpha beta.\n\nGamma delta."
    p = _paragraph(
        "p",
        parent_id="root",
        text=body,
        breadcrumbs=crumbs,
        line_range=(1, 3),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "p": p},
    )

    chunks = Chunker(max_tokens=20).chunk_document(doc)
    single = next(c for c in chunks if "Alpha beta." in c.content_text and "Gamma delta." in c.content_text)
    assert single.node_ids == ["p"]
    assert single.node_types == [NodeType.PARAGRAPH.value]


def test_every_chunk_respects_max_tokens_including_header():
    crumbs = ["doc.md", "Chapter", "Part A"]
    root = _section(
        "root",
        parent_id=None,
        children_ids=["h2"],
        breadcrumbs=["doc.md"],
        level=1,
        title="doc.md",
    )
    h2 = _section(
        "h2",
        parent_id="root",
        children_ids=["h3"],
        breadcrumbs=["doc.md", "Chapter"],
        level=2,
        title="Chapter",
    )
    h3 = _section(
        "h3",
        parent_id="h2",
        children_ids=["p1", "p2"],
        breadcrumbs=crumbs,
        level=3,
        title="Part A",
    )
    p1 = _paragraph("p1", parent_id="h3", text="Short.", breadcrumbs=crumbs, line_range=(1, 1))
    p2 = _paragraph(
        "p2",
        parent_id="h3",
        text="Another short line.",
        breadcrumbs=crumbs,
        line_range=(2, 2),
    )
    doc = IRDocument(
        source_uri="doc.md",
        root_id="root",
        nodes={"root": root, "h2": h2, "h3": h3, "p1": p1, "p2": p2},
    )

    max_tokens = 12
    chunks = Chunker(max_tokens=max_tokens).chunk_document(doc)
    for chunk in chunks:
        assert estimate_tokens(chunk.content_text) <= max_tokens
