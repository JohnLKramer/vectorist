from __future__ import annotations

from pathlib import Path

import pytest

from vectorist.adapters.markdown import MarkdownAdapter
from vectorist.chunking.chunker import Chunker, _context_header
from vectorist.chunking.table_split import split_table
from vectorist.chunking.tokenizer import estimate_tokens
from vectorist.ir.models import IRDocument, NodeType, SectionNode, TableNode


@pytest.fixture
def wide_table_md(fixtures_dir: Path) -> str:
    return (fixtures_dir / "wide_table.md").read_text(encoding="utf-8")


def _table_from_markdown(md: str) -> TableNode:
    doc = MarkdownAdapter().parse("wide_table.md", md)
    tables = [n for n in doc.nodes.values() if n.node_type == NodeType.TABLE]
    assert len(tables) == 1
    table = tables[0]
    assert isinstance(table, TableNode)
    return table


def test_split_table_40_rows_sticky_headers_in_every_slice(wide_table_md: str):
    table = _table_from_markdown(wide_table_md)
    assert len(table.rows) == 40
    assert table.headers == ["Metric", "Value"]

    max_tokens = 80
    assert estimate_tokens(table.raw_content) > max_tokens

    slices = split_table(table, max_tokens=max_tokens)
    assert len(slices) >= 2
    for slice_text in slices:
        assert "Metric" in slice_text
        assert "Value" in slice_text
        assert "| --- |" in slice_text or "|---|" in slice_text.replace(" ", "")
        assert estimate_tokens(slice_text) <= max_tokens or len(slices) == 1


def test_continued_slices_use_exact_header_pattern(wide_table_md: str):
    table = _table_from_markdown(wide_table_md)
    table = table.model_copy(update={"caption": "Quarterly KPIs"})
    rows_per_group = 5
    slices = split_table(table, max_tokens=60, rows_per_group=rows_per_group)
    assert len(slices) == len(table.rows) // rows_per_group

    header_line = "| Metric | Value |"
    separator_line = "| --- | --- |"

    for slice_index, slice_text in enumerate(slices):
        lines = slice_text.split("\n")
        row_start = slice_index * rows_per_group + 1
        row_end = row_start + rows_per_group - 1
        expected_rows = table.rows[row_start - 1 : row_end]

        if slice_index == 0:
            assert lines[0] == "[Table: Quarterly KPIs]"
        else:
            assert lines[0] == (
                f"[Table: Quarterly KPIs (Continued - Rows {row_start} to {row_end})]"
            )

        assert lines[1] == header_line
        assert lines[2] == separator_line
        data_lines = lines[3:]
        assert len(data_lines) == rows_per_group
        for expected_row, data_line in zip(expected_rows, data_lines, strict=True):
            assert data_line == f"| {expected_row[0]} | {expected_row[1]} |"


def test_chunker_splits_oversized_table_into_multiple_chunks(wide_table_md: str):
    doc = MarkdownAdapter().parse("wide_table.md", wide_table_md)
    chunks = Chunker(max_tokens=100).chunk_document(doc)
    table_chunks = [c for c in chunks if NodeType.TABLE.value in c.node_types]
    assert len(table_chunks) >= 2
    for chunk in table_chunks:
        assert "Metric" in chunk.content_text
        assert "Value" in chunk.content_text
