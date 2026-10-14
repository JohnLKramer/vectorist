from __future__ import annotations

import itertools
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from vectorist.adapters.language import detect_code_language
from vectorist.ir.models import (
    CodeBlockNode,
    IRDocument,
    IRNode,
    MediaNode,
    NodeType,
    ParagraphNode,
    SectionNode,
    TableNode,
)

_HEADING_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _line_range(token: Token) -> tuple[int, int]:
    if token.map is None:
        return (1, 1)
    start, end = token.map
    return (start + 1, end)


class MarkdownAdapter:
    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark").enable(["table", "image"])

    def parse(self, source_uri: str, content: str) -> IRDocument:
        builder = _IRBuilder(source_uri=source_uri, content=content)
        tokens = self._md.parse(content)
        builder.build(tokens)
        return builder.to_document()


class _IRBuilder:
    def __init__(self, source_uri: str, content: str) -> None:
        self._source_uri = source_uri
        self._content = content
        self._file_label = Path(source_uri).name
        self._id_gen = (f"n{i}" for i in itertools.count(1))
        self._nodes: dict[str, IRNode] = {}
        self._root_id: str | None = None
        self._section_stack: list[tuple[int, str]] = []

    def _next_id(self) -> str:
        return next(self._id_gen)

    def _breadcrumbs(self) -> list[str]:
        titles = [self._nodes[sid].title for _, sid in self._section_stack]  # type: ignore[attr-defined]
        return [self._file_label, *titles]

    def _current_parent_id(self) -> str | None:
        if not self._section_stack:
            return None
        return self._section_stack[-1][1]

    def _add_child(self, parent_id: str | None, child_id: str) -> None:
        if parent_id is None:
            return
        parent = self._nodes[parent_id]
        parent.children_ids.append(child_id)

    def _open_section(self, level: int, title: str, line_range: tuple[int, int]) -> str:
        while self._section_stack and self._section_stack[-1][0] >= level:
            self._section_stack.pop()

        parent_id = self._current_parent_id()
        node_id = self._next_id()
        breadcrumbs = [self._file_label] + [
            self._nodes[sid].title for _, sid in self._section_stack  # type: ignore[attr-defined]
        ]

        section = SectionNode(
            node_id=node_id,
            node_type=NodeType.SECTION,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=breadcrumbs,
            line_range=line_range,
            raw_content=title,
            metadata={},
            level=level,
            title=title,
        )
        self._nodes[node_id] = section
        self._add_child(parent_id, node_id)
        self._section_stack.append((level, node_id))

        if self._root_id is None:
            self._root_id = node_id

        return node_id

    def _add_paragraph(self, text: str, line_range: tuple[int, int]) -> None:
        parent_id = self._current_parent_id()
        node_id = self._next_id()
        node = ParagraphNode(
            node_id=node_id,
            node_type=NodeType.PARAGRAPH,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=line_range,
            raw_content=text,
            metadata={},
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)

    def _add_code_block(self, token: Token) -> None:
        parent_id = self._current_parent_id()
        node_id = self._next_id()
        info = token.info if token.info else None
        language, detection_source = detect_code_language(info, self._source_uri)
        node = CodeBlockNode(
            node_id=node_id,
            node_type=NodeType.CODE_BLOCK,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=_line_range(token),
            raw_content=token.content,
            metadata={},
            language=language,
            detection_source=detection_source,
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)

    def _add_table(self, tokens: list[Token], start: int) -> int:
        parent_id = self._current_parent_id()
        node_id = self._next_id()
        open_token = tokens[start]
        headers: list[str] = []
        rows: list[list[str]] = []
        in_head = False
        row: list[str] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != "table_close":
            tok = tokens[i]
            if tok.type == "thead_open":
                in_head = True
            elif tok.type == "thead_close":
                in_head = False
            elif tok.type == "tr_open":
                row = []
            elif tok.type == "tr_close":
                if in_head:
                    headers = row
                elif row:
                    rows.append(row)
            elif tok.type in ("th_open", "td_open") and i + 1 < len(tokens):
                inline = tokens[i + 1]
                if inline.type == "inline":
                    row.append(inline.content)
                i += 1
            i += 1

        node = TableNode(
            node_id=node_id,
            node_type=NodeType.TABLE,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=_line_range(open_token),
            raw_content=self._slice_lines(open_token),
            metadata={},
            caption=None,
            headers=headers,
            column_types=[],
            rows=rows,
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)
        return i

    def _add_media(self, src: str, alt_text: str, line_range: tuple[int, int]) -> None:
        parent_id = self._current_parent_id()
        node_id = self._next_id()
        node = MediaNode(
            node_id=node_id,
            node_type=NodeType.MEDIA,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=line_range,
            raw_content=alt_text or src,
            metadata={},
            media_type="image",
            alt_text=alt_text,
            caption="",
            asset_uri_or_hash=src,
            enrichment_status="PENDING",
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)

    def _slice_lines(self, token: Token) -> str:
        if token.map is None:
            return ""
        lines = self._content.splitlines()
        return "\n".join(lines[token.map[0] : token.map[1]])

    def _handle_inline(self, token: Token) -> bool:
        """Return True if inline was fully handled (e.g. lone image)."""
        if not token.children:
            return False
        images = [c for c in token.children if c.type == "image"]
        if len(images) == 1 and len(token.children) == 1:
            img = images[0]
            src = img.attrGet("src") or ""
            alt = img.content or img.attrGet("alt") or ""
            self._add_media(src, alt, _line_range(token))
            return True
        return False

    def build(self, tokens: list[Token]) -> None:
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = _HEADING_LEVEL[tok.tag]
                inline = tokens[i + 1]
                title = inline.content if inline.type == "inline" else ""
                self._open_section(level, title, _line_range(tok))
                i += 3
                continue
            if tok.type == "paragraph_open":
                inline = tokens[i + 1]
                if inline.type == "inline" and inline.content.strip():
                    if not self._handle_inline(inline):
                        self._add_paragraph(inline.content, _line_range(tok))
                i += 3
                continue
            if tok.type == "fence":
                self._add_code_block(tok)
                i += 1
                continue
            if tok.type == "table_open":
                i = self._add_table(tokens, i) + 1
                continue
            i += 1

    def to_document(self) -> IRDocument:
        if self._root_id is None:
            node_id = self._next_id()
            section = SectionNode(
                node_id=node_id,
                node_type=NodeType.SECTION,
                parent_id=None,
                children_ids=[],
                breadcrumbs=[self._file_label],
                line_range=(1, max(1, len(self._content.splitlines()))),
                raw_content=self._file_label,
                metadata={},
                level=1,
                title=self._file_label,
            )
            self._nodes[node_id] = section
            self._root_id = node_id
        return IRDocument(
            source_uri=self._source_uri,
            root_id=self._root_id,
            nodes=self._nodes,
        )
