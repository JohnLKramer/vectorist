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
        self._synthetic_root_id: str | None = None

    def _next_id(self) -> str:
        return next(self._id_gen)

    def _breadcrumbs(self) -> list[str]:
        titles = [self._nodes[sid].title for _, sid in self._section_stack]  # type: ignore[attr-defined]
        return [self._file_label, *titles]

    def _current_parent_id(self) -> str | None:
        if self._section_stack:
            return self._section_stack[-1][1]
        if self._synthetic_root_id is not None:
            return self._synthetic_root_id
        return None

    def _ensure_synthetic_root(self) -> str:
        if self._synthetic_root_id is not None:
            return self._synthetic_root_id

        node_id = self._next_id()
        line_count = max(1, len(self._content.splitlines()))
        section = SectionNode(
            node_id=node_id,
            node_type=NodeType.SECTION,
            parent_id=None,
            children_ids=[],
            breadcrumbs=[self._file_label],
            line_range=(1, line_count),
            raw_content=self._file_label,
            metadata={},
            level=1,
            title=self._file_label,
        )
        self._nodes[node_id] = section
        self._synthetic_root_id = node_id
        self._root_id = node_id
        return node_id

    def _promote_to_synthetic_document_root(self) -> None:
        former_root_id = self._root_id
        if former_root_id is None:
            self._ensure_synthetic_root()
            return
        synthetic_id = self._ensure_synthetic_root()
        if former_root_id == synthetic_id:
            return
        former_root = self._nodes[former_root_id]
        if former_root.parent_id is not None:
            return
        former_root.parent_id = synthetic_id
        self._nodes[synthetic_id].children_ids.append(former_root_id)
        self._root_id = synthetic_id

    def _add_child(self, parent_id: str | None, child_id: str) -> None:
        if parent_id is None:
            return
        parent = self._nodes[parent_id]
        parent.children_ids.append(child_id)

    def _open_section(self, level: int, title: str, line_range: tuple[int, int]) -> str:
        while self._section_stack and self._section_stack[-1][0] >= level:
            self._section_stack.pop()

        if not self._section_stack:
            if self._synthetic_root_id is not None:
                pass
            elif self._root_id is not None:
                self._promote_to_synthetic_document_root()

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

        if self._root_id is None or self._root_id == self._synthetic_root_id:
            if self._synthetic_root_id is None and parent_id is None:
                self._root_id = node_id

        return node_id

    def _attach_top_level_block(self, parent_id: str | None) -> str | None:
        if self._section_stack or parent_id is not None:
            return parent_id
        return self._ensure_synthetic_root()

    def _add_paragraph(self, text: str, line_range: tuple[int, int]) -> None:
        parent_id = self._attach_top_level_block(self._current_parent_id())
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
        parent_id = self._attach_top_level_block(self._current_parent_id())
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
        parent_id = self._attach_top_level_block(self._current_parent_id())
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
        parent_id = self._attach_top_level_block(self._current_parent_id())
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

    def _process_inline_paragraph(self, token: Token, line_range: tuple[int, int]) -> None:
        if not token.children:
            if token.content.strip():
                self._add_paragraph(token.content, line_range)
            return

        images = [c for c in token.children if c.type == "image"]
        if not images:
            self._add_paragraph(token.content, line_range)
            return

        text_buffer = ""
        for child in token.children:
            if child.type == "text":
                text_buffer += child.content
            elif child.type == "softbreak" or child.type == "hardbreak":
                text_buffer += "\n"
            elif child.type == "image":
                if text_buffer.strip():
                    self._add_paragraph(text_buffer.strip(), line_range)
                text_buffer = ""
                src = child.attrGet("src") or ""
                alt = child.content or child.attrGet("alt") or ""
                self._add_media(src, alt, line_range)
            else:
                text_buffer += child.content
        if text_buffer.strip():
            self._add_paragraph(text_buffer.strip(), line_range)

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
                    self._process_inline_paragraph(inline, _line_range(tok))
                i += 3
                continue
            if tok.type in ("fence", "code_block"):
                self._add_code_block(tok)
                i += 1
                continue
            if tok.type == "html_block":
                if tok.content.strip():
                    self._add_paragraph(tok.content.rstrip("\n"), _line_range(tok))
                i += 1
                continue
            if tok.type == "table_open":
                i = self._add_table(tokens, i) + 1
                continue
            i += 1

    def to_document(self) -> IRDocument:
        if self._root_id is None:
            self._ensure_synthetic_root()
        return IRDocument(
            source_uri=self._source_uri,
            root_id=self._root_id,
            nodes=self._nodes,
        )
