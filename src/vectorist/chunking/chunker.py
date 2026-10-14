from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vectorist.chunking.tokenizer import estimate_tokens
from vectorist.ir.models import IRDocument, IRNode, NodeType, SectionNode


def _context_header(breadcrumbs: list[str]) -> str:
    return f"[Context: {' > '.join(breadcrumbs)}]"


def _is_boundary_section(node: IRNode) -> bool:
    return node.node_type == NodeType.SECTION and isinstance(node, SectionNode) and node.level <= 2


def _oversized_code_or_table(node: IRNode) -> bool:
    return node.node_type in (NodeType.CODE_BLOCK, NodeType.TABLE)


def _split_paragraph_text(text: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(blocks) > 1:
        return blocks

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        return sentences
    return [text]


@dataclass
class Chunk:
    chunk_index: int
    content_text: str
    breadcrumbs: list[str]
    node_ids: list[str]
    node_types: list[str]
    line_range: tuple[int, int]
    metadata: dict[str, Any]


class Chunker:
    def __init__(self, max_tokens: int = 512) -> None:
        self._max_tokens = max_tokens

    def chunk_document(self, doc: IRDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for region_nodes in self._iter_regions(doc):
            chunks.extend(self._chunk_region(region_nodes))
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    def _iter_regions(self, doc: IRDocument) -> list[list[IRNode]]:
        regions: list[list[IRNode]] = []
        self._regions_in_container(doc, doc.root_id, regions)
        return regions

    def _regions_in_container(
        self, doc: IRDocument, container_id: str, regions: list[list[IRNode]]
    ) -> None:
        container = doc.nodes[container_id]
        buffer: list[IRNode] = []

        def flush() -> None:
            if buffer:
                regions.append(list(buffer))
                buffer.clear()

        for child_id in container.children_ids:
            child = doc.nodes[child_id]
            if _is_boundary_section(child):
                flush()
                self._regions_in_container(doc, child.node_id, regions)
            elif child.node_type == NodeType.SECTION:
                buffer.extend(self._leaf_content_nodes(doc, child))
            else:
                buffer.append(child)
        flush()

    def _leaf_content_nodes(self, doc: IRDocument, section: IRNode) -> list[IRNode]:
        nodes: list[IRNode] = []
        for child_id in section.children_ids:
            child = doc.nodes[child_id]
            if child.node_type == NodeType.SECTION:
                nodes.extend(self._leaf_content_nodes(doc, child))
            else:
                nodes.append(child)
        return nodes

    def _chunk_region(self, nodes: list[IRNode]) -> list[Chunk]:
        chunks: list[Chunk] = []
        batch: list[IRNode] = []
        batch_texts: list[str] = []
        breadcrumbs = nodes[0].breadcrumbs if nodes else []

        def batch_token_count() -> int:
            if not batch_texts:
                return 0
            return estimate_tokens("\n\n".join(batch_texts))

        def flush_batch() -> None:
            nonlocal batch, batch_texts
            if not batch:
                return
            chunks.append(self._make_chunk(batch, breadcrumbs, batch_texts))
            batch = []
            batch_texts = []

        for node in nodes:
            breadcrumbs = node.breadcrumbs
            header_tokens = estimate_tokens(_context_header(breadcrumbs))
            limit = self._max_tokens - header_tokens

            if _oversized_code_or_table(node) and estimate_tokens(node.raw_content) > limit:
                flush_batch()
                raise NotImplementedError(
                    "oversized code/table splitting is handled in later tasks"
                )

            pieces = (
                self._split_text_to_limit(node.raw_content, limit)
                if estimate_tokens(node.raw_content) > limit
                else [node.raw_content]
            )

            for piece in pieces:
                piece_tokens = estimate_tokens(piece)
                if piece_tokens > limit:
                    if _oversized_code_or_table(node):
                        raise NotImplementedError(
                            "oversized code/table splitting is handled in later tasks"
                        )

                if batch and batch_token_count() + piece_tokens > limit:
                    flush_batch()

                if len(pieces) == 1 and piece == node.raw_content:
                    batch.append(node)
                    batch_texts.append(piece)
                else:
                    batch.append(node)
                    batch_texts.append(piece)

        flush_batch()
        return chunks

    def _split_text_to_limit(self, text: str, limit: int) -> list[str]:
        pieces: list[str] = []
        for block in _split_paragraph_text(text):
            if estimate_tokens(block) <= limit:
                pieces.append(block)
                continue
            words = block.split()
            current: list[str] = []
            for word in words:
                candidate = " ".join(current + [word])
                if current and estimate_tokens(candidate) > limit:
                    pieces.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                pieces.append(" ".join(current))
        return pieces

    def _make_chunk(
        self,
        nodes: list[IRNode],
        breadcrumbs: list[str],
        texts: list[str],
    ) -> Chunk:
        body = "\n\n".join(texts)
        node_ids = [n.node_id for n in nodes]
        node_types = [n.node_type.value for n in nodes]
        starts = [n.line_range[0] for n in nodes]
        ends = [n.line_range[1] for n in nodes]
        metadata: dict[str, Any] = {}
        for n in nodes:
            metadata.update(n.metadata)

        header = _context_header(breadcrumbs)
        content_text = f"{header}\n\n{body}" if body else header
        return Chunk(
            chunk_index=0,
            content_text=content_text,
            breadcrumbs=list(breadcrumbs),
            node_ids=node_ids,
            node_types=node_types,
            line_range=(min(starts), max(ends)),
            metadata=metadata,
        )
