from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vectorist.chunking.code_split import split_code
from vectorist.chunking.tokenizer import estimate_tokens
from vectorist.ir.models import CodeBlockNode, IRDocument, IRNode, NodeType, SectionNode


def _context_header(breadcrumbs: list[str]) -> str:
    return f"[Context: {' > '.join(breadcrumbs)}]"


def _content_text(breadcrumbs: list[str], body_parts: list[str]) -> str:
    header = _context_header(breadcrumbs)
    body = "\n\n".join(body_parts)
    return f"{header}\n\n{body}" if body else header


def _is_boundary_section(node: IRNode) -> bool:
    return node.node_type == NodeType.SECTION and isinstance(node, SectionNode) and node.level <= 2


def _oversized_code_or_table(node: IRNode) -> bool:
    return node.node_type in (NodeType.CODE_BLOCK, NodeType.TABLE)


def _split_oversized_content(node: IRNode, limit: int) -> list[str]:
    if node.node_type == NodeType.TABLE:
        raise NotImplementedError("oversized table splitting is handled in later tasks")
    if isinstance(node, CodeBlockNode):
        return split_code(node.raw_content, node.language, limit)
    raise NotImplementedError("oversized code/table splitting is handled in later tasks")


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
        batch: list[tuple[IRNode, str]] = []

        def batch_breadcrumbs() -> list[str]:
            return batch[0][0].breadcrumbs if batch else []

        def flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            crumbs = batch_breadcrumbs()
            chunk = self._make_chunk(batch, crumbs)
            assert estimate_tokens(chunk.content_text) <= self._max_tokens
            chunks.append(chunk)
            batch = []

        for node in nodes:
            crumbs = node.breadcrumbs
            header = _context_header(crumbs)
            limit = self._max_tokens - estimate_tokens(header)

            if _oversized_code_or_table(node) and estimate_tokens(node.raw_content) > limit:
                flush_batch()
                pieces = _split_oversized_content(node, limit)
            elif estimate_tokens(node.raw_content) > limit:
                pieces = self._split_text_to_limit(node.raw_content, limit)
            else:
                pieces = [node.raw_content]

            for piece in pieces:
                piece_tokens = estimate_tokens(piece)
                if piece_tokens > limit and node.node_type == NodeType.TABLE:
                    flush_batch()
                    raise NotImplementedError(
                        "oversized table splitting is handled in later tasks"
                    )

                if batch and batch_breadcrumbs() != node.breadcrumbs:
                    flush_batch()

                trial = batch + [(node, piece)]
                trial_crumbs = trial[0][0].breadcrumbs
                trial_text = _content_text(trial_crumbs, [t for _, t in trial])
                if batch and estimate_tokens(trial_text) > self._max_tokens:
                    flush_batch()
                    trial = [(node, piece)]
                    trial_crumbs = node.breadcrumbs
                    trial_text = _content_text(trial_crumbs, [piece])

                if estimate_tokens(trial_text) > self._max_tokens:
                    if node.node_type == NodeType.TABLE:
                        flush_batch()
                        raise NotImplementedError(
                            "oversized table splitting is handled in later tasks"
                        )
                    flush_batch()
                    chunk = self._make_chunk([(node, piece)], node.breadcrumbs)
                    assert estimate_tokens(chunk.content_text) <= self._max_tokens
                    chunks.append(chunk)
                    continue

                batch = trial

        flush_batch()
        return chunks

    def _word_split(self, text: str, limit: int) -> list[str]:
        pieces: list[str] = []
        words = text.split()
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

    def _split_text_to_limit(self, text: str, limit: int) -> list[str]:
        blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not blocks:
            return [text]

        pieces: list[str] = []
        for block in blocks:
            if estimate_tokens(block) <= limit:
                pieces.append(block)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", block.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 1:
                for sentence in sentences:
                    if estimate_tokens(sentence) <= limit:
                        pieces.append(sentence)
                    else:
                        pieces.extend(self._word_split(sentence, limit))
            else:
                pieces.extend(self._word_split(block, limit))
        return pieces

    def _make_chunk(
        self,
        items: list[tuple[IRNode, str]],
        breadcrumbs: list[str],
    ) -> Chunk:
        texts = [text for _, text in items]
        body = "\n\n".join(texts)
        seen: set[str] = set()
        node_ids: list[str] = []
        node_types: list[str] = []
        for node, _ in items:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            node_ids.append(node.node_id)
            node_types.append(node.node_type.value)

        nodes = [node for node, _ in items]
        starts = [n.line_range[0] for n in nodes]
        ends = [n.line_range[1] for n in nodes]
        metadata: dict[str, Any] = {}
        for n in nodes:
            metadata.update(n.metadata)

        content_text = _content_text(breadcrumbs, texts)
        return Chunk(
            chunk_index=0,
            content_text=content_text,
            breadcrumbs=list(breadcrumbs),
            node_ids=node_ids,
            node_types=node_types,
            line_range=(min(starts), max(ends)),
            metadata=metadata,
        )
