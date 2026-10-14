from __future__ import annotations

from vectorist.chunking.tokenizer import estimate_tokens
from vectorist.ir.models import TableNode


def _header_line(headers: list[str]) -> str:
    return "| " + " | ".join(headers) + " |"


def _separator_line(headers: list[str]) -> str:
    return "| " + " | ".join("---" for _ in headers) + " |"


def _row_line(row: list[str]) -> str:
    return "| " + " | ".join(row) + " |"


def _caption_prefix(
    caption: str | None,
    *,
    continued: bool,
    row_start: int,
    row_end: int,
) -> str | None:
    if continued:
        if caption:
            return f"[Table: {caption} (Continued - Rows {row_start} to {row_end})]"
        return f"[Table: (Continued - Rows {row_start} to {row_end})]"
    if caption:
        return f"[Table: {caption}]"
    return None


def _render_slice(
    node: TableNode,
    rows: list[list[str]],
    *,
    row_start: int,
    row_end: int,
    is_first: bool,
) -> str:
    parts: list[str] = []
    prefix = _caption_prefix(
        node.caption,
        continued=not is_first,
        row_start=row_start,
        row_end=row_end,
    )
    if prefix:
        parts.append(prefix)
    parts.append(_header_line(node.headers))
    parts.append(_separator_line(node.headers))
    for row in rows:
        parts.append(_row_line(row))
    return "\n".join(parts)


def _fits(node: TableNode, rows: list[list[str]], *, row_start: int, row_end: int, is_first: bool, max_tokens: int) -> bool:
    text = _render_slice(node, rows, row_start=row_start, row_end=row_end, is_first=is_first)
    return estimate_tokens(text) <= max_tokens


def _largest_row_group(
    node: TableNode,
    start_index: int,
    max_tokens: int,
    *,
    is_first: bool,
    cap: int | None,
) -> int:
    remaining = len(node.rows) - start_index
    if remaining <= 0:
        return 0
    upper = remaining if cap is None else min(remaining, cap)
    best = 0
    for count in range(1, upper + 1):
        row_start = start_index + 1
        row_end = start_index + count
        chunk_rows = node.rows[start_index : start_index + count]
        if _fits(node, chunk_rows, row_start=row_start, row_end=row_end, is_first=is_first, max_tokens=max_tokens):
            best = count
        else:
            break
    if best == 0:
        return 1
    return best


def split_table(
    node: TableNode,
    max_tokens: int,
    rows_per_group: int | None = None,
) -> list[str]:
    """Each string includes caption + header row + row group."""
    if not node.rows:
        text = _render_slice(node, [], row_start=0, row_end=0, is_first=True)
        return [text]

    full = _render_slice(
        node,
        node.rows,
        row_start=1,
        row_end=len(node.rows),
        is_first=True,
    )
    if rows_per_group is None and estimate_tokens(full) <= max_tokens:
        return [full]

    slices: list[str] = []
    index = 0
    is_first = True
    while index < len(node.rows):
        count = _largest_row_group(
            node,
            index,
            max_tokens,
            is_first=is_first,
            cap=rows_per_group,
        )
        chunk_rows = node.rows[index : index + count]
        row_start = index + 1
        row_end = index + count
        slices.append(
            _render_slice(
                node,
                chunk_rows,
                row_start=row_start,
                row_end=row_end,
                is_first=is_first,
            )
        )
        index += count
        is_first = False
    return slices
