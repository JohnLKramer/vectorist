from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tree_sitter import Language, Parser, Tree

from vectorist.chunking.tokenizer import estimate_tokens as default_estimate_tokens

_TOP_LEVEL_DEF_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {"function_definition", "class_definition", "decorated_definition"}
    ),
    "javascript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        }
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "interface_declaration",
            "type_alias_declaration",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration"}),
    "rust": frozenset(
        {
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
        }
    ),
    "java": frozenset({"class_declaration", "interface_declaration", "method_declaration"}),
}

_GRAMMAR_LOADERS: dict[str, Any] = {}


def _load_grammar(language: str) -> Language | None:
    if language in _GRAMMAR_LOADERS:
        return _GRAMMAR_LOADERS[language]

    loader: Any | None = None
    if language == "python":
        import tree_sitter_python as mod

        loader = mod.language
    elif language == "javascript":
        import tree_sitter_javascript as mod

        loader = mod.language
    elif language == "typescript":
        import tree_sitter_typescript as mod

        loader = mod.language_typescript
    elif language == "go":
        import tree_sitter_go as mod

        loader = mod.language
    elif language == "rust":
        import tree_sitter_rust as mod

        loader = mod.language
    elif language == "java":
        import tree_sitter_java as mod

        loader = mod.language

    if loader is None:
        return None

    lang = Language(loader())
    _GRAMMAR_LOADERS[language] = lang
    return lang


def _parse(language: str, source: str) -> Tree | None:
    grammar = _load_grammar(language)
    if grammar is None:
        return None
    parser = Parser(grammar)
    return parser.parse(source.encode("utf-8"))


def _top_level_segments(tree: Tree, language: str, source: str) -> list[str]:
    def_types = _TOP_LEVEL_DEF_TYPES.get(language)
    if def_types is None:
        return []

    root = tree.root_node
    if root is None:
        return []

    data = source.encode("utf-8")
    segments: list[str] = []
    cursor = 0
    for child in root.children:
        if child.type in def_types:
            if child.start_byte > cursor:
                segments.append(data[cursor : child.start_byte].decode("utf-8"))
            segments.append(data[child.start_byte : child.end_byte].decode("utf-8"))
            cursor = child.end_byte
    if cursor < len(data):
        segments.append(data[cursor:].decode("utf-8"))

    segments = [s for s in segments if s]
    if len(segments) <= 1:
        return []
    return segments


def _line_indent(line: str) -> int:
    stripped = line.lstrip(" \t")
    return len(line) - len(stripped)


def _split_priority(line: str, prev_indent: int | None) -> int:
    stripped = line.strip()
    if not stripped:
        return 3
    indent = _line_indent(line)
    if prev_indent is not None and indent < prev_indent:
        return 2
    if stripped.endswith(";") or stripped.endswith("}"):
        return 0
    if stripped.startswith("{") or stripped.endswith("{"):
        return 1
    return 4


def _word_split_line(line: str, max_tokens: int, est: Callable[[str], int]) -> list[str]:
    words = line.split()
    if not words:
        return [line]
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and est(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_line_on_delimiters(line: str, max_tokens: int, est: Callable[[str], int]) -> list[str]:
    if est(line) <= max_tokens:
        return [line]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    token_buf = ""
    for ch in line:
        token_buf += ch
        at_boundary = ch in ";{}"
        if at_boundary:
            piece_tokens = est(token_buf)
            if current and current_tokens + piece_tokens > max_tokens:
                parts.append("".join(current))
                current = [token_buf]
                current_tokens = piece_tokens
            else:
                current.append(token_buf)
                current_tokens += piece_tokens
            token_buf = ""
    if token_buf:
        piece_tokens = est(token_buf)
        if current and current_tokens + piece_tokens > max_tokens:
            parts.append("".join(current))
            current = [token_buf]
        else:
            current.append(token_buf)
    if current:
        parts.append("".join(current))
    if parts and all(est(part) <= max_tokens for part in parts):
        return parts
    return _word_split_line(line, max_tokens, est)


def _pack_units(units: list[str], max_tokens: int, est: Callable[[str], int]) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = est(unit)
        if unit_tokens > max_tokens:
            if current:
                packed.append("".join(current))
                current = []
                current_tokens = 0
            packed.extend(_split_line_on_delimiters(unit, max_tokens, est))
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            packed.append("".join(current))
            current = [unit]
            current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens += unit_tokens

    if current:
        packed.append("".join(current))
    return packed


def _fallback_split(
    source: str,
    max_tokens: int,
    est: Callable[[str], int],
) -> list[str]:
    if est(source) <= max_tokens:
        return [source]

    if not source:
        return [source]

    lines = source.splitlines(keepends=True)
    if not lines:
        return _split_line_on_delimiters(source, max_tokens, est)

    split_after: list[int] = []
    prev_indent: int | None = None
    for index, line in enumerate(lines):
        priority = _split_priority(line, prev_indent)
        if priority < 4:
            split_after.append(index)
        if line.strip():
            prev_indent = _line_indent(line)

    def best_cut(end_index: int, start_index: int) -> int | None:
        best: tuple[int, int] | None = None
        for cut in split_after:
            if cut < start_index or cut >= end_index:
                continue
            priority = _split_priority(lines[cut], None)
            if best is None or priority < best[0] or (
                priority == best[0] and cut > best[1]
            ):
                best = (priority, cut)
        return best[1] if best else None

    parts: list[str] = []
    start = 0
    while start < len(lines):
        end = start
        tokens = 0
        inner_broke = False
        while end < len(lines):
            line = lines[end]
            line_tokens = est(line)
            if tokens + line_tokens > max_tokens and end > start:
                inner_broke = True
                break
            if line_tokens > max_tokens:
                if end > start:
                    inner_broke = True
                    break
                chunk = "".join(lines[start : end + 1])
                parts.extend(_split_line_on_delimiters(chunk, max_tokens, est))
                start = end + 1
                inner_broke = True
                break
            tokens += line_tokens
            end += 1

        if start >= len(lines):
            break

        if not inner_broke and end >= len(lines):
            parts.append("".join(lines[start:end]))
            break

        if end == start:
            parts.extend(_split_line_on_delimiters(lines[start], max_tokens, est))
            start += 1
            continue

        cut = best_cut(end - 1, start)
        if cut is not None and cut >= start:
            parts.append("".join(lines[start : cut + 1]))
            start = cut + 1
            continue

        parts.append("".join(lines[start:end]))
        start = end

    return [p for p in parts if p]


def _pack_segments(
    segments: list[str],
    max_tokens: int,
    est: Callable[[str], int],
) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for segment in segments:
        segment_tokens = est(segment)
        if segment_tokens > max_tokens:
            if current:
                packed.append("".join(current))
                current = []
                current_tokens = 0
            packed.extend(_fallback_split(segment, max_tokens, est))
            continue

        if current and current_tokens + segment_tokens > max_tokens:
            packed.append("".join(current))
            current = [segment]
            current_tokens = segment_tokens
        else:
            current.append(segment)
            current_tokens += segment_tokens

    if current:
        packed.append("".join(current))
    return packed


def split_code(
    source: str,
    language: str,
    max_tokens: int,
    estimate_tokens: Callable[[str], int] = default_estimate_tokens,
) -> list[str]:
    """Return ordered code slices each <= max_tokens when possible."""
    if max_tokens <= 0:
        return [source]
    if estimate_tokens(source) <= max_tokens:
        return [source]

    normalized = language.strip().lower()
    if normalized in {"ts", "tsx"}:
        normalized = "typescript"
    if normalized in {"js", "jsx"}:
        normalized = "javascript"
    if normalized in {"py"}:
        normalized = "python"

    tree = _parse(normalized, source)
    if tree is not None:
        segments = _top_level_segments(tree, normalized, source)
        if segments:
            return _pack_segments(segments, max_tokens, estimate_tokens)

    return _fallback_split(source, max_tokens, estimate_tokens)
