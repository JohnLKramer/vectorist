from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tree_sitter import Language, Node, Parser, Tree

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

_INNER_DEF_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "decorated_definition"}),
    "javascript": frozenset({"method_definition", "function_declaration"}),
    "typescript": frozenset({"method_definition", "function_declaration"}),
    "go": frozenset({"method_declaration"}),
    "rust": frozenset({"function_item"}),
    "java": frozenset({"method_declaration", "constructor_declaration"}),
}

_CONTAINER_BODY_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"block", "class_definition"}),
    "javascript": frozenset({"class_body", "statement_block"}),
    "typescript": frozenset({"class_body", "statement_block"}),
    "go": frozenset({"block"}),
    "rust": frozenset({"declaration_list", "field_declaration_list"}),
    "java": frozenset({"class_body", "interface_body"}),
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


def _decode_slice(data: bytes, start: int, end: int) -> str:
    return data[start:end].decode("utf-8")


def _segments_at_boundaries(
    data: bytes,
    boundaries: list[tuple[int, int]],
    total_len: int,
) -> list[str]:
    if len(boundaries) <= 1:
        return []
    segments: list[str] = []
    cursor = 0
    for start, end in boundaries:
        if start > cursor:
            segments.append(_decode_slice(data, cursor, start))
        segments.append(_decode_slice(data, start, end))
        cursor = end
    if cursor < total_len:
        segments.append(_decode_slice(data, cursor, total_len))
    return [s for s in segments if s]


def _inner_definitions_under(start: Node, language: str) -> list[Node]:
    """Return sibling methods inside one class or function.

    A decorated definition counts as one method so the decorator stays with
    the signature and body.
    """
    inner_types = _INNER_DEF_TYPES.get(language, frozenset())
    descend_types = (
        inner_types
        | _CONTAINER_BODY_TYPES.get(language, frozenset())
        | _TOP_LEVEL_DEF_TYPES.get(language, frozenset())
    )
    queue: list[Node] = [start]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        children = list(current.children)
        found = [child for child in children if child.type in inner_types]
        if len(found) > 1:
            return found
        for child in children:
            if child.type in descend_types:
                queue.append(child)
    return []


def _boundary_nodes(node: Node, language: str, *, module_level: bool) -> list[Node]:
    top_types = _TOP_LEVEL_DEF_TYPES.get(language, frozenset())
    inner_types = _INNER_DEF_TYPES.get(language, frozenset())
    body_types = _CONTAINER_BODY_TYPES.get(language, frozenset())

    if module_level:
        matched = [child for child in node.children if child.type in top_types]
        if len(matched) > 1:
            return matched
        if len(matched) == 1:
            inner = _inner_definitions_under(matched[0], language)
            if len(inner) > 1:
                return inner
            return matched

    if node.type in top_types | inner_types | body_types:
        inner = _inner_definitions_under(node, language)
        if len(inner) > 1:
            return inner

    containers = [
        child
        for child in node.children
        if child.type in top_types | inner_types | body_types
    ]
    if len(containers) == 1:
        return _inner_definitions_under(containers[0], language)
    return []


def _tree_definition_segments(
    tree: Tree,
    source: str,
    language: str,
    *,
    module_level: bool = True,
) -> list[str]:
    root = tree.root_node
    if root is None:
        return []

    data = source.encode("utf-8")
    nodes = _boundary_nodes(root, language, module_level=module_level)
    if len(nodes) <= 1 and module_level:
        nodes = _boundary_nodes(root, language, module_level=False)
    if len(nodes) <= 1:
        return []

    boundaries = [(n.start_byte, n.end_byte) for n in nodes]
    return _segments_at_boundaries(data, boundaries, len(data))


def _split_oversized_with_tree(
    segment: str,
    language: str,
    max_tokens: int,
    est: Callable[[str], int],
) -> list[str] | None:
    tree = _parse(language, segment)
    if tree is None:
        return None

    for module_level in (True, False):
        inner = _tree_definition_segments(
            tree, segment, language, module_level=module_level
        )
        if len(inner) > 1:
            return _pack_segments(inner, language, max_tokens, est)
    return None


def _line_indent(line: str) -> int:
    stripped = line.lstrip(" \t")
    return len(line) - len(stripped)


def _delimiter_priority_at_line(line: str) -> int | None:
    if not line.strip():
        return 2
    start = len(line) - len(line.lstrip())
    end = len(line.rstrip()) - 1
    last = line[end]
    if last in ";}" and not _in_string_at(line, end):
        return 0
    first = line[start]
    if (first == "{" and not _in_string_at(line, start)) or (
        last == "{" and not _in_string_at(line, end)
    ):
        return 1
    return None


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


def _in_string_at(line: str, index: int) -> bool:
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    length = len(line)
    while i < index:
        if in_triple_single:
            if line.startswith("'''", i):
                in_triple_single = False
                i += 3
                continue
            i += 1
            continue
        if in_triple_double:
            if line.startswith('"""', i):
                in_triple_double = False
                i += 3
                continue
            i += 1
            continue
        if in_single:
            if line[i] == "\\" and i + 1 < length:
                i += 2
                continue
            if line[i] == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if line[i] == "\\" and i + 1 < length:
                i += 2
                continue
            if line[i] == '"':
                in_double = False
            i += 1
            continue
        if line.startswith("'''", i):
            in_triple_single = True
            i += 3
            continue
        if line.startswith('"""', i):
            in_triple_double = True
            i += 3
            continue
        if line[i] == "'":
            in_single = True
            i += 1
            continue
        if line[i] == '"':
            in_double = True
            i += 1
            continue
        i += 1
    return in_single or in_double or in_triple_single or in_triple_double


def _split_line_on_delimiters(line: str, max_tokens: int, est: Callable[[str], int]) -> list[str]:
    if est(line) <= max_tokens:
        return [line]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    token_buf = ""
    for index, ch in enumerate(line):
        token_buf += ch
        at_boundary = ch in ";{}" and not _in_string_at(line, index)
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

    if not parts:
        return _word_split_line(line, max_tokens, est)

    final: list[str] = []
    for part in parts:
        if est(part) <= max_tokens:
            final.append(part)
        else:
            final.extend(_word_split_line(part, max_tokens, est))
    return final


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
            packed.append(unit)
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

    split_after: list[tuple[int, int]] = []
    split_before: list[tuple[int, int]] = []
    prev_indent: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            split_after.append((index, 2))
        else:
            indent = _line_indent(line)
            if prev_indent is not None and indent < prev_indent:
                split_before.append((index, 3))
            delim = _delimiter_priority_at_line(line)
            if delim is not None:
                split_after.append((index, delim))
            prev_indent = indent

    def take_chunk(start: int, end_exclusive: int) -> None:
        if start < end_exclusive:
            parts.append("".join(lines[start:end_exclusive]))

    def best_end_exclusive(start: int, max_end: int) -> int | None:
        best: tuple[int, int] | None = None
        for line_index, priority in split_before:
            if line_index <= start or line_index > max_end:
                continue
            if best is None or priority < best[0] or (
                priority == best[0] and line_index > best[1]
            ):
                best = (priority, line_index)
        for line_index, priority in split_after:
            cut_end = line_index + 1
            if cut_end <= start or cut_end > max_end:
                continue
            if best is None or priority < best[0] or (
                priority == best[0] and cut_end > best[1]
            ):
                best = (priority, cut_end)
        return best[1] if best else None

    parts: list[str] = []
    start = 0
    while start < len(lines):
        end = start
        tokens = 0
        while end < len(lines):
            line = lines[end]
            line_tokens = est(line)
            if line_tokens > max_tokens:
                break
            if tokens + line_tokens > max_tokens and end > start:
                break
            tokens += line_tokens
            end += 1

        if end == start:
            parts.extend(_split_line_on_delimiters(lines[start], max_tokens, est))
            start += 1
            continue

        if end >= len(lines):
            take_chunk(start, end)
            break

        cut_end = best_end_exclusive(start, end)
        if cut_end is not None:
            take_chunk(start, cut_end)
            start = cut_end
        else:
            take_chunk(start, end)
            start = end

    return [p for p in parts if p]


def _split_oversized_segment(
    segment: str,
    language: str,
    max_tokens: int,
    est: Callable[[str], int],
) -> list[str]:
    tree_parts = _split_oversized_with_tree(segment, language, max_tokens, est)
    if tree_parts is not None:
        return tree_parts
    return _fallback_split(segment, max_tokens, est)


def _pack_segments(
    segments: list[str],
    language: str,
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
            for piece in _split_oversized_segment(segment, language, max_tokens, est):
                piece_tokens = est(piece)
                if piece_tokens > max_tokens:
                    packed.append(piece)
                    continue
                if current and current_tokens + piece_tokens > max_tokens:
                    packed.append("".join(current))
                    current = [piece]
                    current_tokens = piece_tokens
                else:
                    current.append(piece)
                    current_tokens += piece_tokens
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
        segments = _tree_definition_segments(tree, source, normalized, module_level=True)
        if segments:
            return _pack_segments(segments, normalized, max_tokens, estimate_tokens)

    return _fallback_split(source, max_tokens, estimate_tokens)
