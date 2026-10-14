from __future__ import annotations

import signal

import pytest

from vectorist.chunking.code_split import split_code
from vectorist.chunking.tokenizer import estimate_tokens


def test_fallback_splits_on_braces_and_semicolons():
    src = "function a(){ return 1; }\nfunction b(){ return 2; }\n"
    parts = split_code(src, language="unknown", max_tokens=8)
    assert len(parts) >= 2
    assert all("function" in p or "return" in p for p in parts)


def test_python_treesitter_does_not_split_mid_function_when_budget_allows_whole_functions():
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    parts = split_code(
        src,
        language="python",
        max_tokens=estimate_tokens("def a():\n    return 1\n") + 1,
    )
    assert any("def a" in p for p in parts)


def test_fallback_returns_when_over_budget_line_is_not_last():
    src = "x = 1\n" + ("word " * 50) + "\ny = 2\n"

    def _alarm(*_args: object) -> None:
        raise TimeoutError("split_code hung")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(2)
    try:
        parts = split_code(src, language="unknown", max_tokens=10)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    assert len(parts) >= 1
    assert "y = 2" in "".join(parts)


def test_python_class_splits_one_method_per_slice_when_over_budget():
    src = (
        "class C:\n"
        "    def m1(self):\n"
        "        return 1\n"
        "    def m2(self):\n"
        "        return 2\n"
    )
    method_budget = estimate_tokens("    def m1(self):\n        return 1\n") + 2
    parts = split_code(src, language="python", max_tokens=method_budget)
    assert len(parts) >= 2
    assert sum(1 for p in parts if "def m1" in p) == 1
    assert sum(1 for p in parts if "def m2" in p) == 1


def test_semicolon_inside_string_is_not_a_split_point():
    src = 'x = "a;b;c"; y = 1\nz = 2\n'
    parts = split_code(src, language="unknown", max_tokens=estimate_tokens('x = "a;b;c";') + 1)
    assert any('"a;b;c"' in p for p in parts)
    assert not any(p.strip() == '"a;b;c"' for p in parts)


def test_dedent_starts_next_slice_at_less_indented_line():
    src = "    inner()\nplain()\n"
    parts = split_code(src, language="unknown", max_tokens=1)
    assert len(parts) >= 2
    assert parts[0].strip() == "inner()"
    assert parts[1].lstrip().startswith("plain()")


def test_delimiter_split_kept_when_line_also_has_long_span():
    src = "short(); " + ("word " * 30) + "; tail()\n"
    parts = split_code(src, language="unknown", max_tokens=8)
    assert parts[0].strip().endswith("short();")
    assert any("tail()" in p for p in parts)
