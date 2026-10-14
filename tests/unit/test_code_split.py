from __future__ import annotations

import signal

import pytest

from vectorist.chunking.code_split import split_code
from vectorist.chunking.tokenizer import estimate_tokens


def test_fallback_splits_on_braces_and_semicolons():
    src = "function a(){ return 1; }\nfunction b(){ return 2; }\n"
    parts = split_code(src, language="unknown", max_tokens=8)
    assert len(parts) >= 2
    assert "".join(parts) == src
    assert parts[0].rstrip("\n").endswith("}")
    assert "function a(){ return 1;" in parts[0]
    assert parts[1].startswith("function b()")


def test_python_treesitter_does_not_split_mid_function_when_budget_allows_whole_functions():
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    parts = split_code(
        src,
        language="python",
        max_tokens=estimate_tokens("def a():\n    return 1\n") + 1,
    )
    a_slices = [part for part in parts if "def a" in part or "return 1" in part]
    assert a_slices
    assert all("def a" in part and "return 1" in part for part in a_slices)


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


def test_python_class_keeps_method_body_with_signature_when_over_budget():
    src = (
        "class C:\n"
        "    @dec\n"
        "    def m1(self):\n"
        "        return 1\n"
        "    def m2(self):\n"
        "        return 2\n"
    )
    decorated = "    @dec\n    def m1(self):\n        return 1\n"
    plain = "    def m2(self):\n        return 2\n"
    budget = estimate_tokens(decorated)
    assert estimate_tokens(src) > budget
    assert estimate_tokens(decorated) <= budget
    assert estimate_tokens(plain) <= budget
    parts = split_code(src, language="python", max_tokens=budget)
    assert len(parts) >= 2
    for part in parts:
        if "def m1" in part or "return 1" in part or "@dec" in part:
            assert "def m1" in part
            assert "return 1" in part
            assert "@dec" in part
        if "def m2" in part or "return 2" in part:
            assert "def m2" in part
            assert "return 2" in part
    assert any("def m1" in part and "return 1" in part for part in parts)
    assert any("def m2" in part and "return 2" in part for part in parts)


def test_semicolon_inside_string_on_fitting_line_is_not_split_point():
    src = 'msg = "SELECT 1;\nmore code\n\n' + ("word " * 20)
    fitting = 'msg = "SELECT 1;\nmore code\n\n'
    budget = estimate_tokens(fitting)
    assert estimate_tokens(src) > budget
    assert estimate_tokens('msg = "SELECT 1;\n') <= budget
    parts = split_code(src, language="unknown", max_tokens=budget)
    assert 'msg = "SELECT 1;' in parts[0]
    assert "more code" in parts[0]
    assert "word" not in parts[0]


def test_semicolon_inside_string_is_not_a_split_point():
    src = 'x = "a;b;c"; y = 1\nz = 2\n'
    parts = split_code(src, language="unknown", max_tokens=estimate_tokens('x = "a;b;c";') + 1)
    assert any('"a;b;c"' in p for p in parts)
    assert not any(p.strip() == '"a;b;c"' for p in parts)


def test_dedent_starts_next_slice_at_less_indented_line():
    src = "    inner()\nplain()\nfollow()\n"
    fitting = "    inner()\nplain()\n"
    budget = estimate_tokens(fitting)
    assert estimate_tokens(src) > budget
    assert estimate_tokens("    inner()\n") + estimate_tokens("plain()\n") <= budget
    parts = split_code(src, language="unknown", max_tokens=budget)
    assert len(parts) >= 2
    assert "plain()" not in parts[0]
    assert parts[0].strip() == "inner()"


def test_blank_line_outranks_dedent():
    src = "    inner()\nplain()\n\n" + ("word " * 20)
    fitting = "    inner()\nplain()\n\n"
    budget = estimate_tokens(fitting) + 1
    assert estimate_tokens(src) > budget
    assert estimate_tokens("plain()") <= budget
    parts = split_code(src, language="unknown", max_tokens=budget)
    assert "plain()" in parts[0]
    assert "word" not in parts[0]


def test_java_semicolon_inside_fitting_method_does_not_split_method():
    src = (
        "class C {\n"
        "  void m1() {\n"
        "    int x = 1;\n"
        "    int z = 3;\n"
        "  }\n"
        "  void m2() {\n"
        "    int y = 2;\n"
        "    int w = 4;\n"
        "  }\n"
        "}\n"
    )
    method = (
        "  void m1() {\n"
        "    int x = 1;\n"
        "    int z = 3;\n"
        "  }\n"
    )
    budget = estimate_tokens(method)
    assert estimate_tokens(src) > budget
    assert estimate_tokens(method) <= budget
    parts = split_code(src, language="java", max_tokens=budget)
    assert any(
        "void m1" in part and "int x = 1;" in part and "int z = 3;" in part
        for part in parts
    )
    assert not any("int x = 1;" in part and "void m1" not in part for part in parts)


def test_delimiter_split_kept_when_line_also_has_long_span():
    src = "short(); " + ("word " * 30) + "; tail()\n"
    parts = split_code(src, language="unknown", max_tokens=8)
    assert parts[0].strip().endswith("short();")
    assert any("tail()" in p for p in parts)
