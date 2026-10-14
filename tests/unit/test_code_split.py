from __future__ import annotations

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
