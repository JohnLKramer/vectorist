from vectorist.adapters.language import detect_code_language


def test_fence_info_string_takes_precedence():
    assert detect_code_language("python", "file.txt") == ("python", "fence")


def test_fence_info_string_strips_and_uses_first_token():
    assert detect_code_language("  typescript highlight", "") == ("typescript", "fence")


def test_extension_from_source_uri_when_no_fence_info():
    assert detect_code_language(None, "src/main.py") == ("python", "extension")
    assert detect_code_language("", "lib/app.js") == ("javascript", "extension")


def test_unknown_when_no_fence_and_no_known_extension():
    assert detect_code_language(None, "readme.md") == ("unknown", "unknown")
    assert detect_code_language(None, "") == ("unknown", "unknown")
