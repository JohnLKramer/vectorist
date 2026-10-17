import pytest

from vectorist.cli.crawl.urls import Scope, is_html_candidate, normalize


def test_normalize_strips_fragment_query_and_trailing_slash():
    assert normalize("HTTPS://Docs.Example/a/b/?x=1#frag", False) == "https://docs.example/a/b"
    assert normalize("https://docs.example/a?x=1", True) == "https://docs.example/a?x=1"
    assert normalize("https://docs.example", False) == "https://docs.example/"
    assert normalize("https://docs.example/", False) == "https://docs.example/"


def test_scope_prefix_boundary():
    scope = Scope("https://h.example/docs/")
    assert scope.contains("https://h.example/docs")
    assert scope.contains("https://h.example/docs/a/b")
    assert not scope.contains("https://h.example/docs-old/a")
    assert not scope.contains("https://h.example/other")
    assert not scope.contains("http://h.example/docs/a")
    assert not scope.contains("https://other.example/docs/a")
    assert scope.list_prefix == "https://h.example/docs"


def test_root_scope_contains_whole_host():
    scope = Scope("https://h.example")
    assert scope.contains("https://h.example/anything")
    assert scope.list_prefix == "https://h.example/"


def test_start_url_naming_a_file_scopes_to_its_directory():
    scope = Scope("https://h.example/docs/index.html")
    assert scope.contains("https://h.example/docs/other.html")
    assert not scope.contains("https://h.example/blog/x")
    assert scope.list_prefix == "https://h.example/docs"
    assert Scope("https://h.example/index.html").contains("https://h.example/any")


def test_scope_rejects_non_http():
    with pytest.raises(ValueError):
        Scope("ftp://h.example/x")


def test_non_html_extensions_skipped():
    assert not is_html_candidate("https://h/a/logo.png")
    assert not is_html_candidate("https://h/a/file.PDF")
    assert is_html_candidate("https://h/a/page.html")
    assert is_html_candidate("https://h/a/page")
    assert is_html_candidate("https://h/a.b/page")
