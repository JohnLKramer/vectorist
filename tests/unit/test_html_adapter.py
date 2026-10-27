import pytest

from vectorist.adapters.html import HtmlAdapter, HtmlParseError
from vectorist.ir.models import NodeType


def _parse(html: str, uri: str = "https://docs.example/pool"):
    return HtmlAdapter().parse(uri, html)


def _nodes(doc, node_type):
    return [n for n in doc.nodes.values() if n.node_type == node_type]


def test_main_content_only_and_breadcrumbs(fixtures_dir):
    doc = _parse((fixtures_dir / "html" / "guide.html").read_text())
    text = " ".join(n.raw_content for n in doc.nodes.values())
    assert "Home" not in text and "Copyright" not in text and "not content" not in text
    paragraphs = {n.raw_content: n for n in _nodes(doc, NodeType.PARAGRAPH)}
    assert paragraphs["Set the size."].breadcrumbs[-1] == "Configuration"
    assert paragraphs["Reuse connections to cut latency."].breadcrumbs[-1] == "Pooling"


def test_code_language_table_and_media(fixtures_dir):
    doc = _parse((fixtures_dir / "html" / "guide.html").read_text())
    (code,) = _nodes(doc, NodeType.CODE_BLOCK)
    assert code.language == "python" and "pool.start()" in code.raw_content
    (table,) = _nodes(doc, NodeType.TABLE)
    assert table.headers == ["Name", "Default"] and table.rows == [["size", "10"]]
    assert table.caption == "Settings"
    (media,) = _nodes(doc, NodeType.MEDIA)
    assert media.asset_uri_or_hash == "/diagram.png" and media.alt_text == "Pool diagram"


def test_line_ranges_come_from_source(fixtures_dir):
    doc = _parse((fixtures_dir / "html" / "guide.html").read_text())
    para = next(n for n in _nodes(doc, NodeType.PARAGRAPH) if n.raw_content.startswith("Reuse"))
    assert para.line_range[0] == 8


def test_body_fallback_strips_chrome(fixtures_dir):
    doc = _parse((fixtures_dir / "html" / "no_main.html").read_text())
    text = " ".join(n.raw_content for n in doc.nodes.values())
    assert "Body text here." in text
    for chrome in ("Site header", "Menu", "Sidebar", "Footer text"):
        assert chrome not in text


def test_role_main_is_selected():
    doc = _parse('<body><div>junk</div><div role="main"><p>real</p></div></body>')
    text = " ".join(n.raw_content for n in doc.nodes.values())
    assert "real" in text and "junk" not in text


def test_empty_page_fails():
    with pytest.raises(HtmlParseError):
        _parse("<html><body><nav>only nav</nav></body></html>")


def test_malformed_html_parses():
    doc = _parse("<main><h1>Open<p>Unclosed paragraph<p>Another")
    assert any("Another" in n.raw_content for n in doc.nodes.values())


def test_noisy_image_is_skipped_and_diagram_is_pending(fixtures_dir):
    html = (fixtures_dir / "html" / "guide.html").read_text().replace(
        '<img src="/diagram.png" alt="Pool diagram">',
        '<img src="/diagram.png" alt="Pool diagram">\n<img src="/px.png" alt="x" width="1" height="1">',
    )
    doc = _parse(html)
    media = {n.metadata["ref"]: n for n in _nodes(doc, NodeType.MEDIA)}
    assert media["/diagram.png"].enrichment_status == "PENDING"
    assert media["/px.png"].enrichment_status == "SKIPPED"
    assert media["/px.png"].metadata["skip_reason"] == "too-small"
