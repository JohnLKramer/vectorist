import pytest
from bs4 import BeautifulSoup

from vectorist.adapters.image_filter import image_skip_reason


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<img src="/a.png" alt="">', "empty-alt"),
        ('<img src="/a.png" alt="x" role="presentation">', "presentation"),
        ('<img src="/a.png" alt="x" aria-hidden="true">', "aria-hidden"),
        ('<img src="/a.png" alt="x" width="16">', "too-small"),
        ('<img src="/a.png" alt="x" height="47">', "too-small"),
        ('<img src="/img/site-logo.png" alt="x">', "noise-name"),
        ('<img src="/a.png" alt="x" class="avatar big">', "noise-name"),
        ('<img src="/a.png" alt="x" id="tracking-1">', "noise-name"),
        ('<img src="/ads/b.png" alt="x">', "noise-name"),
        ('<img src="data:image/svg+xml;base64,AAA" alt="x">', "svg-data-uri"),
        ('<img src="/diagram.png" alt="Pool diagram" width="640">', None),
        ('<img src="/diagram.png">', None),
    ],
)
def test_image_skip_reason(html, expected):
    assert image_skip_reason(BeautifulSoup(html, "html.parser").img) == expected
