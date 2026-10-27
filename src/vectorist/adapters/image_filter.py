from __future__ import annotations

import re

from bs4 import Tag

_MIN_DIMENSION = 48
_NOISE = re.compile(
    r"icon|logo|avatar|badge|sprite|pixel|tracking|emoji|banner|advert|/ads?/",
    re.IGNORECASE,
)


def image_skip_reason(tag: Tag) -> str | None:
    if str(tag.get("aria-hidden", "")).lower() == "true":
        return "aria-hidden"
    if str(tag.get("role", "")).lower() == "presentation":
        return "presentation"
    if tag.get("alt") == "":
        return "empty-alt"
    if any(_below_minimum(tag.get(name)) for name in ("width", "height")):
        return "too-small"
    src = str(tag.get("src") or "")
    if src.lower().startswith("data:image/svg+xml"):
        return "svg-data-uri"
    classes = tag.get("class") or []
    candidates = [src, str(tag.get("id") or ""), *classes]
    if any(_NOISE.search(value) for value in candidates):
        return "noise-name"
    return None


def _below_minimum(value: object) -> bool:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return match is not None and int(match.group(1)) < _MIN_DIMENSION
