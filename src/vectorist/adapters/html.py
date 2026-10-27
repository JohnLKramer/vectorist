from __future__ import annotations

from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from vectorist.adapters.image_filter import image_skip_reason
from vectorist.adapters.language import detect_code_language
from vectorist.adapters.markdown import _IRBuilder
from vectorist.ir.models import CodeBlockNode, IRDocument, TableNode

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_TEXT_BLOCKS = {"p", "li", "blockquote", "dt", "dd", "figcaption", "summary"}
_INLINE = {
    "a", "span", "em", "strong", "b", "i", "code", "small", "sup", "sub", "abbr",
    "kbd", "mark", "u", "br", "label", "cite", "q", "s", "time", "var", "samp",
}
_ALWAYS_DROPPED = ("script", "style", "noscript", "template")
_CHROME_TAGS = ("nav", "header", "footer", "aside")
_CHROME_ROLES = {"navigation", "banner", "contentinfo"}


class HtmlParseError(ValueError):
    pass


def _end_line(tag: Tag, start: int) -> int:
    return start + tag.get_text().strip().count("\n")


def _start_line(tag: Tag) -> int:
    return tag.sourceline or 1


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ").split())


def _code_language_class(tag: Tag) -> str | None:
    for el in [tag, *tag.find_all("code", limit=1)]:
        for cls in el.get("class") or []:
            for prefix in ("language-", "lang-"):
                if cls.startswith(prefix) and len(cls) > len(prefix):
                    return cls[len(prefix):]
    return None


def _select_main(soup: BeautifulSoup, source_uri: str) -> Tag:
    for selector in ("main", "article", "[role=main]"):
        found = soup.select_one(selector)
        if found is not None and _text(found):
            return found
    body = soup.body or soup
    for tag in body.find_all(_CHROME_TAGS):
        tag.decompose()
    for tag in body.find_all(attrs={"role": True}):
        if tag.attrs is not None and tag.get("role") in _CHROME_ROLES:
            tag.decompose()
    return body


class _HtmlIRBuilder(_IRBuilder):
    def add_code(self, tag: Tag) -> None:
        parent_id = self._attach_top_level_block(self._current_parent_id())
        node_id = self._next_id()
        start = _start_line(tag)
        language, source = detect_code_language(_code_language_class(tag), self._source_uri)
        node = CodeBlockNode(
            node_id=node_id,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=(start, start + tag.get_text().count("\n")),
            raw_content=tag.get_text().rstrip("\n"),
            language=language,
            detection_source=source,
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)

    def add_table(self, tag: Tag) -> None:
        parent_id = self._attach_top_level_block(self._current_parent_id())
        node_id = self._next_id()
        rows: list[list[str]] = []
        headers: list[str] = []
        for tr in tag.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [_text(c) for c in cells]
            if not headers and cells and all(c.name == "th" for c in cells):
                headers = row
            elif row:
                rows.append(row)
        caption = tag.find("caption")
        start = _start_line(tag)
        node = TableNode(
            node_id=node_id,
            parent_id=parent_id,
            children_ids=[],
            breadcrumbs=self._breadcrumbs(),
            line_range=(start, _end_line(tag, start)),
            raw_content=str(tag),
            caption=_text(caption) if caption else None,
            headers=headers,
            rows=rows,
        )
        self._nodes[node_id] = node
        self._add_child(parent_id, node_id)

    def walk(self, container: Tag) -> None:
        buffer: list[str] = []
        buffer_start = [1]

        def flush(end_tag: Tag | None = None) -> None:
            text = " ".join(" ".join(buffer).split())
            buffer.clear()
            if text:
                start = buffer_start[0]
                self._add_paragraph(text, (start, start))

        for child in container.children:
            if isinstance(child, NavigableString):
                if not buffer:
                    buffer_start[0] = _start_line(container)
                buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in _INLINE:
                if not buffer:
                    buffer_start[0] = _start_line(child)
                buffer.append(child.get_text(" "))
                continue
            flush()
            if name in _HEADINGS:
                title = _text(child)
                if title:
                    start = _start_line(child)
                    self._open_section(_HEADINGS[name], title, (start, _end_line(child, start)))
            elif name == "pre":
                self.add_code(child)
            elif name == "table":
                self.add_table(child)
            elif name == "img":
                self._add_image(child)
            elif name in _TEXT_BLOCKS:
                text = _text(child)
                if text:
                    start = _start_line(child)
                    self._add_paragraph(text, (start, _end_line(child, start)))
                for img in child.find_all("img"):
                    self._add_image(img)
            else:
                self.walk(child)
        flush()

    def _add_image(self, tag: Tag) -> None:
        src = tag.get("src") or ""
        alt = tag.get("alt") or ""
        start = _start_line(tag)
        self._add_media(str(src), str(alt), (start, start), image_skip_reason(tag))


class HtmlAdapter:
    def parse(self, source_uri: str, content: str) -> IRDocument:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup.find_all(_ALWAYS_DROPPED):
            tag.decompose()
        h1 = soup.find("h1")
        title_tag = soup.find("title")
        label = (
            (_text(h1) if h1 else "")
            or (_text(title_tag) if title_tag else "")
            or urlparse(source_uri).path.rstrip("/").rsplit("/", 1)[-1]
            or source_uri
        )
        main = _select_main(soup, source_uri)
        if not _text(main):
            raise HtmlParseError(f"no indexable text found in HTML for {source_uri}")
        builder = _HtmlIRBuilder(source_uri=source_uri, content=content)
        builder._file_label = label
        builder.walk(main)
        return builder.to_document()
