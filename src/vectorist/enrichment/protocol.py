from __future__ import annotations

from typing import Protocol


class ImageEnricherProtocol(Protocol):
    @property
    def provider_id(self) -> str: ...

    def describe(self, image: bytes, content_type: str, alt_text: str) -> str: ...
