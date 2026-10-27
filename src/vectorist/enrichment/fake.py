from __future__ import annotations

from vectorist.hashing import sha256_bytes


class FakeImageEnricher:
    """Deterministic enricher for tests."""

    @property
    def provider_id(self) -> str:
        return "fake"

    def describe(self, image: bytes, content_type: str, alt_text: str) -> str:
        return f"fake description of image {sha256_bytes(image)[:8]}"
