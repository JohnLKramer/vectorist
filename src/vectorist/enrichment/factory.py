from __future__ import annotations

import os

from vectorist.config import Settings
from vectorist.enrichment.fake import FakeImageEnricher
from vectorist.enrichment.protocol import ImageEnricherProtocol


def build_enricher(settings: Settings) -> ImageEnricherProtocol | None:
    if settings.enricher == "none":
        return None
    if settings.enricher == "fake":
        return FakeImageEnricher()
    if settings.enricher == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("enricher 'claude' requires ANTHROPIC_API_KEY to be set")
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "enricher 'claude' requires the claude extra: uv sync --extra claude"
            ) from exc
        from vectorist.enrichment.claude import ClaudeImageEnricher

        return ClaudeImageEnricher(settings.claude_model)
    if settings.enricher == "ocr":
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "enricher 'ocr' requires the ocr extra: uv sync --extra ocr"
            ) from exc
        from vectorist.enrichment.ocr import OcrImageEnricher

        return OcrImageEnricher()
    raise RuntimeError(f"unsupported enricher: {settings.enricher}")
