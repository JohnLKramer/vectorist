import io
import os

import pytest

pytestmark = pytest.mark.enricher


def _text_png() -> bytes:
    pil = pytest.importorskip("PIL.Image", reason="Pillow is not installed")
    draw_mod = pytest.importorskip("PIL.ImageDraw", reason="Pillow is not installed")
    image = pil.new("RGB", (400, 100), "white")
    draw_mod.Draw(image).text((10, 40), "Pool schedule 9am", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_claude_describes_image():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    pytest.importorskip("anthropic", reason="the claude extra is not installed")
    from vectorist.config import Settings
    from vectorist.enrichment.claude import ClaudeImageEnricher

    png = _text_png()
    model = os.environ.get("VECTORIST_CLAUDE_MODEL") or Settings().claude_model
    text = ClaudeImageEnricher(model).describe(png, "image/png", "Pool schedule")

    assert text.strip()


def test_ocr_reads_text_image():
    pytest.importorskip(
        "rapidocr_onnxruntime", reason="the ocr extra is not installed"
    )
    from vectorist.enrichment.ocr import OcrImageEnricher

    png = _text_png()
    text = OcrImageEnricher().describe(png, "image/png", "")

    assert text.strip()
