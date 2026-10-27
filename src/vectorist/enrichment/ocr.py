from __future__ import annotations


class OcrImageEnricher:
    def __init__(self) -> None:
        self._engine = None

    @property
    def provider_id(self) -> str:
        return "ocr"

    def describe(self, image: bytes, content_type: str, alt_text: str) -> str:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        result, _elapse = self._engine(image)
        if not result:
            return ""
        return "\n".join(item[1] for item in result)
