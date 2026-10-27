from __future__ import annotations

import base64

_PROMPT = (
    "Describe what this image shows in plain language. Include any text that "
    "appears in it and, for a diagram, the structure of the diagram (its parts "
    "and how they connect). The alt text from the document is: {alt_text!r}"
)


_MAX_TOKENS = 4096


class ClaudeImageEnricher:
    def __init__(self, model: str, client=None) -> None:
        self._model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    @property
    def provider_id(self) -> str:
        return f"claude:{self._model}"

    def describe(self, image: bytes, content_type: str, alt_text: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": base64.b64encode(image).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": _PROMPT.format(alt_text=alt_text)},
                    ],
                }
            ],
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise RuntimeError(
                f"Claude model {self._model} stopped at max_tokens "
                f"({_MAX_TOKENS}); the image description is truncated"
            )
        for block in response.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(
            f"Claude model {self._model} returned no text block for the image"
        )
