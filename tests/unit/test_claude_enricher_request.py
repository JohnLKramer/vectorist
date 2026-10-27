import base64
from types import SimpleNamespace

import pytest

from vectorist.enrichment.claude import ClaudeImageEnricher


class StubMessages:
    def __init__(self, content, **extra):
        self.content = content
        self.extra = extra
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content, **self.extra)


def _client(content, **extra):
    return SimpleNamespace(messages=StubMessages(content, **extra))


def test_describe_sends_image_and_alt_text_and_returns_text():
    client = _client([SimpleNamespace(type="text", text="a diagram")])
    enricher = ClaudeImageEnricher("claude-test", client=client)

    result = enricher.describe(b"png", "image/png", "Pool diagram")

    assert result == "a diagram"
    assert enricher.provider_id == "claude:claude-test"
    request = client.messages.calls[0]
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 4096
    blocks = request["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(b"png").decode("ascii"),
    }
    assert blocks[1]["type"] == "text"
    assert "Pool diagram" in blocks[1]["text"]


def test_describe_raises_when_response_has_no_text_block():
    client = _client([SimpleNamespace(type="tool_use")])
    enricher = ClaudeImageEnricher("claude-test", client=client)

    with pytest.raises(RuntimeError):
        enricher.describe(b"png", "image/png", "alt")


def test_describe_raises_when_description_is_truncated():
    client = _client(
        [SimpleNamespace(type="text", text="a diagram of")], stop_reason="max_tokens"
    )
    enricher = ClaudeImageEnricher("claude-test", client=client)

    with pytest.raises(RuntimeError, match="claude-test.*max_tokens"):
        enricher.describe(b"png", "image/png", "alt")


def test_describe_accepts_end_turn_stop_reason():
    client = _client([SimpleNamespace(type="text", text="a diagram")], stop_reason="end_turn")
    enricher = ClaudeImageEnricher("claude-test", client=client)

    assert enricher.describe(b"png", "image/png", "alt") == "a diagram"
