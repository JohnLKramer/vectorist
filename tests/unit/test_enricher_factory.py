from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vectorist.config import Settings
from vectorist.enrichment.factory import build_enricher


def _settings(tmp_path: Path, enricher: str) -> Settings:
    return Settings(
        cache_path=tmp_path / "cache.db",
        vec_path=tmp_path / "vec.db",
        enricher=enricher,
    )


def test_none_returns_none(tmp_path: Path) -> None:
    assert build_enricher(_settings(tmp_path, "none")) is None


def test_fake_returns_fake_enricher(tmp_path: Path) -> None:
    enricher = build_enricher(_settings(tmp_path, "fake"))

    assert enricher is not None
    assert enricher.provider_id == "fake"
    first = enricher.describe(b"abc", "image/png", "alt")
    assert first == enricher.describe(b"abc", "image/png", "other alt")
    assert first != enricher.describe(b"abd", "image/png", "alt")


def test_unknown_enricher_fails_naming_setting(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported enricher: bogus"):
        build_enricher(_settings(tmp_path, "bogus"))


def test_claude_without_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_enricher(_settings(tmp_path, "claude"))


def test_real_providers_name_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    with pytest.raises(RuntimeError, match="uv sync --extra claude"):
        build_enricher(_settings(tmp_path, "claude"))
    with pytest.raises(RuntimeError, match="uv sync --extra ocr"):
        build_enricher(_settings(tmp_path, "ocr"))
