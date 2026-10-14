from __future__ import annotations

import hashlib
import math

_DEFAULT_DIMENSIONS = 384
_FAKE_MODEL_ID = "fake/BAAI/bge-small-en-v1.5"


def _hash_to_unit_vector(text: str, *, kind: str, dimensions: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{kind}:{text}:{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest) - 3, 4):
            if len(values) >= dimensions:
                break
            chunk = int.from_bytes(digest[offset : offset + 4], "big", signed=False)
            values.append((chunk / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0] * dimensions
    return [value / norm for value in values]


class FakeEmbeddingProvider:
    """Deterministic hash-based unit vectors of length 384 for tests."""

    def __init__(self, dimensions: int = _DEFAULT_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return _FAKE_MODEL_ID

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_unit_vector(text, kind="document", dimensions=self._dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_to_unit_vector(text, kind="query", dimensions=self._dimensions)
