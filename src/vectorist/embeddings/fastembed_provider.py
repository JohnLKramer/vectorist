from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from fastembed import TextEmbedding

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


class FastEmbedProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._model_lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return TextEmbedding.get_embedding_size(self._model_name)

    @property
    def model_id(self) -> str:
        return self._model_name

    def _ensure_model(self) -> TextEmbedding:
        with self._model_lock:
            if self._model is None:
                self._model = TextEmbedding(
                    model_name=self._model_name, lazy_load=True
                )
            return self._model

    def _to_vectors(self, arrays: Iterable[np.ndarray]) -> list[list[float]]:
        return [array.tolist() for array in arrays]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        return self._to_vectors(model.passage_embed(texts))

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        return next(model.query_embed(text)).tolist()
