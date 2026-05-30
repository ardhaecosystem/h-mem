"""Sentence-transformer embedding wrapper for H-Mem.

Provides:
- Lazy model loading
- Batch encoding
- Async-friendly (runs in thread pool to avoid blocking)
- Device selection (CPU / CUDA)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from hmem.config import HMemConfig


class SentenceEmbedder:
    """Wrapper around sentence-transformers for H-Mem."""

    _model: Any = None  # type: ignore[no-any-return]
    _model_name: str = ""

    def __init__(self, config: HMemConfig | None = None) -> None:
        self.config = config
        self.model_name = (config.embedding_model if config else "sentence-transformers/all-MiniLM-L6-v2")
        self.device = (config.embedding_device if config else "cpu")
        self._embedding_dim = (config.embedding_dim if config else 384)

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformer model."""
        if SentenceEmbedder._model is not None and SentenceEmbedder._model_name == self.model_name:
            return SentenceEmbedder._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        SentenceEmbedder._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=str(Path.home() / ".cache" / "sentence_transformers"),
        )
        SentenceEmbedder._model_name = self.model_name
        return SentenceEmbedder._model

    def encode(self, texts: str | list[str], **kwargs: Any) -> np.ndarray:
        """Encode text(s) into dense vectors.

        Args:
            texts: Single string or list of strings.
            **kwargs: Passed to model.encode (show_progress_bar, etc.)

        Returns:
            Numpy array of shape (d,) for single text, (n, d) for list.
        """
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            **kwargs,
        )
        if single:
            embeddings = embeddings.reshape(-1)
        return embeddings  # type: ignore[no-any-return]

    async def encode_async(self, texts: str | list[str], **kwargs: Any) -> np.ndarray:
        """Async wrapper that runs encoding in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # uses default executor
            lambda: self.encode(texts, **kwargs),
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._embedding_dim
