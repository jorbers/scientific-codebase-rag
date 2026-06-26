"""Embedder protocol and fail-loud registry. Zero heavy dependencies."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedding model must satisfy. dim must match the Qdrant named-space dim."""

    name: str
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbedderRegistry:
    """Fail-loud map of model name → live Embedder instance."""

    def __init__(self) -> None:
        self._store: dict[str, Embedder] = {}

    def register(self, name: str, embedder: Embedder) -> None:
        """Raises ValueError if name is already registered — silent overwrites hide bugs."""
        if name in self._store:
            raise ValueError(
                f"Embedder '{name}' is already registered. "
                f"Unregister it first if you intend to replace it."
            )
        self._store[name] = embedder

    def get(self, name: str) -> Embedder:
        """Raises KeyError with available names if name is not registered."""
        try:
            return self._store[name]
        except KeyError:
            available = list(self._store.keys())
            raise KeyError(
                f"No embedder registered under '{name}'. "
                f"Registered: {available}"
            )

    def list_names(self) -> list[str]:
        return list(self._store.keys())
