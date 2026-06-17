"""
embed/base.py

Embedder protocol and registry.

The protocol is intentionally minimal: name, dim, embed_batch, embed_query_batch.
Every model file (jinacode.py, qwen.py, unixcoder.py) implements this.
The registry in config.py maps model name -> Embedder instance -> named spaces.

Nothing here imports model weights. This file has zero heavy dependencies.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol that every embedding model must satisfy.

    Attributes
    ----------
    name : str
        Short identifier used as the registry key and in log events.
        Must be unique across all registered embedders.
    dim : int
        Output vector dimension. Must match the declared Qdrant named space dim.
        Wrong dim here causes a schema mismatch at upsert time, not at declaration time —
        which is silent and hard to debug. Verify dim by encoding a probe string before
        creating the collection (see foundation plan D4 warning on jina-code-0.5b).

    Methods
    -------
    embed_batch(texts) -> list[list[float]]
        Embed a list of document strings. Returns one vector per input text.
        Empty input returns an empty list — do not raise.
        Callers are responsible for batching to a size the model can handle.
    embed_query_batch(texts) -> list[list[float]]
        Embed a list of query strings. Returns one vector per input text.
        Empty input returns an empty list — do not raise.
        Callers are responsible for batching to a size the model can handle.
    """

    name: str
    dim: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbedderRegistry:
    """
    Registry mapping model name -> Embedder instance.

    config.py holds the canonical mapping of model name -> named spaces.
    This registry holds the live instances.

    Usage
    -----
    reg = EmbedderRegistry()
    reg.register("jinacode", JinaCodeEmbedder())
    embedder = reg.get("jinacode")
    vecs = embedder.embed_batch(["def solve(): ..."])
    """

    def __init__(self) -> None:
        self._store: dict[str, Embedder] = {}

    def register(self, name: str, embedder: Embedder) -> None:
        """
        Register an embedder under a name.

        Raises
        ------
        ValueError
            If a name is already registered. Silent overwrites hide bugs
            (e.g. the wrong model ends up behind a name). Fail loudly instead.
        """
        if name in self._store:
            raise ValueError(
                f"Embedder '{name}' is already registered. "
                f"Unregister it first if you intend to replace it."
            )
        self._store[name] = embedder

    def get(self, name: str) -> Embedder:
        """
        Retrieve an embedder by name.

        Raises
        ------
        KeyError
            If the name has not been registered.
        """
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
