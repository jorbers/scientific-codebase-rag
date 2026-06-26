"""
embed/unixcoder.py

Embedder backed by microsoft/unixcoder-base (dim=768).

UniXcoder is a RoBERTa-based model pretrained on code. Mean pooling is applied
over the full last hidden state with attention-mask weighting; this mirrors the
approach used in the original UniXcoder paper for code retrieval tasks.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from kernelpack_rag.embed.base import Embedder

_MODEL_ID = "microsoft/unixcoder-base"


def _mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class UniXcoderEmbedder:
    name: str = "unixcoder"
    dim: int = 768

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self._model = AutoModel.from_pretrained(_MODEL_ID)
        self._model.eval()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = self._model(**encoded)
        vecs = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        return vecs.tolist()

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        # UniXcoder uses no query prefix — same encoding for documents and queries.
        return self.embed_batch(texts)
