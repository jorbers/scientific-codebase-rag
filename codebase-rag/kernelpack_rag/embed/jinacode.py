"""
embed/jinacode.py

Embedder backed by jinaai/jina-code-embeddings-0.5b (dim=896).

This model does not use SentenceTransformer task routing. Retrieval behavior is
selected by prepending instruction text to the input before tokenization:
- embed_batch: passage prefix for indexing chunks
- embed_query_batch: nl2code query prefix for search-time queries

The model card specifies last-token (EOS) pooling.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from kernelpack_rag.embed.base import Embedder

_MODEL_ID = "jinaai/jina-code-embeddings-0.5b"
_QUERY_PREFIX = "Find the most relevant code snippet given the following query:\n"
_PASSAGE_PREFIX = "Candidate code snippet:\n"
_MAX_LENGTH = 8192


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class JinaCodeEmbedder:
    name: str = "jinacode"
    dim: int = 896

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            padding_side="left",
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
        self._model.eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
        embeddings = _last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (chunks). Use at index time with the passage prefix."""
        if not texts:
            return []
        return self._encode([_PASSAGE_PREFIX + text for text in texts])

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed queries. Use at search time with the nl2code query prefix."""
        if not texts:
            return []
        return self._encode([_QUERY_PREFIX + text for text in texts])


__all__ = ["JinaCodeEmbedder"]
