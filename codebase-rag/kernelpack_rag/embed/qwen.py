"""
embed/qwen.py

Embedder backed by Qwen/Qwen3-Embedding-0.6B (dim=1024).

Uses last-token pooling with L2 normalization, per the model card.
- embed_batch: no instruction prefix — for indexing documents
- embed_query_batch: instruction-prefixed — for query embedding at search time
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from kernelpack_rag.embed.base import Embedder

_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_QUERY_TASK = "Given a scientific computing question, retrieve relevant code that answers it"


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def _instruct(query: str) -> str:
    return f"Instruct: {_QUERY_TASK}\nQuery: {query}"


class QwenEmbedder:
    name: str = "qwen3"
    dim: int = 1024

    def __init__(self) -> None:
        # padding_side='left' required for last-token pooling correctness
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, padding_side="left")
        self._model = AutoModel.from_pretrained(_MODEL_ID)
        self._model.eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
        embeddings = _last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (chunks). Use at index time. No instruction prefix."""
        if not texts:
            return []
        return self._encode(texts)

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed queries. Use at search time. Instruction-prefixed."""
        if not texts:
            return []
        return self._encode([_instruct(t) for t in texts])