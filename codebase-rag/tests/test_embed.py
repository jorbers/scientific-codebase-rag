"""
Tests for Step 10: embed/base.py and embed/representations.py

Run with: pytest tests/test_embed.py -v

These tests do NOT require Qdrant, model weights, or GPU.
They test the protocol contract and representation logic only.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional

from kernelpack_rag.embed.base import Embedder, EmbedderRegistry
from kernelpack_rag.embed.representations import build_representation, RepresentationKey



# ---------------------------------------------------------------------------
# Minimal fixture: a fake chunk payload (as the ingestor would produce it)
# ---------------------------------------------------------------------------

def make_chunk(
    text: str = "def solve(self, rhs):\n    return self._solver(rhs)",
    llm_summary: str = "Solves the linear system Ax=b using an iterative method.",
    context_header: str = "# class: PoissonSolver\n# neighbors: __init__, assemble",
    docstring: Optional[str] = None,
) -> dict:
    """Minimal chunk payload dict matching D7 spec."""
    return {
        "text": text,
        "llm_summary": llm_summary,
        "context_header": context_header,
        "docstring": docstring,  # None when chunk has no docstring
    }


# ---------------------------------------------------------------------------
# Dummy embedder for protocol tests (no real model needed)
# ---------------------------------------------------------------------------

class DummyEmbedder:
    """Implements the Embedder protocol with fake vectors."""

    name: str = "dummy"
    dim: int = 4

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Return a deterministic fake vector: [len(t), 0, 0, 0] normalized
        out = []
        for t in texts:
            v = [float(len(t)), 0.0, 0.0, 0.0]
            norm = v[0] or 1.0
            out.append([x / norm for x in v])
        return out


# ---------------------------------------------------------------------------
# embed/base.py — protocol and registry tests
# ---------------------------------------------------------------------------

class TestEmbedderProtocol:
    def test_dummy_satisfies_protocol(self):
        """DummyEmbedder satisfies the Embedder protocol."""
        e: Embedder = DummyEmbedder()  # type: ignore[assignment]
        assert hasattr(e, "name")
        assert hasattr(e, "dim")
        assert hasattr(e, "embed_batch")

    def test_embed_batch_returns_correct_shape(self):
        e = DummyEmbedder()
        texts = ["hello", "world", "rbf stencil"]
        vecs = e.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == e.dim for v in vecs)

    def test_embed_batch_returns_lists_of_floats(self):
        e = DummyEmbedder()
        vecs = e.embed_batch(["test"])
        assert isinstance(vecs[0], list)
        assert isinstance(vecs[0][0], float)

    def test_embed_batch_empty_input(self):
        e = DummyEmbedder()
        vecs = e.embed_batch([])
        assert vecs == []


class TestEmbedderRegistry:
    def test_register_and_lookup(self):
        reg = EmbedderRegistry()
        reg.register("dummy", DummyEmbedder())
        assert reg.get("dummy") is not None

    def test_lookup_unknown_raises(self):
        reg = EmbedderRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent_model")

    def test_list_names(self):
        reg = EmbedderRegistry()
        reg.register("a", DummyEmbedder())
        reg.register("b", DummyEmbedder())
        names = reg.list_names()
        assert set(names) == {"a", "b"}

    def test_register_duplicate_raises(self):
        """Registering the same name twice should fail loudly, not silently overwrite."""
        reg = EmbedderRegistry()
        reg.register("dup", DummyEmbedder())
        with pytest.raises(ValueError):
            reg.register("dup", DummyEmbedder())


# ---------------------------------------------------------------------------
# embed/representations.py — representation builder tests
# ---------------------------------------------------------------------------

class TestBuildRepresentation:

    # --- ctx ---

    def test_ctx_ordering(self):
        """ctx = summary + context_header + text, in that order."""
        chunk = make_chunk(
            llm_summary="Summary text.",
            context_header="# class: Foo",
            text="def bar(): pass",
        )
        result = build_representation(chunk, RepresentationKey.CTX)
        assert result is not None
        summary_pos = result.index("Summary text.")
        header_pos = result.index("# class: Foo")
        text_pos = result.index("def bar(): pass")
        assert summary_pos < header_pos < text_pos

    def test_ctx_missing_summary_still_works(self):
        """If llm_summary is None or empty, ctx falls back to header + text."""
        chunk = make_chunk(llm_summary="", context_header="# class: Foo", text="def bar(): pass")
        result = build_representation(chunk, RepresentationKey.CTX)
        assert result is not None
        assert "def bar(): pass" in result

    def test_ctx_missing_header_still_works(self):
        chunk = make_chunk(context_header="")
        result = build_representation(chunk, RepresentationKey.CTX)
        assert result is not None
        assert "Solves" in result  # from default summary

    # --- code ---

    def test_code_strips_docstring(self):
        """code repr should not contain the docstring."""
        text_with_doc = (
            'def solve(self, rhs):\n'
            '    """Solve the system."""\n'
            '    return self._solver(rhs)'
        )
        chunk = make_chunk(text=text_with_doc)
        result = build_representation(chunk, RepresentationKey.CODE)
        assert result is not None
        assert "Solve the system." not in result
        assert "return self._solver(rhs)" in result

    def test_code_strips_inline_comments(self):
        text_with_comment = (
            "def solve(self):\n"
            "    # apply preconditioner\n"
            "    return self._x\n"
        )
        chunk = make_chunk(text=text_with_comment)
        result = build_representation(chunk, RepresentationKey.CODE)
        assert result is not None
        assert "apply preconditioner" not in result
        assert "return self._x" in result

    def test_code_no_docstring_returns_text(self):
        """If there's nothing to strip, code == text."""
        chunk = make_chunk(text="def f(x):\n    return x")
        result = build_representation(chunk, RepresentationKey.CODE)
        assert result is not None
        assert "def f(x):" in result

    # --- codecom ---

    def test_codecom_is_verbatim_text(self):
        """codecom returns the raw text field unchanged."""
        chunk = make_chunk(text="def f(x):\n    # comment\n    return x")
        result = build_representation(chunk, RepresentationKey.CODECOM)
        assert result == chunk["text"]

    # --- com ---

    def test_com_returns_docstring_when_present(self):
        text_with_doc = (
            'def solve(self):\n'
            '    """Solve using GMRES."""\n'
            '    return self._x\n'
        )
        chunk = make_chunk(text=text_with_doc)
        result = build_representation(chunk, RepresentationKey.COM)
        assert result is not None
        assert "GMRES" in result

    def test_com_includes_inline_comments(self):
        text = "def f(x):\n    # normalize\n    return x / x.max()\n"
        chunk = make_chunk(text=text)
        result = build_representation(chunk, RepresentationKey.COM)
        assert result is not None
        assert "normalize" in result

    def test_com_returns_none_when_no_comments_or_docstring(self):
        """Returns None when there is nothing to extract — ingestor skips the vector."""
        chunk = make_chunk(text="def f(x):\n    return x * 2")
        result = build_representation(chunk, RepresentationKey.COM)
        assert result is None

    # --- edge cases ---

    def test_invalid_key_raises(self):
        chunk = make_chunk()
        with pytest.raises((ValueError, KeyError)):
            build_representation(chunk, "not_a_real_key")  # type: ignore[arg-type]

    def test_all_keys_covered(self):
        """Every RepresentationKey should be handled without NotImplementedError."""
        chunk = make_chunk(
            text='def f(x):\n    """doc"""\n    # comment\n    return x',
            llm_summary="Summary.",
            context_header="# class: Foo",
        )
        for key in RepresentationKey:
            # Should not raise; may return None for COM when no comments
            try:
                build_representation(chunk, key)
            except NotImplementedError:
                pytest.fail(f"RepresentationKey.{key.name} is not implemented")

# ---------------------------------------------------------------------------
# Model file smoke tests (no model loading — class attribute checks only)
# ---------------------------------------------------------------------------

class TestModelFileContracts:
    def test_jinacode_contract(self):
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        assert JinaCodeEmbedder.name == "jinacode"
        assert JinaCodeEmbedder.dim == 896
        assert hasattr(JinaCodeEmbedder, "embed_batch")
        assert hasattr(JinaCodeEmbedder, "embed_query_batch")

    def test_qwen_contract(self):
        from kernelpack_rag.embed.qwen import QwenEmbedder
        assert QwenEmbedder.name == "qwen3"
        assert QwenEmbedder.dim == 1024
        assert hasattr(QwenEmbedder, "embed_batch")
        assert hasattr(QwenEmbedder, "embed_query_batch")

    def test_unixcoder_contract(self):
        from kernelpack_rag.embed.unixcoder import UniXcoderEmbedder
        assert UniXcoderEmbedder.name == "unixcoder"
        assert UniXcoderEmbedder.dim == 768
        assert hasattr(UniXcoderEmbedder, "embed_batch")