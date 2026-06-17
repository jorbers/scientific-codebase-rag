"""Tests for enrich/summarize.py."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kernelpack_rag.chunking.coarse import CoarseChunk
from kernelpack_rag.enrich.summarize import content_hash, summarize_chunk, summarize_all


def make_chunk(text="def foo():\n    return 1\n", qualname="foo"):
    return CoarseChunk(
        text=text,
        qualname=qualname,
        chunk_type="function",
        line_range=(1, 2),
        source_file="src/kernelpack/test.py",
        parent_class=None,
        module="kernelpack.test",
    )


def make_client(response_text="This function returns 1."):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response_text))]
    )
    return client


# content_hash

def test_content_hash_is_sha256():
    text = "def foo(): pass"
    expected = hashlib.sha256(text.encode()).hexdigest()
    assert content_hash(text) == expected


def test_content_hash_is_string():
    assert isinstance(content_hash("hello"), str)


def test_content_hash_different_texts_differ():
    assert content_hash("abc") != content_hash("def")


# summarize_chunk — cache miss

def test_summarize_chunk_calls_llm_on_miss(tmp_path):
    chunk = make_chunk()
    client = make_client("Computes foo.")
    summary, _ = summarize_chunk(chunk, tmp_path, client)
    assert client.chat.completions.create.called
    assert summary == "Computes foo."


def test_summarize_chunk_writes_cache_on_miss(tmp_path):
    chunk = make_chunk()
    client = make_client("Computes foo.")
    _, h = summarize_chunk(chunk, tmp_path, client)
    cache_file = tmp_path / f"{h}.txt"
    assert cache_file.exists()
    assert cache_file.read_text() == "Computes foo."


def test_summarize_chunk_returns_hash(tmp_path):
    chunk = make_chunk()
    client = make_client("Computes foo.")
    _, h = summarize_chunk(chunk, tmp_path, client)
    assert h == content_hash(chunk.text)


# summarize_chunk — cache hit

def test_summarize_chunk_reads_cache_on_hit(tmp_path):
    chunk = make_chunk()
    h = content_hash(chunk.text)
    (tmp_path / f"{h}.txt").write_text("Cached summary.")
    client = make_client()
    summary, _ = summarize_chunk(chunk, tmp_path, client)
    assert summary == "Cached summary."
    assert not client.chat.completions.create.called


def test_summarize_chunk_creates_cache_dir(tmp_path):
    cache_dir = tmp_path / "new_cache"
    chunk = make_chunk()
    client = make_client("Summary.")
    summarize_chunk(chunk, cache_dir, client)
    assert cache_dir.exists()


# summarize_chunk — error handling

def test_summarize_chunk_returns_empty_on_llm_error(tmp_path):
    chunk = make_chunk()
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API error")
    summary, h = summarize_chunk(chunk, tmp_path, client)
    assert summary == ""
    assert h == content_hash(chunk.text)


def test_summarize_chunk_does_not_write_cache_on_error(tmp_path):
    chunk = make_chunk()
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API error")
    _, h = summarize_chunk(chunk, tmp_path, client)
    assert not (tmp_path / f"{h}.txt").exists()


# summarize_all

def test_summarize_all_returns_dict(tmp_path):
    chunks = [make_chunk("def a(): pass", "a"), make_chunk("def b(): pass", "b")]
    client = make_client("A summary.")
    result = summarize_all(chunks, tmp_path, client)
    assert isinstance(result, dict)


def test_summarize_all_keys_are_hashes(tmp_path):
    chunk = make_chunk()
    client = make_client("Summary.")
    result = summarize_all([chunk], tmp_path, client)
    assert content_hash(chunk.text) in result


def test_summarize_all_correct_count(tmp_path):
    chunks = [make_chunk(f"def f{i}(): pass", f"f{i}") for i in range(3)]
    client = make_client("Summary.")
    result = summarize_all(chunks, tmp_path, client)
    assert len(result) == 3


def test_summarize_all_uses_cache(tmp_path):
    chunk = make_chunk()
    h = content_hash(chunk.text)
    (tmp_path / f"{h}.txt").write_text("Cached.")
    client = make_client()
    result = summarize_all([chunk], tmp_path, client)
    assert not client.chat.completions.create.called
    assert result[h] == "Cached."