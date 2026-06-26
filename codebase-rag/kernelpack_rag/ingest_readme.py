"""Ingest README.md sections from KernelPack into the kernelpack_code collection."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from qdrant_client import models
from tqdm import tqdm

from kernelpack_rag.chunking.metadata import KP_NAMESPACE
from kernelpack_rag.config import CODE_COLLECTION, make_client
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.embed.sparse import to_qdrant_sparse
from kernelpack_rag.enrich.summarize import content_hash
from kernelpack_rag.ingest import BATCH_SIZE, UpsertBatcher

_kp_src = os.environ.get("KP_SRC", "")
README_PATH = (
    Path(_kp_src).parent.parent / "README.md"
    if _kp_src
    else Path("README.md")
)

_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_MAX_TOKENS = 400


def split_readme(text: str) -> list[tuple[str, str]]:
    """Return (title, body) pairs for each section of the README.

    The preamble before the first ## header is yielded as title "overview".
    Each ## section runs from its header line to the start of the next header.
    """
    matches = list(_SECTION_RE.finditer(text))
    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip() if matches else text.strip()
    if preamble:
        sections.append(("overview", preamble))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        sections.append((title, body))

    return sections


def _subsplit_section(
    title: str, body: str, tokenizer
) -> list[tuple[str, str, int]]:
    """Split a section into (title, text, paragraph_idx) chunks.

    Sections at or under _MAX_TOKENS are returned as a single chunk at index 0.
    Longer sections are split on blank-line boundaries; each non-empty paragraph
    becomes its own chunk indexed from 0.
    """
    if len(tokenizer.encode(body)) <= _MAX_TOKENS:
        return [(title, body, 0)]
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    return [(title, para, idx) for idx, para in enumerate(paragraphs)]


def _point_id(section_title: str, paragraph_idx: int) -> str:
    return str(uuid.uuid5(KP_NAMESPACE, f"readme:README.md:{section_title}:coarse:{paragraph_idx}"))


def _payload(section_title: str, section_text: str) -> dict:
    return {
        "text": section_text,
        "context_header": None,
        "llm_summary": None,
        "chunk_type": "readme",
        "module": "kernelpack",
        "parent_class": None,
        "function_name": None,
        "math_terms": [],
        "cross_refs": [],
        "cross_ref_ids": [],
        "has_numba": False,
        "source_file": "README.md",
        "line_range": [0, 0],
        "granularity": "coarse",
        "parent_id": None,
        "math_source_id": None,
        "content_hash": content_hash(section_text),
    }


def ingest_readme(readme_path: Path = README_PATH) -> list[str]:
    """Embed and upsert all README sections. Returns the list of point ID strings."""
    text = readme_path.read_text()
    sections = split_readme(text)
    tqdm.write(f"Found {len(sections)} README sections")

    tqdm.write("Loading embedder...")
    embedder = JinaCodeEmbedder()

    chunks: list[tuple[str, str, int]] = []
    for title, body in sections:
        chunks.extend(_subsplit_section(title, body, embedder._tokenizer))
    if len(chunks) > len(sections):
        tqdm.write(f"Expanded to {len(chunks)} chunks after paragraph splitting")

    qdrant = make_client()

    point_ids = [_point_id(title, para_idx) for title, _, para_idx in chunks]
    existing = qdrant.retrieve(
        collection_name=CODE_COLLECTION,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,
    )
    existing_hashes = {str(p.id): (p.payload or {}).get("content_hash") for p in existing}

    to_embed = [
        (title, body, para_idx)
        for (title, body, para_idx), pid in zip(chunks, point_ids)
        if existing_hashes.get(pid) != content_hash(body)
    ]
    skipped = len(chunks) - len(to_embed)

    batcher = UpsertBatcher(qdrant, CODE_COLLECTION)

    if to_embed:
        texts = [body for _, body, _ in to_embed]
        tqdm.write(f"Embedding {len(texts)} chunks...")
        dense_vecs = embedder.embed_batch(texts)

        for (title, body, para_idx), dense_vec in tqdm(
            zip(to_embed, dense_vecs), total=len(to_embed), desc="upserting README chunks"
        ):
            pid = _point_id(title, para_idx)
            batcher.add(
                models.PointStruct(
                    id=pid,
                    vector={
                        "ctx__jinacode": dense_vec,
                        "bm25_code": to_qdrant_sparse(body),
                    },
                    payload=_payload(title, body),
                )
            )

        batcher.flush()

    tqdm.write(
        f"Done: {len(chunks)} chunks total — "
        f"{skipped} unchanged (skipped), {len(to_embed)} re-embedded"
    )

    found = qdrant.retrieve(
        collection_name=CODE_COLLECTION,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    tqdm.write(f"Sanity check: {len(found)}/{len(point_ids)} points confirmed in Qdrant")

    return point_ids


if __name__ == "__main__":
    ingest_readme()
