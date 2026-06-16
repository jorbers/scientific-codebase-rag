"""Paper corpus chunker.

Loads hand-curated .md files from a papers directory.
Each .md file is one chunk. Metadata comes from a sidecar .json
with the same stem. math_terms in the sidecar are the bridge
to code chunks in leg 2 of two-leg retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperChunk:
    text: str
    source_file: str
    section: str
    math_terms: list[str] = field(default_factory=list)
    equation_labels: list[str] = field(default_factory=list)
    source: str = ""


def load_paper_chunks(papers_dir: str | Path) -> list[PaperChunk]:
    """Load all paper chunks from a directory of .md + sidecar .json files.

    For each .md file found, looks for a matching .json with the same stem.
    The .md provides chunk text; the .json provides math_terms and metadata.
    .md files with no sidecar are loaded with empty math_terms.

    Args:
        papers_dir: Path to directory containing .md and .json files.

    Returns:
        List of PaperChunk, one per .md file found.
    """
    papers_dir = Path(papers_dir)
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    chunks: list[PaperChunk] = []

    for md_path in sorted(papers_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        meta: dict = {}
        json_path = md_path.with_suffix(".json")
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))

        chunks.append(PaperChunk(
            text=text,
            source_file=str(md_path),
            section=meta.get("section", md_path.stem),
            math_terms=meta.get("math_terms", []),
            equation_labels=meta.get("equation_labels", []),
            source=meta.get("source", ""),
        ))

    return chunks