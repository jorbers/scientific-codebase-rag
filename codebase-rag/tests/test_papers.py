"""Tests for chunking/papers.py."""

import json
import pytest
from pathlib import Path
from kernelpack_rag.chunking.papers import load_paper_chunks, PaperChunk


@pytest.fixture
def papers_dir(tmp_path):
    # chunk with sidecar
    (tmp_path / "rbf_kernels.md").write_text("## RBF Kernels\nMatern C4: exp(-r)*(3+3r+r^2)")
    (tmp_path / "rbf_kernels.json").write_text(json.dumps({
        "section": "RBF Kernel Families",
        "math_terms": ["matern_kernel", "c4_kernel", "shape_parameter"],
        "source": "arXiv 2603.23074",
    }))
    # chunk without sidecar
    (tmp_path / "readme_notes.md").write_text("## README\nRBF-FD stencil assembly.")
    return tmp_path


def test_returns_list_of_paper_chunks(papers_dir):
    result = load_paper_chunks(papers_dir)
    assert isinstance(result, list)
    assert all(isinstance(c, PaperChunk) for c in result)


def test_loads_both_files(papers_dir):
    result = load_paper_chunks(papers_dir)
    assert len(result) == 2


def test_text_populated(papers_dir):
    result = load_paper_chunks(papers_dir)
    for chunk in result:
        assert chunk.text


def test_math_terms_from_sidecar(papers_dir):
    result = load_paper_chunks(papers_dir)
    rbf = next(c for c in result if "rbf_kernels" in c.source_file)
    assert "matern_kernel" in rbf.math_terms
    assert "c4_kernel" in rbf.math_terms


def test_no_sidecar_gives_empty_math_terms(papers_dir):
    result = load_paper_chunks(papers_dir)
    readme = next(c for c in result if "readme_notes" in c.source_file)
    assert readme.math_terms == []


def test_section_from_sidecar(papers_dir):
    result = load_paper_chunks(papers_dir)
    rbf = next(c for c in result if "rbf_kernels" in c.source_file)
    assert rbf.section == "RBF Kernel Families"


def test_section_fallback_to_stem(papers_dir):
    result = load_paper_chunks(papers_dir)
    readme = next(c for c in result if "readme_notes" in c.source_file)
    assert readme.section == "readme_notes"


def test_source_file_is_string(papers_dir):
    result = load_paper_chunks(papers_dir)
    for chunk in result:
        assert isinstance(chunk.source_file, str)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paper_chunks(tmp_path / "nonexistent")


def test_empty_md_skipped(papers_dir):
    (papers_dir / "empty.md").write_text("   ")
    result = load_paper_chunks(papers_dir)
    names = [Path(c.source_file).stem for c in result]
    assert "empty" not in names