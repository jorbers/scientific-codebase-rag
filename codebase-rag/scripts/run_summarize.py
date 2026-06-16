from pathlib import Path
import os
import sys

import openai

from kernelpack_rag.chunking.coarse import CoarseChunk, chunk_file
from kernelpack_rag.enrich.summarize import summarize_all


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        return 1

    client = openai.OpenAI()
    repo_root = Path(__file__).resolve().parent.parent
    paths = sorted(Path("/Users/jordanchambers/public-projects/kernelpack-python/src/kernelpack").rglob("*.py"))

    chunks: list[CoarseChunk] = []
    for path in paths:
        try:
            chunks.extend(chunk_file(path))
        except Exception as error:
            print(f"Warning: failed to chunk {path}: {error}")

    print(f"Found {len(chunks)} chunks across {len(paths)} files.")
    cache_dir = repo_root / "summaries_cache"
    summarize_all(chunks, cache_dir, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
