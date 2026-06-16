"""Summarize coarse code chunks with an OpenAI chat client."""

from pathlib import Path
import hashlib


SYSTEM_PROMPT = """You are a technical documentation assistant for a scientific computing
library that implements RBF-FD (radial basis function finite difference)
methods for solving PDEs. Write concise, accurate summaries of Python
functions and classes."""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def summarize_chunk(chunk, cache_dir, client) -> tuple[str, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    h = content_hash(chunk.text)
    cache_path = cache_dir / f"{h}.txt"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(), h

    user_prompt = f"""Summarize the following Python code from the KernelPack library in 2-4
sentences. Focus on: what it computes or does, what the key parameters
mean, and how it fits into the RBF-FD or solver workflow. Do not repeat
the function signature. Do not include code.

Module: {chunk.source_file}
Name: {chunk.qualname}

{chunk.text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as error:
        print(f"Warning: failed to summarize {chunk.qualname}: {error}")
        return "", h

    cache_path.write_text(summary)
    return summary, h


def summarize_all(chunks, cache_dir, client) -> dict[str, str]:
    cache_dir = Path(cache_dir)
    summaries: dict[str, str] = {}
    generated = 0
    cache_hits = 0
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        h = content_hash(chunk.text)
        cache_path = cache_dir / f"{h}.txt"
        is_cached = cache_path.exists() and cache_path.stat().st_size > 0

        summary, h = summarize_chunk(chunk, cache_dir, client)
        summaries[h] = summary

        if is_cached:
            cache_hits += 1
            status = "cached"
        else:
            generated += 1
            status = "generated"
        print(f"  [{index}/{total}] {status} | {chunk.qualname} ({chunk.source_file})")

    print(f"Done. {generated} generated, {cache_hits} cache hits, {total} total.")
    return summaries
