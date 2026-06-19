"""Shared Qdrant filter and scroll helpers."""

from __future__ import annotations

from qdrant_client import QdrantClient, models


def _scroll_points(
    client: QdrantClient,
    collection_name: str,
    *,
    scroll_filter: models.Filter | None = None,
    with_payload: bool = True,
    with_vectors: bool | list[str] = False,
    limit: int | None = None,
):
    offset = None
    remaining = limit

    while True:
        page_limit = min(256, remaining) if limit is not None else 256
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=page_limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        for point in points:
            yield point

        if next_offset is None:
            break
        if limit is not None:
            remaining -= len(points)
            if remaining <= 0:
                break
        offset = next_offset


def _field_equals_filter(field_name: str, value: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=field_name,
                match=models.MatchValue(value=value),
            )
        ]
    )


def _field_any_filter(field_name: str, values: list[str]) -> models.Filter:
    return models.Filter(
        should=[
            models.FieldCondition(
                key=field_name,
                match=models.MatchValue(value=value),
            )
            for value in values
        ]
    )
