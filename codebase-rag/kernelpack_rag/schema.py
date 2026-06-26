"""Qdrant collection schema creation and validation."""

from __future__ import annotations

from collections.abc import Mapping

from qdrant_client import QdrantClient, models

from kernelpack_rag.config import COLLECTIONS_CONFIG


_PAYLOAD_INDEXES = {
    "kernelpack_code": {
        "keyword": (
            "granularity",
            "chunk_type",
            "module",
            "parent_class",
            "function_name",
            "math_terms",
            "parent_id",
        ),
        "bool": ("has_numba",),
    },
    "kernelpack_papers": {
        "keyword": ("math_terms", "section", "equation_labels"),
        "bool": (),
    },
}


def ensure_collections(client: QdrantClient) -> None:
    """Create missing collections and hard-fail on any existing schema mismatch."""
    existing = _collection_names(client)

    for collection_name, expected_config in COLLECTIONS_CONFIG.items():
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=_build_vectors_config(expected_config["vectors"]),
                sparse_vectors_config=_build_sparse_vectors_config(
                    expected_config["sparse_vectors"]
                ),
            )
        else:
            _validate_collection(client, collection_name, expected_config)

        _ensure_payload_indexes(client, collection_name)


def _collection_names(client: QdrantClient) -> set[str]:
    return {collection.name for collection in client.get_collections().collections}


def _build_vectors_config(
    vectors_config: Mapping[str, Mapping[str, object]]
) -> dict[str, models.VectorParams]:
    return {
        name: models.VectorParams(
            size=(size := int(config["size"])),
            distance=_distance_model(config["distance"]),
            quantization_config=_quantization_config_for(size),
        )
        for name, config in vectors_config.items()
    }


def _build_sparse_vectors_config(
    sparse_vectors_config: Mapping[str, Mapping[str, object]]
) -> dict[str, models.SparseVectorParams]:
    return {
        name: models.SparseVectorParams(modifier=_modifier_model(config["modifier"]))
        for name, config in sparse_vectors_config.items()
    }


def _distance_model(value: object) -> models.Distance:
    if _enum_value(value) == "Cosine":
        return models.Distance.COSINE
    raise ValueError(f"Unsupported distance in COLLECTIONS_CONFIG: {value!r}")


def _modifier_model(value: object) -> models.Modifier:
    if _enum_value(value).lower() == "idf":
        return models.Modifier.IDF
    raise ValueError(f"Unsupported sparse modifier in COLLECTIONS_CONFIG: {value!r}")


def _quantization_config_for(size: int):
    if size != 1024:
        return None
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
    )


def _validate_collection(
    client: QdrantClient,
    collection_name: str,
    expected_config: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> None:
    info = client.get_collection(collection_name)
    params = info.config.params

    _validate_dense_vectors(
        collection_name,
        _as_mapping(params.vectors),
        expected_config["vectors"],
    )
    _validate_sparse_vectors(
        collection_name,
        _as_mapping(params.sparse_vectors),
        expected_config["sparse_vectors"],
    )


def _validate_dense_vectors(
    collection_name: str,
    actual_vectors: Mapping[str, object],
    expected_vectors: Mapping[str, Mapping[str, object]],
) -> None:
    _validate_key_set(collection_name, "vectors", actual_vectors, expected_vectors)

    for name, expected in expected_vectors.items():
        actual = actual_vectors[name]
        actual_size = getattr(actual, "size", None)
        expected_size = int(expected["size"])
        if actual_size != expected_size:
            raise ValueError(
                f"{collection_name}.vectors.{name}.size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )

        actual_distance = _enum_value(getattr(actual, "distance", None))
        expected_distance = _enum_value(expected["distance"])
        if actual_distance != expected_distance:
            raise ValueError(
                f"{collection_name}.vectors.{name}.distance mismatch: "
                f"expected {expected_distance}, got {actual_distance}"
            )

        _validate_quantization(collection_name, name, actual)


def _validate_sparse_vectors(
    collection_name: str,
    actual_sparse_vectors: Mapping[str, object],
    expected_sparse_vectors: Mapping[str, Mapping[str, object]],
) -> None:
    _validate_key_set(
        collection_name, "sparse_vectors", actual_sparse_vectors, expected_sparse_vectors
    )

    for name, expected in expected_sparse_vectors.items():
        actual = actual_sparse_vectors[name]
        actual_modifier = _enum_value(getattr(actual, "modifier", None)).lower()
        expected_modifier = _enum_value(expected["modifier"]).lower()
        if actual_modifier != expected_modifier:
            raise ValueError(
                f"{collection_name}.sparse_vectors.{name}.modifier mismatch: "
                f"expected {expected_modifier}, got {actual_modifier}"
            )


def _validate_key_set(
    collection_name: str,
    field_name: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    actual_names = set(actual)
    expected_names = set(expected)

    for missing_name in sorted(expected_names - actual_names):
        raise ValueError(f"{collection_name}.{field_name}.{missing_name} missing")

    for extra_name in sorted(actual_names - expected_names):
        raise ValueError(f"{collection_name}.{field_name}.{extra_name} extra")


def _validate_quantization(collection_name: str, vector_name: str, actual: object) -> None:
    quantization_config = getattr(actual, "quantization_config", None)
    size = getattr(actual, "size", None)

    if size == 1024:
        if not _is_int8_scalar_quantization(quantization_config):
            raise ValueError(
                f"{collection_name}.vectors.{vector_name}.quantization_config "
                "mismatch: expected int8 scalar quantization"
            )
        return

    if quantization_config is not None:
        raise ValueError(
            f"{collection_name}.vectors.{vector_name}.quantization_config "
            "mismatch: expected no quantization"
        )


def _is_int8_scalar_quantization(quantization_config: object) -> bool:
    scalar = _get_model_value(quantization_config, "scalar")
    scalar_type = _get_model_value(scalar, "type")
    return _enum_value(scalar_type).lower() == "int8"


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    indexes = _PAYLOAD_INDEXES.get(collection_name, {})
    for field_name in indexes.get("keyword", ()):
        _ensure_payload_index(
            client, collection_name, field_name, models.PayloadSchemaType.KEYWORD
        )
    for field_name in indexes.get("bool", ()):
        _ensure_payload_index(
            client, collection_name, field_name, models.PayloadSchemaType.BOOL
        )


def _ensure_payload_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    field_schema: models.PayloadSchemaType,
) -> None:
    if _payload_index_exists(client, collection_name, field_name):
        return

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise


def _payload_index_exists(
    client: QdrantClient, collection_name: str, field_name: str
) -> bool:
    payload_schema = getattr(client.get_collection(collection_name), "payload_schema", None)
    return isinstance(payload_schema, Mapping) and field_name in payload_schema


def _as_mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return {}


def _enum_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return "" if enum_value is None else str(enum_value)


def _get_model_value(model: object, field_name: str) -> object:
    if model is None:
        return None
    if isinstance(model, Mapping):
        return model.get(field_name)
    return getattr(model, field_name, None)


__all__ = ["ensure_collections"]
