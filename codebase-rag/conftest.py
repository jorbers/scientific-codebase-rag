import sys
from pathlib import Path

# Allow tests to import subpackages of kernelpack_rag directly by name
# (e.g. `from embed.base import ...` instead of `from kernelpack_rag.embed.base import ...`).
sys.path.insert(0, str(Path(__file__).parent / "kernelpack_rag"))

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires Qdrant on localhost:6333 and KP_SRC env var",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires live Qdrant and KP_SRC)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="Pass --integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def kp_src() -> Path:
    src = os.environ.get("KP_SRC")
    if not src:
        pytest.skip("KP_SRC environment variable not set")
    path = Path(src)
    if not path.exists() or not path.is_dir():
        pytest.skip(f"KP_SRC path does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def qdrant_client():
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        return client
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable on localhost:6333: {exc}")
