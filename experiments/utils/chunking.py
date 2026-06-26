import re
from pathlib import Path


# ── paths ──────────────────────────────────────────────────────────────────────
# Assumes kernelpack-python is cloned in the same parent directory as this repo.
# If not, update this path to point at your local kernelpack-python/src/kernelpack.
REPO_PATH = Path("../../kernelpack-python/src/kernelpack")

# ── retrieval config ───────────────────────────────────────────────────────────
MIN_LINES = 5   # functions with fewer than 5 lines dropped (inclusive)


def load_docs(repo_path: Path = REPO_PATH) -> list[dict]:
    if not repo_path.exists():
        raise FileNotFoundError(
            f"kernelpack source not found at {repo_path.resolve()}. "
            "Clone kernelpack-python as a sibling of this repo or update REPO_PATH."
        )
    docs = []
    for path in sorted(repo_path.rglob("*.py")):  # will include __init__.py files
        docs.append({"path": str(path), "text": path.read_text(encoding="utf-8")})
    print(f"Loaded {len(docs)} files from {repo_path}")
    return docs


def get_class_name(node):
    parent = node.parent
    if parent and parent.type == "decorated_definition":
        parent = parent.parent
    if (parent and parent.type == "block"
            and parent.parent and parent.parent.type == "class_definition"):
        return parent.parent.child_by_field_name("name").text.decode("utf-8")
    return None

def extract_chunks(source: str, tree, path: str, min_lines: int = MIN_LINES):
    """Walk the AST; return (kept, dropped) split by min_lines threshold."""
    kept, dropped = [], []
    def walk(node):
        if node.type == "function_definition":
            start = node.start_point[0]
            end   = node.end_point[0]
            entry = {
                "path":          path,
                "function_name": node.child_by_field_name("name").text.decode("utf-8"),
                "class_name":    get_class_name(node),
                "text":          source[node.start_byte:node.end_byte],
                "start_line":    start,
                "end_line":      end,
            }
            (kept if (end - start + 1) >= min_lines else dropped).append(entry)
            return  # don't recurse into functions
        for child in node.children:
            walk(child)
    walk(tree.root_node)
    return kept, dropped


# ── BM25 tokenization ──────────────────────────────────────────────────────────

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25, splitting code identifiers into sub-words."""
    tokens = []
    for raw_token in re.split(r"[^A-Za-z0-9_]+", text):
        for snake_part in raw_token.split("_"):
            tokens.extend(
                part.lower()
                for part in CAMEL_BOUNDARY.sub(" ", snake_part).split()
                if part
            )
    return tokens


# ── ID and metadata ───────────────────────────────────────────────────────────

def make_chunk_id(chunk: dict) -> str:
    """Stable ID: relative file path + line range."""
    fp = str(Path(chunk["path"]).relative_to(REPO_PATH))
    return f"{fp}:{chunk['start_line']}-{chunk['end_line']}"


def make_metadata(chunk: dict) -> dict:
    """Standard metadata dict stored alongside each chunk in ChromaDB."""
    return {
        "file_path":     str(Path(chunk["path"]).relative_to(REPO_PATH)),
        "function_name": chunk["function_name"],
        "class_name":    chunk["class_name"] or "",
        "start_line":    chunk["start_line"],
        "end_line":      chunk["end_line"],
    }
