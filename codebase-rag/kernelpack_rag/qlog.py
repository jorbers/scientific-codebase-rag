"""JSONL query event logging for KernelPack RAG."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_event(log_path: Path, event: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def fold(log_path: Path, query_id: str) -> dict | None:
    if not log_path.exists():
        return None

    folded: dict = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("query_id") != query_id:
                continue
            event_type = record.get("event")
            if event_type:
                folded[event_type] = record.get("payload", {})

    return folded or None


__all__ = ["write_event", "fold"]
