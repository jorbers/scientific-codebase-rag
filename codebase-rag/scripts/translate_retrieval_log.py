#!/usr/bin/env python3
"""Translate retrieval.jsonl into a human-readable retrieval log.

The raw log only has point_ids, scores, and random query_ids -- none of
which mean anything at a glance. This script:

  - Groups retrieve_code calls that happened close together in time into
    one "session" (e.g. the 4 calls Codex made while generating one
    driver script).
  - Names each session after a slug of what the agent was actually asking
    for, plus date/time -- not a random ID.
  - Writes one README.md per session answering: what was the agent
    generating for, what queries did it send, what chunks came back
    (function + file only, top 3 per query), and which single chunk was
    most consistently retrieved across the whole session (your strongest
    "this was probably the answer" signal).
  - Writes a top-level summary.md as an INDEX across all sessions, not a
    flat dump of every candidate.
  - Skips sessions already on disk (reads back the saved JSON instead of
    re-querying Qdrant) unless --force is passed.

Usage:
    python translate_retrieval_log.py logs/retrieval.jsonl \
        --collection kernelpack_code \
        --qdrant-url http://localhost:6333 \
        --out logs/retrieval_readable
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from qdrant_client import QdrantClient

DISPLAY_TZ = ZoneInfo(os.environ.get("DISPLAY_TZ", "America/Los_Angeles"))

TEXT_PREVIEW_CHARS = 240
TOP_N_PER_QUERY_IN_README = 3

PAYLOAD_FIELDS_TO_SHOW = [
    "function_name",
    "module",
    "parent_class",
    "chunk_type",
    "source_file",
    "line_range",
    "math_terms",
    "granularity",
]


@dataclass
class RetrievalEvent:
    query_id: str
    ts: datetime
    plan: str
    query_text: str
    fusion: str
    reranker_id: str
    spaces: list[str]
    filters: dict
    candidates: list[dict]


def parse_log(path: Path) -> list[RetrievalEvent]:
    events: list[RetrievalEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping unparseable line {line_no}")
                continue
            if record.get("event") != "retrieval":
                continue
            payload = record["payload"]
            events.append(
                RetrievalEvent(
                    query_id=record["query_id"],
                    ts=datetime.fromisoformat(record["ts"]),
                    plan=payload.get("plan", ""),
                    query_text=payload.get("query_text", ""),
                    fusion=payload.get("fusion", ""),
                    reranker_id=payload.get("reranker_id", ""),
                    spaces=payload.get("spaces", []),
                    filters=payload.get("filters", {}),
                    candidates=payload.get("candidates", []),
                )
            )
    return events


def group_into_sessions(
    events: list[RetrievalEvent], window_seconds: float
) -> list[list[RetrievalEvent]]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e.ts)
    sessions: list[list[RetrievalEvent]] = [[events[0]]]
    for prev, curr in zip(events, events[1:]):
        if (curr.ts - prev.ts).total_seconds() <= window_seconds:
            sessions[-1].append(curr)
        else:
            sessions.append([curr])
    return sessions


def slugify(text: str, max_words: int = 6, max_len: int = 40) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    words = [w for w in text.split("-") if w]
    slug = "-".join(words[:max_words])[:max_len].rstrip("-")
    return slug or "session"


def session_label(session: list[RetrievalEvent]) -> str:
    """Stable, content-derived, human-readable session name.

    <task-slug>_<MM-DD>_<HHMM>. Derived from the first event's query_text
    and timestamp, so it doesn't shift if unrelated log entries elsewhere
    get deleted, and it tells you what it was without opening anything.
    """
    first_ev = min(session, key=lambda e: e.ts)
    local_ts = first_ev.ts.astimezone(DISPLAY_TZ)
    slug = slugify(first_ev.query_text)
    return f"{slug}_{local_ts.strftime('%m-%d_%H%M')}"


REGISTRY_FILENAME = ".session_registry.json"


def load_registry(out_dir: Path) -> dict[str, str]:
    path = out_dir / REGISTRY_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_registry(out_dir: Path, registry: dict[str, str]) -> None:
    (out_dir / REGISTRY_FILENAME).write_text(json.dumps(registry, indent=2), encoding="utf-8")


def session_key(session: list["RetrievalEvent"]) -> str:
    """Permanent identity for a session -- independent of any display formatting.

    This never changes even if slugify(), the timezone, or the date format
    change later. The registry maps this to whatever label was assigned the
    first time the session was seen.
    """
    first_ev = min(session, key=lambda e: e.ts)
    return first_ev.query_id


def resolve_label(session: list["RetrievalEvent"], registry: dict[str, str]) -> tuple[str, bool]:
    """Returns (label, is_new). Reuses a previously assigned label if known."""
    key = session_key(session)
    if key in registry:
        return registry[key], False
    label = session_label(session)
    registry[key] = label
    return label, True


def shorten_source_file(path: str) -> str:
    if not path:
        return path
    marker = "kernelpack/"
    idx = path.find(marker)
    return path[idx:] if idx != -1 else path


def fetch_payloads(client: QdrantClient, collection: str, point_ids: list[str]) -> dict[str, dict]:
    unique_ids = list(dict.fromkeys(point_ids))
    if not unique_ids:
        return {}
    points = client.retrieve(collection_name=collection, ids=unique_ids, with_payload=True)
    return {str(p.id): (p.payload or {}) for p in points}


def render_candidate(candidate: dict, payload: dict | None, session_count: int) -> dict:
    out = {
        "point_id": candidate["point_id"],
        "fused_rank": candidate["fused_rank"],
        "fused_score": round(candidate["fused_score"], 4),
        "leg_scores": {k: round(v, 4) for k, v in candidate.get("leg_scores", {}).items()},
        "appears_in_n_of_session_queries": session_count,
    }
    if payload is None:
        out["payload"] = "MISSING -- point_id not found in collection"
        return out
    info = {f: payload.get(f) for f in PAYLOAD_FIELDS_TO_SHOW}
    if info.get("source_file"):
        info["source_file"] = shorten_source_file(info["source_file"])
    text = payload.get("text", "")
    info["text_preview"] = text[:TEXT_PREVIEW_CHARS] + ("..." if len(text) > TEXT_PREVIEW_CHARS else "")
    info["text_length"] = len(text)
    out["payload"] = info
    return out


def build_event_record(ev: RetrievalEvent, label: str, payloads: dict, session_counts: dict) -> dict:
    return {
        "query_id": ev.query_id,
        "ts": ev.ts.astimezone(DISPLAY_TZ).isoformat(),
        "session": label,
        "plan": ev.plan,
        "query_text": ev.query_text,
        "fusion": ev.fusion,
        "reranker_id": ev.reranker_id,
        "spaces": ev.spaces,
        "filters": ev.filters,
        "candidates": [
            render_candidate(c, payloads.get(c["point_id"]), session_counts.get(c["point_id"], 1))
            for c in sorted(ev.candidates, key=lambda c: c["fused_rank"])
        ],
    }


def most_useful_chunk(records: list[dict]) -> tuple[str, dict, int] | None:
    """Pick the candidate retrieved most often across the session's queries.

    Tie-break: best (lowest) fused_rank it ever achieved.
    """
    best_by_id: dict[str, dict] = {}
    for rec in records:
        for c in rec["candidates"]:
            pid = c["point_id"]
            count = c["appears_in_n_of_session_queries"]
            rank = c["fused_rank"]
            if pid not in best_by_id or rank < best_by_id[pid]["fused_rank"]:
                best_by_id[pid] = c
            best_by_id[pid]["appears_in_n_of_session_queries"] = count
    if not best_by_id:
        return None
    pid, c = max(
        best_by_id.items(),
        key=lambda kv: (kv[1]["appears_in_n_of_session_queries"], -kv[1]["fused_rank"]),
    )
    return pid, c, c["appears_in_n_of_session_queries"]


def chunk_display_name(candidate: dict) -> str:
    payload = candidate.get("payload")
    if not isinstance(payload, dict):
        return candidate["point_id"][:8] + " (payload missing)"
    fn = payload.get("function_name") or payload.get("chunk_type") or "?"
    src = payload.get("source_file") or "?"
    return f"`{fn}` in `{src}`"


def write_session_readme(session_dir: Path, label: str, first_query_text: str, records: list[dict]) -> None:
    lines = [f"# Session: {label}", "", "## Task the agent was generating for", "", f"> {first_query_text}", ""]

    lines += ["## Retrieval calls", ""]
    for i, rec in enumerate(records, start=1):
        lines.append(f"### Query {i} ({rec['plan']})")
        lines.append("")
        lines.append(f"**Sent:** {rec['query_text']}")
        lines.append("")
        lines.append("| rank | chunk | source_file | fused_score |")
        lines.append("|---|---|---|---|")
        for c in rec["candidates"][:TOP_N_PER_QUERY_IN_README]:
            payload = c.get("payload")
            fn = payload.get("function_name") if isinstance(payload, dict) else "MISSING"
            src = payload.get("source_file") if isinstance(payload, dict) else "-"
            lines.append(f"| {c['fused_rank']} | `{fn}` | `{src}` | {c['fused_score']} |")
        lines.append("")

    top = most_useful_chunk(records)
    lines.append("## Most useful chunk this session")
    lines.append("")
    if top is None:
        lines.append("No candidates retrieved.")
    else:
        pid, candidate, count = top
        lines.append(
            f"{chunk_display_name(candidate)} -- retrieved in {count} of {len(records)} "
            f"queries, best rank {candidate['fused_rank']}."
        )

    (session_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def index_row(label: str, records: list[dict]) -> str:
    first_query = records[0]["query_text"]
    preview = first_query[:50] + "..." if len(first_query) > 50 else first_query
    top = most_useful_chunk(records)
    top_str = chunk_display_name(top[1]) if top else "-"
    ts = records[0]["ts"][:16].replace("T", " ")
    return f"| [{label}]({label}/README.md) | {ts} | {preview} | {len(records)} | {top_str} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--out", type=Path, default=Path("logs/retrieval_readable"))
    parser.add_argument("--session-window", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events = parse_log(args.log_path)
    if not events:
        print("no retrieval events found in log")
        return

    sessions = group_into_sessions(events, args.session_window)
    registry = load_registry(args.out)
    client: QdrantClient | None = None
    index_rows: list[str] = []
    new_count, skipped_count = 0, 0

    for session in sessions:
        label, is_new_session_key = resolve_label(session, registry)
        session_dir = args.out / label

        if session_dir.exists() and not args.force:
            skipped_count += 1
            records = []
            for i, ev in enumerate(session, start=1):
                json_path = session_dir / f"q{i}_{ev.plan}.json"
                if json_path.exists():
                    records.append(json.loads(json_path.read_text(encoding="utf-8")))
            if records:
                index_rows.append(index_row(label, records))
            else:
                print(f"warning: {session_dir} exists but no JSON records found, use --force to rebuild")
            continue

        new_count += 1
        if client is None:
            client = QdrantClient(url=args.qdrant_url)

        all_ids = [c["point_id"] for ev in session for c in ev.candidates]
        session_counts: dict[str, int] = {}
        for pid in all_ids:
            session_counts[pid] = session_counts.get(pid, 0) + 1
        payloads = fetch_payloads(client, args.collection, list(set(all_ids)))

        session_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i, ev in enumerate(session, start=1):
            rec = build_event_record(ev, label, payloads, session_counts)
            (session_dir / f"q{i}_{ev.plan}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            records.append(rec)

        write_session_readme(session_dir, label, session[0].query_text, records)
        index_rows.append(index_row(label, records))

    summary_lines = [
        "# Retrieval sessions index",
        "",
        "| session | when | task | # queries | most useful chunk |",
        "|---|---|---|---|---|",
        *index_rows,
    ]
    (args.out / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    save_registry(args.out, registry)

    print(f"{new_count} session(s) processed, {skipped_count} already on disk (skipped)")
    print(f"index written to {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()