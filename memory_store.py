"""
Lightweight long-term memory for BlogForge AI.

Stores a record of every blog the pipeline has generated (topic, title, tone,
audience, blog_kind, quality score, timestamp) in a local JSON file and lets
the Orchestrator retrieve topically-related past runs before planning a new
one. This gives the agent persistent memory across sessions without pulling
in a vector-database dependency.

Similarity is computed with difflib's SequenceMatcher over the topic string,
which is dependency-free and good enough for "have I written about something
like this before?" style recall. Swap `_similarity` for an embeddings-based
cosine similarity (e.g. with a Chroma/FAISS store) if you want semantic recall
over large histories.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Dict, List

MEMORY_FILE = Path("memory_store.json")
MAX_RECORDS = 200
MIN_SIMILARITY = 0.25


def _load() -> List[Dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(records: List[Dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def save_memory(record: Dict) -> None:
    """Append a new memory record, keeping only the most recent MAX_RECORDS."""
    records = _load()
    records.append(record)
    _save(records[-MAX_RECORDS:])


def get_relevant_memories(topic: str, k: int = 3) -> List[Dict]:
    """Return up to k past records whose topic most resembles the given topic."""
    records = _load()
    if not records:
        return []
    scored = sorted(records, key=lambda r: _similarity(r.get("topic", ""), topic), reverse=True)
    relevant = [r for r in scored if _similarity(r.get("topic", ""), topic) >= MIN_SIMILARITY]
    return relevant[:k]


def all_memories() -> List[Dict]:
    """Return every stored memory record (most recent last)."""
    return _load()


def memory_count() -> int:
    return len(_load())
