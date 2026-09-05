"""
Ingestion manifest: tracks a content hash per source file so re-running
ingestion only embeds files that are new or changed, instead of
re-embedding the entire corpus every time. This is the mechanism behind
ingest.py's default "delta" behavior.

The manifest is a simple JSON file, one per RAG_MODE (free vs paid have
separate vector collections and therefore separate manifests -- see
config.MANIFEST_PATH):

    {
      "data/sample_docs/HR_Leave_Policy.pdf": {
        "hash": "3f9a1c...",
        "chunk_count": 4,
        "last_ingested": "2026-08-16T10:22:31"
      },
      ...
    }
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any


def file_hash(filepath: str) -> str:
    """SHA-256 of the raw file bytes. Any change to content (including a
    metadata header edit) changes this, which is exactly what should
    trigger re-ingestion of that file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(path: str, manifest: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def record_entry(chunk_count: int) -> dict[str, Any]:
    return {"chunk_count": chunk_count, "last_ingested": datetime.utcnow().isoformat()}
