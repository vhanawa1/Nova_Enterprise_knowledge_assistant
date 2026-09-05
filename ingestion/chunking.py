"""
Chunking strategy.

Uses a recursive character splitter (paragraph -> sentence -> word) so
chunks break on natural boundaries rather than mid-sentence, with a small
overlap to preserve context across chunk edges. Each chunk inherits the
document-level metadata (source, department, sensitivity, version, page)
plus a chunk-level id, so retrieval results can always be traced back to
an exact page and section of an exact document version.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


def _split_text_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple recursive splitter: try paragraphs, then sentences, then hard
    character windows, always respecting chunk_size with overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                # paragraph itself too long -> split on sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= chunk_size:
                        buf = f"{buf} {sent}".strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent[:chunk_size]  # hard cutoff for pathological single sentence
                current = buf
    if current:
        chunks.append(current)

    # apply overlap by prepending tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}".strip())
        chunks = overlapped

    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_SIZE]


def chunk_page_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk a single page/section record into multiple chunk records."""
    pieces = _split_text_recursive(record["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    out = []
    for idx, piece in enumerate(pieces):
        chunk_id = hashlib.md5(
            f"{record['source']}|{record['page']}|{idx}|{piece[:50]}".encode()
        ).hexdigest()[:16]
        meta = dict(record["metadata"])
        meta.update({
            "page": record["page"],
            "chunk_index": idx,
            "chunk_id": chunk_id,
        })
        out.append({"id": chunk_id, "text": piece, "metadata": meta})
    return out


def chunk_documents(page_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_chunks = []
    for rec in page_records:
        all_chunks.extend(chunk_page_record(rec))
    return all_chunks
