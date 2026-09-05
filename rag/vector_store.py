"""
ChromaDB wrapper: persistent local vector store.
Embeddings are computed via rag.llm_client (OpenAI/Azure) and stored
alongside chunk text + metadata, so Chroma is used purely as a vector
index -- no vendor lock-in on the embedding side.
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

import config
from rag.llm_client import embed_texts


class VectorStore:
    def __init__(self, persist_dir: str = config.CHROMA_DIR, collection_name: str = config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 100):
        """chunks: [{"id", "text", "metadata"}]"""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embed_texts(texts)
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
            )

    def query(
        self,
        query_text: str,
        top_k: int = config.TOP_K_RETRIEVE,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # cosine distance -> similarity
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": similarity,
            })
        return out

    def count(self) -> int:
        return self.collection.count()

    def delete_by_source(self, source_filename: str) -> None:
        """Removes every chunk belonging to a given source file."""
        self.collection.delete(where={"source": source_filename})

    def get_all_documents_summary(self) -> list[dict[str, Any]]:
        """Distinct source documents currently indexed, for the UI's
        metadata-filter dropdowns and the eval dashboard."""
        data = self.collection.get(include=["metadatas"])
        seen = {}
        for m in data["metadatas"]:
            src = m.get("source")
            if src not in seen:
                seen[src] = {
                "source": src,
                "organization": m.get("organization"),   # <-- ADD THIS LINE
                "department": m.get("department"),
                "sensitivity": m.get("sensitivity"),
                "version": m.get("version"),
                "title": m.get("title"),
                }
        return list(seen.values())

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
