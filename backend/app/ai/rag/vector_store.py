from threading import Lock
from typing import Any

from app.ai.config import get_config


class ChromaVectorStore:
    def __init__(self, path, collection_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB dependencies are not installed") from exc
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def ids(self) -> list[str]:
        return self._collection.get(include=[]).get("ids", [])

    def upsert(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        if ids:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
            )

    def delete_all(self) -> None:
        existing = self._collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def search(self, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
        count = self.count
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )
        if not result.get("ids") or not result["ids"][0]:
            return []
        return [
            {
                "text": text,
                "metadata": metadata or {},
                "distance": distance,
            }
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


_store: ChromaVectorStore | None = None
_lock = Lock()


def get_vector_store() -> ChromaVectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                config = get_config()
                _store = ChromaVectorStore(config.chroma_path, config.chroma_collection)
    return _store


def reset_vector_store() -> None:
    global _store
    with _lock:
        _store = None