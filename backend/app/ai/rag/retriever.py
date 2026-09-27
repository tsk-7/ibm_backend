import re
from typing import Any

from app.ai.config import get_config
from app.ai.rag.embeddings import get_embeddings
from app.ai.rag.ingestion import load_knowledge_chunks
from app.ai.rag.vector_store import get_vector_store


def _fallback_knowledge_search(query: str, count: int) -> list[dict[str, Any]]:
    try:
        _, chunks = load_knowledge_chunks()
    except Exception:
        return []
    if not chunks:
        return []
    lowered = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
    tokens = [token for token in lowered.split() if len(token) > 2]
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        metadata = chunk.metadata
        text = (chunk.text or "").casefold()
        title = str(metadata.get("title", "")).casefold()
        source = metadata.get("source", "")
        if tokens:
            overlap = sum(1 for token in tokens if token in text or token in title)
            score = overlap / max(len(tokens), 1)
            if overlap == 0:
                score = 0.02
        else:
            score = 0.02
        if score > 0:
            scored.append((score, {
                "text": chunk.text,
                "source": source,
                "title": metadata.get("title", "Knowledge Base"),
                "category": metadata.get("category", "general"),
                "chunk_id": metadata.get("chunk_id", ""),
                "score": score,
            }))
    if not scored:
        first = chunks[0]
        return [{
            "text": first.text,
            "source": first.metadata.get("source", ""),
            "title": first.metadata.get("title", "Knowledge Base"),
            "category": first.metadata.get("category", "general"),
            "chunk_id": first.metadata.get("chunk_id", ""),
            "distance": 0.99,
            "score": 0.01,
        }]
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:count]
    return [{
        "text": item["text"],
        "source": item["source"],
        "title": item["title"],
        "category": item["category"],
        "chunk_id": item["chunk_id"],
        "distance": max(0.0, 1.0 - float(item["score"])),
        "score": max(0.0, min(1.0, float(item["score"]))),
    } for _, item in ranked]


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    count = top_k or get_config().top_k
    try:
        vector = get_embeddings().encode([query])[0]
        results = get_vector_store().search(vector, count)
        if results:
            return [
                {
                    "text": item["text"],
                    "source": item["metadata"].get("source", ""),
                    "title": item["metadata"].get("title", "Knowledge Base"),
                    "category": item["metadata"].get("category", "general"),
                    "chunk_id": item["metadata"].get("chunk_id", ""),
                    "distance": item["distance"],
                    "score": max(0.0, 1.0 - float(item["distance"])),
                }
                for item in results
            ]
    except Exception:
        pass
    return _fallback_knowledge_search(query, count)