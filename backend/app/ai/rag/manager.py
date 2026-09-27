import logging
import os

from app.ai.config import get_config
from app.ai.rag.embeddings import get_embeddings, initialize_embeddings
from app.ai.rag.ingestion import knowledge_base_path, load_knowledge_chunks
from app.ai.rag.vector_store import get_vector_store


logger = logging.getLogger(__name__)
_status = {"ai": "not_configured", "vector_store": "unavailable", "knowledge_base": "unavailable"}


def initialize_agent() -> None:
    config = get_config()
    _status["ai"] = "ok" if config.llm_api_key else "not_configured"
    root = knowledge_base_path()
    _status["knowledge_base"] = "ok" if root.is_dir() and any(root.rglob("*.md")) else "unavailable"
    try:
        get_vector_store()
        _status["vector_store"] = "ok"
        if os.getenv("AI_EAGER_INITIALIZE", "true").strip().lower() not in {"0", "false", "no"}:
            initialize_embeddings()
    except Exception as exc:
        _status["vector_store"] = "unavailable"
        logger.warning("AI retrieval startup initialization unavailable: %s", type(exc).__name__)
    if _status["knowledge_base"] != "ok":
        logger.warning("Knowledge base is missing or contains no Markdown documents")


def reindex_knowledge() -> dict[str, int | bool]:
    documents, chunks = load_knowledge_chunks(
        chunk_size=get_config().chunk_size,
        overlap=get_config().chunk_overlap,
    )
    embeddings = get_embeddings().encode([chunk.text for chunk in chunks]) if chunks else []
    store = get_vector_store()
    indexed_ids = [
        str(chunk.metadata["source"]) + "::" + str(chunk.metadata["chunk_id"])
        for chunk in chunks
    ]
    for offset in range(0, len(chunks), 64):
        current = chunks[offset:offset + 64]
        store.upsert(
            ids=indexed_ids[offset:offset + 64],
            documents=[chunk.text for chunk in current],
            embeddings=embeddings[offset:offset + 64],
            metadata=[chunk.metadata for chunk in current],
        )
    stale_ids = set(store.ids()) - set(indexed_ids)
    store.delete(list(stale_ids))
    _status["vector_store"] = "ok"
    _status["knowledge_base"] = "ok"
    return {"success": True, "documents": documents, "chunks": len(chunks)}


def agent_health() -> dict[str, str]:
    config = get_config()
    root = knowledge_base_path()
    knowledge_status = "ok" if root.is_dir() and any(root.rglob("*.md")) else "unavailable"
    ai_status = "ok" if config.llm_api_key else "not_configured"
    return {
        "backend": "ok",
        "ai": ai_status,
        "vector_store": _status["vector_store"],
        "knowledge_base": knowledge_status,
    }