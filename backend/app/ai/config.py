import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.ai.rag.ingestion import BACKEND_ROOT


@dataclass(frozen=True)
class AgentConfig:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    top_k: int
    max_tool_calls: int
    max_message_length: int
    chunk_size: int
    chunk_overlap: int
    conversation_turn_limit: int
    preview_row_limit: int
    request_timeout: float


@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    chroma_path = Path(os.getenv("CHROMA_PATH", "storage/chroma"))
    if not chroma_path.is_absolute():
        chroma_path = BACKEND_ROOT / chroma_path
    return AgentConfig(
        llm_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        llm_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        llm_model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chroma_path=chroma_path,
        chroma_collection=os.getenv("CHROMA_COLLECTION", "datainsight_knowledge"),
        top_k=max(1, min(20, int(os.getenv("TOP_K", "5")))),
        max_tool_calls=max(1, min(10, int(os.getenv("MAX_TOOL_CALLS", "5")))),
        max_message_length=max(100, min(10000, int(os.getenv("MAX_MESSAGE_LENGTH", "4000")))),
        chunk_size=max(200, int(os.getenv("CHUNK_SIZE", "1000"))),
        chunk_overlap=max(0, int(os.getenv("CHUNK_OVERLAP", "150"))),
        conversation_turn_limit=max(2, min(40, int(os.getenv("CONVERSATION_TURN_LIMIT", "20")))),
        preview_row_limit=max(1, min(20, int(os.getenv("AGENT_PREVIEW_ROW_LIMIT", "8")))),
        request_timeout=max(1.0, min(180.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "90")))),
    )