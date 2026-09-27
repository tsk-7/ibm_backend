import hmac
import logging
import os
import time
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.ai.action_manager import cancel_confirmation, execute_confirmation
from app.ai.agent import answer_chat
from app.ai.rag.manager import reindex_knowledge
from app.ai.rag.retriever import retrieve


router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)


class AgentTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    dataset_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=80)
    history: list[AgentTurn] = Field(default_factory=list, max_length=40)


class AgentConfirmRequest(BaseModel):
    confirmation_id: str = Field(min_length=20, max_length=100)
    confirm: bool


@router.post("/chat")
async def chat(request: AgentChatRequest, http_request: Request):
    request_id = http_request.state.request_id
    started = time.perf_counter()
    logger.info("Agent request started request_id=%s dataset_id=%s", request_id, request.dataset_id or "none")
    history = [{"role": turn.role, "content": turn.content} for turn in request.history]
    try:
        response = await answer_chat(request.message, request.dataset_id, request.conversation_id, history, request_id)
        logger.info(
            "Agent request completed request_id=%s duration_ms=%s success=%s",
            request_id, round((time.perf_counter() - started) * 1000, 2), not response["error"],
        )
        return response
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/confirm")
def confirm_action(request: AgentConfirmRequest, http_request: Request):
    if not request.confirm:
        cancelled = cancel_confirmation(request.confirmation_id)
        return {"success": cancelled, "verified": False, "message": "Action cancelled." if cancelled else "This confirmation is invalid or expired."}
    return execute_confirmation(request.confirmation_id, http_request.state.request_id)


@router.post("/reindex")
def reindex(authorization: str | None = Header(default=None)):
    expected_token = os.getenv("AGENT_ADMIN_TOKEN", "")
    if expected_token and not hmac.compare_digest(authorization or "", f"Bearer {expected_token}"):
        raise HTTPException(status_code=403, detail="Not authorized.")
    try:
        return reindex_knowledge()
    except (FileNotFoundError, RuntimeError, ValueError):
        logger.exception("Knowledge reindex failed")
        raise HTTPException(status_code=503, detail="Knowledge indexing is temporarily unavailable.")
    except Exception as exc:
        logger.exception("Knowledge reindex failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Knowledge indexing is temporarily unavailable.") from exc


@router.get("/knowledge/search")
def knowledge_search(q: str = Query(min_length=1, max_length=4000), k: int = Query(default=5, ge=1, le=20)):
    try:
        results = retrieve(q, k)
    except Exception as exc:
        logger.info("Knowledge search unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Knowledge search is temporarily unavailable.") from exc
    return {"results": results}