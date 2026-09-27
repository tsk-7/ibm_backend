import logging
from typing import Any

import httpx

from app.ai.config import get_config


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Safe, user-displayable error from the configured LLM provider."""


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.transport = transport

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise LLMError("OpenRouter is not configured. Set OPENROUTER_API_KEY in the backend environment.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tsk-7/ibm_backend",
            "X-Title": "IBM DataInsight Studio",
        }
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.warning("OpenRouter request timed out")
            raise LLMError("The AI service timed out. Please try again.") from exc
        except httpx.RequestError as exc:
            logger.warning("OpenRouter connection failed: %s", type(exc).__name__)
            raise LLMError("The AI service is temporarily unavailable.") from exc

        if response.status_code == 401:
            logger.warning("OpenRouter rejected configured credentials")
            raise LLMError("OpenRouter rejected the API key. Verify OPENROUTER_API_KEY.")
        if response.status_code == 403:
            logger.warning("OpenRouter denied the request (HTTP 403)")
            raise LLMError("OpenRouter denied the request. Verify account access and model permissions.")
        if response.status_code == 400:
            logger.warning("OpenRouter rejected the request parameters (HTTP 400)")
            raise LLMError("OpenRouter rejected the request. Verify the configured model and request settings.")
        if response.status_code == 402:
            logger.warning("OpenRouter account has insufficient credits (HTTP 402)")
            raise LLMError("OpenRouter reports insufficient credits for this request.")
        if response.status_code == 429:
            logger.warning("OpenRouter rate limit reached")
            raise LLMError("The AI service is busy. Please try again shortly.")
        if response.is_error:
            logger.warning("OpenRouter returned HTTP %s", response.status_code)
            raise LLMError("The AI service is temporarily unavailable.")
        try:
            data = response.json()
            return data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("OpenRouter returned an invalid response shape")
            raise LLMError("The AI service returned an invalid response.") from exc


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        config = get_config()
        _client = LLMClient(
            config.llm_api_key,
            config.llm_base_url,
            config.llm_model,
            config.request_timeout,
        )
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None