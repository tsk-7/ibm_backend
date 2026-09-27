import pytest
from fastapi.testclient import TestClient

from app.ai.tools.dataset_tools import (
    get_dataset_preview,
    get_dataset_profile,
    get_duplicates,
    get_history,
    get_missing_values,
)
from app.ai.tools.health_tools import get_data_health
from app.ai.tools.statistics_tools import get_statistics
from app.ai.tools.visualization_tools import get_visualization_recommendations
from app.ai.llm_client import LLMClient, LLMError
from app.ai.config import get_config
from app.ai.rag.ingestion import load_knowledge_chunks
from app.ai.rag.vector_store import ChromaVectorStore


@pytest.fixture
def agent_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAINSIGHT_DB_PATH", str(tmp_path / "database" / "agent.db"))
    monkeypatch.setenv("DATAINSIGHT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AI_EAGER_INITIALIZE", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    from app.ai.config import get_config
    from app.ai.conversation import reset_conversations
    from app.ai.llm_client import reset_llm_client

    get_config.cache_clear()
    reset_conversations()
    reset_llm_client()
    from app.main import app

    with TestClient(app) as client:
        yield client
    get_config.cache_clear()


def _upload_duplicate_dataset(client, csv=b"Department,Age\nSales,25\nSales,25\nHR,30\n"):
    response = client.post("/api/datasets/upload", files={"file": ("sample.csv", csv, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def test_ingestion_preserves_source_title_and_chunk_bounds(tmp_path):
    source = tmp_path / "data_preprocessing" / "normalization.md"
    source.parent.mkdir()
    source.write_text("# Scaling guide\n\n" + ("Scaling changes numeric features. " * 40), encoding="utf-8")

    documents, chunks = load_knowledge_chunks(tmp_path, chunk_size=300, overlap=40)

    assert documents == 1
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert chunks[0].metadata["title"] == "Scaling guide"
    assert chunks[0].metadata["category"] == "data_preprocessing"
    assert chunks[0].metadata["source"] == "knowledge_base/data_preprocessing/normalization.md"
    assert chunks[0].metadata["chunk_id"] == "normalization_001"


def test_openrouter_configuration_uses_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1/")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "240")
    get_config.cache_clear()

    config = get_config()

    assert config.llm_api_key == "test-openrouter-key"
    assert config.llm_base_url == "https://openrouter.example/v1"
    assert config.llm_model == "nvidia/nemotron-3.5-lightning:free"
    assert config.request_timeout == 180
    get_config.cache_clear()


def test_generated_text_repairs_mojibake_without_changing_unicode():
    from app.ai.agent import _repair_mojibake

    assert _repair_mojibake("Missing rate: 0â€¯%") == "Missing rate: 0\u202f%"
    assert _repair_mojibake("café and normal text") == "café and normal text"


def test_chat_without_dataset_supports_general_questions_and_returns_contract(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])

    response = agent_client.post("/api/agent/chat", json={"message": "What is normalization?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["message"] == body["answer"]
    assert body["citations"] == []
    assert body["requires_confirmation"] is False
    assert body["conversation_id"]


def test_dataset_question_without_active_dataset_is_explicit(agent_client):
    response = agent_client.post("/api/agent/chat", json={"message": "How many missing values are there?"})

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "No dataset is currently selected. Select a dataset to ask about its contents or request a data operation."
    )


def test_general_statistics_question_is_not_mistaken_for_dataset_question(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])

    response = agent_client.post("/api/agent/chat", json={"message": "What is the mean?"})

    assert response.status_code == 200
    assert "No dataset is currently selected" not in response.json()["answer"]


def test_agent_request_id_is_returned(agent_client):
    response = agent_client.post(
        "/api/agent/chat",
        headers={"X-Request-ID": "frontend-turn-42"},
        json={"message": "What is normalization?"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "frontend-turn-42"


def test_read_tool_uses_current_services_and_reports_exact_missing_count(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    dataset_id = _upload_duplicate_dataset(agent_client)

    response = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "How many duplicate rows are in my dataset?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "1 duplicate rows" in body["answer"]
    assert body["tool_calls"][0]["tool"] == "get_duplicates"
    assert body["tool_calls"][0]["status"] == "completed"


def test_destructive_request_requires_one_time_confirmation_and_verifies(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    dataset_id = _upload_duplicate_dataset(agent_client)

    proposal = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "Remove duplicate rows"},
    )
    proposed = proposal.json()
    assert proposal.status_code == 200
    assert proposed["requires_confirmation"] is True
    assert proposed["action"]["type"] == "remove_duplicates"
    assert proposed["action"]["confirmation_id"]
    assert agent_client.get(f"/api/datasets/{dataset_id}/profile").json()["duplicate_rows"] == 1

    confirmation = agent_client.post(
        "/api/agent/confirm",
        json={"confirmation_id": proposed["action"]["confirmation_id"], "confirm": True},
    )

    assert confirmation.status_code == 200
    assert confirmation.json()["success"] is True
    assert confirmation.json()["verified"] is True
    assert confirmation.json()["after"]["duplicate_rows"] == 0
    assert agent_client.get(f"/api/datasets/{dataset_id}/history").json()
    replay = agent_client.post(
        "/api/agent/confirm",
        json={"confirmation_id": proposed["action"]["confirmation_id"], "confirm": True},
    )
    assert replay.json()["success"] is False


def test_confirmation_can_be_cancelled_without_mutation(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    dataset_id = _upload_duplicate_dataset(agent_client)
    proposal = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "Remove duplicate rows"},
    ).json()

    cancelled = agent_client.post(
        "/api/agent/confirm",
        json={"confirmation_id": proposal["action"]["confirmation_id"], "confirm": False},
    )

    assert cancelled.json()["success"] is True
    assert agent_client.get(f"/api/datasets/{dataset_id}/profile").json()["duplicate_rows"] == 1


def test_confirmed_missing_value_action_uses_existing_preprocessing_service(agent_client, monkeypatch):
    import app.ai.agent as agent_module
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    dataset_id = _upload_duplicate_dataset(
        agent_client,
        b"Department,Age\nSales,25\nSales,25\nHR,\n",
    )

    proposal = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "Fill missing Age values using median"},
    ).json()
    assert proposal["requires_confirmation"] is True
    assert proposal["action"]["type"] == "fill_missing_values"
    completed = agent_client.post(
        "/api/agent/confirm",
        json={"confirmation_id": proposal["action"]["confirmation_id"], "confirm": True},
    ).json()

    assert completed["success"] is True
    assert completed["verified"] is True
    assert completed["result"]["missing_before"] == 1
    assert completed["result"]["missing_after"] == 0


def test_dataset_tool_wrappers_delegate_to_existing_analytics_services(agent_client):
    dataset_id = _upload_duplicate_dataset(
        agent_client,
        b"Department,Score\nSales,10\nSales,10\nHR,12\nIT,1000\n,\n",
    )

    profile = get_dataset_profile(dataset_id)
    preview = get_dataset_preview(dataset_id, 1000)
    missing = get_missing_values(dataset_id)
    duplicates = get_duplicates(dataset_id)
    statistics = get_statistics(dataset_id, "Score")
    health = get_data_health(dataset_id)
    charts = get_visualization_recommendations(dataset_id)
    history = get_history(dataset_id)

    assert profile["rows"] == 5
    assert len(preview["rows"]) <= 8
    assert missing["total"] == 2
    assert duplicates["duplicate_rows"] == 1
    assert "Score" in statistics["numerical"]
    assert 0 <= health["overall_score"] <= 100
    assert charts["recommendations"]
    assert history == []


def test_vector_store_upsert_delete_and_retriever_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class FakeCollection:
        entries = {}

        def count(self):
            return len(self.entries)

        def upsert(self, ids, documents, embeddings, metadatas):
            assert len(embeddings) == len(ids)
            for row_id, document, metadata in zip(ids, documents, metadatas):
                self.entries[row_id] = (document, metadata)

        def get(self, include):
            assert isinstance(include, list)
            return {"ids": list(self.entries)}

        def delete(self, ids):
            for row_id in ids:
                self.entries.pop(row_id, None)

        def query(self, query_embeddings, n_results, include):
            assert query_embeddings and "documents" in include
            selected = list(self.entries.items())[:n_results]
            return {
                "ids": [[row_id for row_id, _ in selected]],
                "documents": [[item[0] for _, item in selected]],
                "metadatas": [[item[1] for _, item in selected]],
                "distances": [[0.1 for _ in selected]],
            }

    collection = FakeCollection()
    monkeypatch.setitem(
        __import__("sys").modules,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda path: SimpleNamespace(
            get_or_create_collection=lambda name, metadata: collection,
        )),
    )
    store = ChromaVectorStore(tmp_path / "chroma", "test_collection")
    store.upsert(["chunk-1"], ["A local source chunk"], [[0.1, 0.2]], [{"title": "Guide", "source": "guide.md"}])
    assert store.count == 1

    import app.ai.rag.retriever as retriever
    monkeypatch.setattr(retriever, "get_embeddings", lambda: SimpleNamespace(encode=lambda _texts: [[0.1, 0.2]]))
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)
    result = retriever.retrieve("query", top_k=1)
    assert result[0]["title"] == "Guide"
    assert result[0]["score"] == pytest.approx(0.9)
    store.delete(["chunk-1"])
    assert store.count == 0


def test_openrouter_request_uses_configured_model_and_tool_schema():
    import anyio
    import httpx

    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["referer"] = request.headers.get("HTTP-Referer")
        captured["payload"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Ready."}}]})

    client = LLMClient(
        "test-key",
        "https://openrouter.ai/api/v1",
        "nvidia/nemotron-3.5-lightning:free",
        5,
        transport=httpx.MockTransport(handler),
    )
    tools = [{"type": "function", "function": {"name": "get_data_health"}}]
    result = anyio.run(client.chat, [{"role": "user", "content": "Check quality"}], tools)

    assert result["content"] == "Ready."
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "nvidia/nemotron-3.5-lightning:free"
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == "auto"


def test_openrouter_rate_limit_has_safe_error():
    import anyio
    import httpx

    client = LLMClient(
        "test-key", "https://openrouter.ai/api/v1", "test-model", 2,
        transport=httpx.MockTransport(lambda request: httpx.Response(429)),
    )

    with pytest.raises(LLMError, match="busy"):
        anyio.run(client.chat, [{"role": "user", "content": "hello"}])


def test_openrouter_invalid_key_error_never_exposes_credentials():
    import anyio
    import httpx

    client = LLMClient(
        "private-test-value", "https://openrouter.ai/api/v1", "test-model", 2,
        transport=httpx.MockTransport(lambda request: httpx.Response(401)),
    )

    with pytest.raises(LLMError) as error:
        anyio.run(client.chat, [{"role": "user", "content": "hello"}])
    assert "private-test-value" not in str(error.value)


def test_chat_marks_openrouter_forbidden_response_as_error(agent_client, monkeypatch):
    from dataclasses import replace

    import app.ai.agent as agent_module
    from app.ai.config import get_config

    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    monkeypatch.setattr(agent_module, "get_config", lambda: replace(get_config(), llm_api_key="test-key"))

    class ForbiddenClient:
        async def chat(self, messages, tools=None):
            raise LLMError("OpenRouter denied the request. Verify account access and model permissions.")

    monkeypatch.setattr(agent_module, "get_llm_client", lambda: ForbiddenClient())
    response = agent_client.post("/api/agent/chat", json={"message": "What is normalization?"})

    assert response.status_code == 200
    assert response.json()["error"] is True
    assert response.json()["answer"].startswith("OpenRouter denied the request")


def test_chat_marks_invalid_openrouter_key_response_as_error(agent_client, monkeypatch):
    from dataclasses import replace

    import app.ai.agent as agent_module
    from app.ai.config import get_config

    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    monkeypatch.setattr(agent_module, "get_config", lambda: replace(get_config(), llm_api_key="test-key"))

    class InvalidKeyClient:
        async def chat(self, messages, tools=None):
            raise LLMError("OpenRouter rejected the API key. Verify OPENROUTER_API_KEY.")

    monkeypatch.setattr(agent_module, "get_llm_client", lambda: InvalidKeyClient())
    response = agent_client.post("/api/agent/chat", json={"message": "What is normalization?"})

    assert response.status_code == 200
    assert response.json()["error"] is True
    assert response.json()["answer"].startswith("OpenRouter rejected the API key")


def test_model_tool_call_uses_service_result_and_maps_citation(agent_client, monkeypatch):
    from dataclasses import replace

    import app.ai.agent as agent_module
    from app.ai.config import get_config

    dataset_id = _upload_duplicate_dataset(agent_client)
    knowledge = [{
        "text": "A histogram shows the distribution of numeric observations.",
        "source": "knowledge_base/visualization/histogram.md",
        "title": "Histograms",
        "category": "visualization",
        "chunk_id": "histogram_001",
        "score": 0.91,
    }]
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: knowledge)
    monkeypatch.setattr(agent_module, "get_config", lambda: replace(get_config(), llm_api_key="test-key"))

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def chat(self, _messages, _tools=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_duplicates", "arguments": "{}"},
                    }],
                }
            return {"role": "assistant", "content": "The dataset contains 1 duplicate row. [1]", "tool_calls": []}

    fake_client = FakeClient()
    monkeypatch.setattr(agent_module, "get_llm_client", lambda: fake_client)

    response = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "How many duplicates are present?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1]")
    assert body["tool_calls"] == [{"tool": "get_duplicates", "status": "completed", "duration_ms": body["tool_calls"][0]["duration_ms"]}]
    assert body["citations"][0]["title"] == "Histograms"
    assert body["citations"][0]["chunk_id"] == "histogram_001"


def test_knowledge_reindex_and_search_endpoints(agent_client, monkeypatch):
    import app.api.agent as api_agent

    monkeypatch.setenv("AGENT_ADMIN_TOKEN", "")
    monkeypatch.setattr(api_agent, "reindex_knowledge", lambda: {"success": True, "documents": 2, "chunks": 7})
    monkeypatch.setattr(api_agent, "retrieve", lambda query, top_k: [{"text": "Relevant", "title": "Guide", "source": "knowledge_base/guide.md", "score": 0.8}])

    indexed = agent_client.post("/api/agent/reindex")
    search = agent_client.get("/api/agent/knowledge/search?q=guide&k=3")

    assert indexed.json() == {"success": True, "documents": 2, "chunks": 7}
    assert search.status_code == 200
    assert search.json()["results"][0]["title"] == "Guide"


def test_invalid_dataset_is_not_hidden_as_an_llm_error(agent_client):
    response = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": "DS-NOT-FOUND", "message": "How many missing values?"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found."


def test_agent_tool_loop_stops_at_configured_limit(agent_client, monkeypatch):
    from dataclasses import replace

    import app.ai.agent as agent_module
    from app.ai.config import get_config

    dataset_id = _upload_duplicate_dataset(agent_client)
    monkeypatch.setattr(agent_module, "_retrieved_context", lambda _: [])
    monkeypatch.setattr(agent_module, "get_config", lambda: replace(get_config(), llm_api_key="test-key", max_tool_calls=1))

    class RepeatingToolClient:
        calls = 0

        async def chat(self, _messages, _tools=None):
            self.calls += 1
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call-{self.calls}",
                    "type": "function",
                    "function": {"name": "get_duplicates", "arguments": "{}"},
                }],
            }

    fake_client = RepeatingToolClient()
    monkeypatch.setattr(agent_module, "get_llm_client", lambda: fake_client)
    response = agent_client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "Inspect this dataset."},
    )

    assert response.status_code == 200
    assert fake_client.calls == 2
    assert len(response.json()["tool_calls"]) == 1
    assert "inspection limit" in response.json()["answer"]


def test_write_intents_are_parsed_into_allowlisted_actions(monkeypatch):
    import app.ai.agent as agent_module

    monkeypatch.setattr(agent_module, "dataset_info", lambda _dataset_id: {"column_names": ["Age", "Salary"]})

    assert agent_module._write_intent("Fill missing Age values using median", "DS001") == (
        "fill_missing_values", {"method": "median", "column": "Age"}
    )
    assert agent_module._write_intent("Convert Salary to float", "DS001") == (
        "convert_dtype", {"column": "Salary", "dtype": "float"}
    )
    assert agent_module._write_intent("Rename Salary to AnnualSalary", "DS001") == (
        "rename_column", {"old_name": "Salary", "new_name": "AnnualSalary"}
    )
    assert agent_module._write_intent("Delete the Salary column", "DS001") == (
        "delete_column", {"column": "Salary"}
    )
    assert agent_module._write_intent("Create calculated column Total as Salary * 2", "DS001") == (
        "create_column", {"name": "Total", "expression": "Salary * 2"}
    )