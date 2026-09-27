# IBM DataInsight Studio Backend

A small FastAPI backend for real dataset upload, inspection, preprocessing, statistics, and chart recommendations. It uses SQLite for metadata and history, and Pandas/NumPy for tabular work. Uploaded originals are retained unchanged; processing works on a separate CSV copy.

## Setup

Requires Python 3.12.

From the `backend/` directory, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to customize storage, AI, CORS, or logging. By default metadata uses SQLite at `database/datainsight.db`; set `DATAINSIGHT_DATABASE_URL` to use MySQL instead, for example `mysql://USER:PASSWORD@127.0.0.1:3306/datainsight`. The backend creates the named database if needed, so the MySQL user needs database-creation permission. `DATAINSIGHT_DB_PATH` applies only to SQLite. Files remain under `data/`. `CORS_ORIGINS` is a comma-separated list; the Vite development origins are enabled by default.

OpenRouter/NVIDIA Nemotron settings are loaded from the repository-root `.env` or `backend/.env`. Set `OPENROUTER_API_KEY`; `OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1`, and `OPENROUTER_MODEL` defaults to `nvidia/nemotron-3.5-lightning:free`. `LLM_TIMEOUT_SECONDS` defaults to 90 (bounded to 180 seconds). Credentials stay on the backend and are never returned or logged. Without a key, the assistant supports bounded deterministic dataset answers and cited RAG excerpts, but no LLM synthesis. First eager startup loads the local `sentence-transformers/all-MiniLM-L6-v2` model and may download it; set `AI_EAGER_INITIALIZE=false` to defer that load until the first RAG request. If Windows blocks Torch, the backend falls back to FastEmbed's local ONNX implementation of the same MiniLM model; its files use `FASTEMBED_CACHE_PATH` (default `storage/models/`).

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`  
Health check: `http://localhost:8000/api/health` (reports connectivity to the configured SQLite or MySQL database)

Run the integration tests from this directory:

```powershell
pytest
```

## API

All application endpoints are under `/api`. JSON requests and responses use UTF-8; download endpoints return file streams.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Backend, database, AI configuration, vector store, and knowledge-base health |
| POST | `/api/agent/chat` | RAG-grounded answer and bounded dataset-tool orchestration |
| POST | `/api/agent/confirm` | Confirm or cancel a short-lived, one-time proposed write action |
| POST | `/api/agent/reindex` | Rebuild persistent local knowledge embeddings (development/admin endpoint) |
| GET | `/api/agent/knowledge/search?q=normalization&k=5` | Search indexed knowledge chunks and metadata |
| POST | `/api/datasets/upload` | Upload CSV, XLSX, or XLS (`multipart/form-data`, field `file`) |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/{dataset_id}` | Dataset information and current processing status |
| GET | `/api/datasets/{dataset_id}/preview?limit=100&offset=0` | Page through current data |
| GET | `/api/datasets/{dataset_id}/profile` | Types, missingness, uniqueness, memory, and duplicates |
| POST | `/api/datasets/{dataset_id}/preprocess/missing-values` | Remove or impute missing values |
| POST | `/api/datasets/{dataset_id}/preprocess/duplicates` | Detect or remove duplicate rows |
| POST | `/api/datasets/{dataset_id}/preprocess/dtype` | Convert a column's data type |
| POST | `/api/datasets/{dataset_id}/columns/rename` | Rename a column |
| DELETE | `/api/datasets/{dataset_id}/columns/{column_name}` | Delete a column |
| POST | `/api/datasets/{dataset_id}/columns/create` | Create a calculated column using safe arithmetic |
| GET | `/api/datasets/{dataset_id}/outliers` | Numerical-column IQR outlier report |
| GET | `/api/datasets/{dataset_id}/statistics` | Numerical and categorical statistics |
| GET | `/api/datasets/{dataset_id}/visualization/recommendations` | Type-based chart suggestions |
| GET | `/api/datasets/{dataset_id}/health` | Deterministic data quality scores and formulas |
| GET | `/api/datasets/{dataset_id}/history` | Processing operations and affected rows |
| GET | `/api/datasets/{dataset_id}/download?format=csv` | Download latest processed data as CSV or XLSX |

The chat request accepts optional `dataset_id` and `conversation_id`, a `message`, and optional bounded `history`. Dataset-specific questions without a selected dataset receive an explicit no-selection response. The structured response contains `answer`, `citations`, `tool_calls`, `requires_confirmation`, `action`, and `conversation_id`; compatibility fields `message`, `tools_used`, and `pending_confirmation` are also returned.

Only read-only tools are exposed to Nemotron. They call the existing DataInsight services; preview rows are capped. A write request is staged with a one-time `confirmation_id` and is not executed by the model. The frontend should show the returned action details and call `/api/agent/confirm` with `{"confirmation_id":"...","confirm":true}` only after the user confirms. Send `confirm:false` to cancel. Tokens expire after ten minutes and are held in process memory, so run one worker for local development; use a shared store before deploying multiple workers. The agent verifies the changed dataset and history before returning success.

The Markdown knowledge base is read recursively from `knowledge_base/`, chunked with `CHUNK_SIZE` and `CHUNK_OVERLAP`, embedded locally, and stored persistently in `storage/chroma/`. Startup initializes the store but does not reindex. After adding or changing documents, call `POST /api/agent/reindex`; optionally set `AGENT_ADMIN_TOKEN` and send it as a Bearer token to protect the development endpoint. `TOP_K`, `MAX_TOOL_CALLS`, `MAX_MESSAGE_LENGTH`, and `EMBEDDING_MODEL` tune retrieval and request limits.

Missing-value methods: `remove_rows`, `mean`, `median`, `mode`, `custom`, `ffill`, and `bfill`. Provide `column` to operate on one column; omit it to operate across the dataframe where the method supports that. A custom value is supplied as `value`.

Duplicate request actions are `detect` and `remove`. Data type values are `integer`, `float`, `string`, `boolean`, and `datetime`. Calculated expressions support column identifiers, numeric constants, parentheses, and `+`, `-`, `*`, `/`; arbitrary code is rejected.

## Example Requests

Upload a dataset:

```powershell
curl.exe -X POST http://localhost:8000/api/datasets/upload -F "file=@sales.csv"
```

Inspect data and fill a missing value:

```powershell
curl.exe http://localhost:8000/api/datasets/DS-0123456789AB/profile
curl.exe -X POST http://localhost:8000/api/datasets/DS-0123456789AB/preprocess/missing-values `
  -H "Content-Type: application/json" -d '{"column":"Age","method":"median"}'
```

Create a calculated column and download the current version:

```powershell
curl.exe -X POST http://localhost:8000/api/datasets/DS-0123456789AB/columns/create `
  -H "Content-Type: application/json" -d '{"name":"Total","operation":"Price * Quantity"}'
curl.exe -o processed.csv "http://localhost:8000/api/datasets/DS-0123456789AB/download?format=csv"
```

Replace the example dataset ID with the ID returned by upload.

Ask a general knowledge question or inspect an active dataset:

```powershell
curl.exe -X POST http://localhost:8000/api/agent/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"What is normalization?"}'
curl.exe -X POST http://localhost:8000/api/agent/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"How many missing values are there?","dataset_id":"DS-0123456789AB"}'
```

Reindex the local knowledge base and search its retrieved chunks:

```powershell
curl.exe -X POST http://localhost:8000/api/agent/reindex
curl.exe "http://localhost:8000/api/agent/knowledge/search?q=normalization&k=5"
```

## Architecture

`app/api/` contains HTTP routers; `app/services/` owns dataset loading, analysis, history, and mutations; `app/ai/` contains OpenRouter/Nemotron orchestration, confirmation handling, service-backed tools, and RAG; `knowledge_base/` contains source Markdown; `storage/chroma/` contains the persistent local vector index; `app/schemas/` defines validated request/response shapes; `app/database/` initializes SQLite or MySQL metadata storage; and `app/utils/` contains file and JSON conversion helpers.

## Data Storage and Safety

Original uploads are stored under `data/uploads/` with unique dataset IDs and are never used as a write target. The initial working copy and all subsequent processed versions are CSV files under `data/processed/`. Before each successful destructive operation, the previous processed CSV is copied to `data/backups/`; the operation and row counts are added to the configured metadata database. Set `DATAINSIGHT_DATA_DIR` and `DATAINSIGHT_DB_PATH` to move those locations.

Health scoring is repeatable and formula-based: completeness is one minus missing cells/all cells; consistency is one minus duplicate rows/all rows; validity is one minus numerical outlier cells/non-missing numerical cells, with outliers defined by 1.5 times IQR; the overall score is the arithmetic mean of those three component scores. The endpoint returns the formulas with its scores.
