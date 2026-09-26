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

Copy `.env.example` to `.env` to customize storage locations, CORS origins, or logging. By default metadata uses SQLite at `database/datainsight.db`; set `DATAINSIGHT_DATABASE_URL` to use MySQL instead, for example `mysql://USER:PASSWORD@127.0.0.1:3306/datainsight`. The backend creates the named database if needed, so the MySQL user needs database-creation permission. `DATAINSIGHT_DB_PATH` applies only to SQLite. Files remain under `data/`. `CORS_ORIGINS` is a comma-separated list; the Vite development origins are enabled by default.

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
| GET | `/api/health` | API and configured database health |
| POST | `/api/agent/chat` | Dataset-grounded answers for overview, missing values, duplicates, quality, statistics, and chart recommendations |
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

The chat endpoint accepts `dataset_id`, `message`, and optional `history`. It returns `message`, `dataset_id`, `citations`, `tools_used`, `operations`, and `pending_confirmation` for the frontend contract. It uses the existing analytics services and does not require an LLM key; general-knowledge and RAG responses are not enabled unless a separate agent service is added.

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

## Architecture

`app/api/` contains HTTP routers; `app/services/` owns dataset loading, analysis, history, and mutations; `app/schemas/` defines validated request/response shapes; `app/database/` initializes SQLite or MySQL metadata storage; and `app/utils/` contains file and JSON conversion helpers.

## Data Storage and Safety

Original uploads are stored under `data/uploads/` with unique dataset IDs and are never used as a write target. The initial working copy and all subsequent processed versions are CSV files under `data/processed/`. Before each successful destructive operation, the previous processed CSV is copied to `data/backups/`; the operation and row counts are added to the configured metadata database. Set `DATAINSIGHT_DATA_DIR` and `DATAINSIGHT_DB_PATH` to move those locations.

Health scoring is repeatable and formula-based: completeness is one minus missing cells/all cells; consistency is one minus duplicate rows/all rows; validity is one minus numerical outlier cells/non-missing numerical cells, with outliers defined by 1.5 times IQR; the overall score is the arithmetic mean of those three component scores. The endpoint returns the formulas with its scores.
