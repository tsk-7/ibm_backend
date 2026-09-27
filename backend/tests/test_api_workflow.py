import hashlib
from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAINSIGHT_DB_PATH", str(tmp_path / "database" / "test.db"))
    monkeypatch.setenv("DATAINSIGHT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AI_EAGER_INITIALIZE", "false")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def upload_sample(client):
    content = (
        "Department,Age,Score,Salary,Joined\n"
        "Sales,25,1,50000,2024-01-01\n"
        "HR,30,2,60000,2024-01-02\n"
        "Sales,,2,70000,2024-01-03\n"
        "HR,30,2,60000,2024-01-02\n"
        "IT,40,100,80000,2024-01-05\n"
    ).encode()
    response = client.post("/api/datasets/upload", files={"file": ("sales.csv", content, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"], content


def upload_visualization_sample(client):
    content = (
        "Category,Value,Other,Joined,Note\n"
        "A,10,1,2024-01-01,\n"
        "B,20,2,2024-01-02,two\n"
        "A,30,3,2024-01-03,three\n"
        "C,40,4,2024-01-04,four\n"
        "B,50,5,2024-01-05,five\n"
        "A,60,6,2024-01-06,six\n"
        "C,70,7,2024-01-07,seven\n"
        "B,80,8,2024-01-08,eight\n"
        "A,90,9,2024-01-09,nine\n"
        "A,90,9,2024-01-09,nine\n"
    ).encode()
    response = client.post("/api/datasets/upload", files={"file": ("visualization.csv", content, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def test_preprocessing_summary_metadata_and_read_only_previews(client):
    dataset_id, _ = upload_sample(client)
    missing_preview = client.post(
        f"/api/datasets/{dataset_id}/preprocess/missing-values/preview",
        json={"column": "Age", "method": "median"},
    )
    assert missing_preview.status_code == 200, missing_preview.text
    assert missing_preview.json()["replacement_value"] == 30
    assert missing_preview.json()["missing_before"] == 1
    assert missing_preview.json()["estimated_missing_after"] == 0

    summary = client.get(f"/api/datasets/{dataset_id}/preprocessing/summary")
    assert summary.status_code == 200
    report = summary.json()
    assert report["rows"] == 5
    assert report["missing"]["total_cells"] == 1
    assert report["missing"]["percentage"] == 4.0
    age = next(column for column in report["columns"] if column["name"] == "Age")
    assert age["kind"] == "numeric"
    assert age["missing_count"] == 1
    assert age["missing_percentage"] == 20.0
    assert age["non_missing_count"] == 4

    column_response = client.get(f"/api/datasets/{dataset_id}/columns")
    assert column_response.status_code == 200
    assert next(item for item in column_response.json()["columns"] if item["name"] == "Department")["kind"] == "categorical"

    duplicate_preview = client.post(f"/api/datasets/{dataset_id}/preprocess/duplicates/preview")
    assert duplicate_preview.status_code == 200
    assert duplicate_preview.json()["duplicate_count"] == 1
    assert duplicate_preview.json()["percentage"] == 20.0
    assert duplicate_preview.json()["estimated_rows_after_removal"] == 4
    assert client.get(f"/api/datasets/{dataset_id}/profile").json()["missing_values"]["total"] == 1


def test_dynamic_visualization_preview_types_aggregation_and_read_only(client):
    dataset_id = upload_visualization_sample(client)
    original = client.get(f"/api/datasets/{dataset_id}/preview?limit=20").json()
    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
    assert profile["total_rows"] == 10
    assert profile["missing_values"]["total"] == 1
    assert profile["duplicate_rows"] == 1

    cases = [
        ({"chart_type": "bar", "column_a": "Category", "column_b": "Value", "aggregation": "mean"}, "Average Value by Category"),
        ({"chart_type": "pie", "column_a": "Category", "column_b": "Value", "aggregation": "sum"}, "Sum Value by Category"),
        ({"chart_type": "scatter", "column_a": "Value", "column_b": "Other"}, "Value vs Other"),
        ({"chart_type": "line", "column_a": "Joined", "column_b": "Value", "aggregation": "mean"}, "Average Value by Joined"),
        ({"chart_type": "box", "column_a": "Category", "column_b": "Value"}, "Value Distribution by Category"),
        ({"chart_type": "histogram", "column_a": "Value", "bins": 5}, "Value Distribution"),
    ]
    for request, expected_title in cases:
        response = client.post(f"/api/datasets/{dataset_id}/visualizations/preview", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["title"] == expected_title
        assert response.json()["dataset_id"] == dataset_id

    bar = client.post(
        f"/api/datasets/{dataset_id}/visualizations/preview",
        json={"chart_type": "bar", "column_a": "Category", "column_b": "Value", "aggregation": "mean"},
    ).json()
    assert bar["labels"] == ["A", "B", "C"]
    assert bar["values"] == [56.0, 50.0, 55.0]
    aggregations = {
        "sum": [280, 150, 110],
        "median": [60.0, 50.0, 55.0],
        "mode": [90, 20, 40],
        "std": [35.77708763999664, 30.0, 21.213203435596427],
        "average": [56.0, 50.0, 55.0],
    }
    for aggregation, expected in aggregations.items():
        aggregated = client.post(
            f"/api/datasets/{dataset_id}/visualizations/preview",
            json={"chart_type": "bar", "column_a": "Category", "column_b": "Value", "aggregation": aggregation},
        )
        assert aggregated.status_code == 200, aggregated.text
        assert aggregated.json()["values"] == pytest.approx(expected)
    histogram = client.post(
        f"/api/datasets/{dataset_id}/visualizations/preview",
        json={"chart_type": "histogram", "column_a": "Value", "bins": 5},
    ).json()
    assert len(histogram["bins"]) == 6
    assert sum(histogram["counts"]) == 10
    assert client.get(f"/api/datasets/{dataset_id}/preview?limit=20").json() == original
    assert client.get(f"/api/datasets/{dataset_id}/history").json() == []


def test_visualization_rejects_invalid_columns_and_chart_combinations(client):
    dataset_id = upload_visualization_sample(client)
    invalid_column = client.post(
        f"/api/datasets/{dataset_id}/visualizations/preview",
        json={"chart_type": "histogram", "column_a": "Missing"},
    )
    assert invalid_column.status_code == 400
    assert "does not exist" in invalid_column.json()["detail"]

    invalid_scatter = client.post(
        f"/api/datasets/{dataset_id}/visualizations/preview",
        json={"chart_type": "scatter", "column_a": "Category", "column_b": "Value"},
    )
    assert invalid_scatter.status_code == 400
    assert invalid_scatter.json()["detail"] == "Scatter plots require two numerical columns."

    invalid_mean = client.post(
        f"/api/datasets/{dataset_id}/visualizations/preview",
        json={"chart_type": "bar", "column_a": "Category", "column_b": "Note", "aggregation": "mean"},
    )
    assert invalid_mean.status_code == 400
    assert "Mean aggregation requires a numerical column" in invalid_mean.json()["detail"]


@pytest.mark.parametrize("method,expected", [("mean", 31.25), ("median", 30), ("mode", 30)])
def test_missing_value_imputation_methods(client, method, expected):
    dataset_id, _ = upload_sample(client)
    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess/missing-values",
        json={"column": "Age", "method": method},
    )
    assert response.status_code == 200, response.text
    assert response.json()["missing_before"] == 1
    assert response.json()["missing_after"] == 0
    preview = client.get(f"/api/datasets/{dataset_id}/preview?limit=5").json()
    age_index = preview["columns"].index("Age")
    assert preview["rows"][2][age_index] == expected


def test_health_upload_preview_profile_and_listing(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["database"] == "ok"
    assert {"backend", "ai", "vector_store", "knowledge_base"}.issubset(health)
    dataset_id, _ = upload_sample(client)

    datasets = client.get("/api/datasets")
    assert datasets.status_code == 200
    assert datasets.json()[0]["filename"] == "sales.csv"

    info = client.get(f"/api/datasets/{dataset_id}").json()
    assert info["rows"] == 5
    assert info["processing_status"] == "uploaded"
    assert "Joined" in info["column_names"]

    preview = client.get(f"/api/datasets/{dataset_id}/preview?limit=2&offset=1").json()
    assert preview["columns"] == ["Department", "Age", "Score", "Salary", "Joined"]
    assert len(preview["rows"]) == 2
    assert preview["total_rows"] == 5

    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
    assert profile["total_rows"] == 5
    assert profile["missing_values"]["total"] == 1
    assert profile["duplicate_rows"] == 1
    assert "Joined" in profile["datetime_columns"]


def test_agent_chat_answers_dataset_questions_and_discloses_limits(client):
    dataset_id, _ = upload_sample(client)

    duplicate_response = client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "How many duplicate rows?", "history": []},
    )
    assert duplicate_response.status_code == 200, duplicate_response.text
    duplicate_answer = duplicate_response.json()
    assert "1 duplicate rows" in duplicate_answer["message"]
    assert duplicate_answer["dataset_id"] == dataset_id
    assert duplicate_answer["tools_used"] == ["dataset_profile"]
    assert duplicate_answer["pending_confirmation"] is None

    unsupported_response = client.post(
        "/api/agent/chat",
        json={"dataset_id": dataset_id, "message": "Explain quantum mechanics", "history": []},
    )
    assert unsupported_response.status_code == 200
    knowledge_answer = unsupported_response.json()
    assert "LLM is not configured" in knowledge_answer["answer"]
    assert knowledge_answer["citations"][0]["source"].startswith("knowledge_base/")


def test_real_data_analysis_preprocessing_history_and_download(client, tmp_path):
    dataset_id, original = upload_sample(client)
    upload_path = tmp_path / "data" / "uploads" / f"{dataset_id}.csv"
    original_hash = hashlib.sha256(upload_path.read_bytes()).hexdigest()

    duplicates = client.post(f"/api/datasets/{dataset_id}/preprocess/duplicates", json={"action": "detect"}).json()
    assert duplicates["duplicate_count"] == 1
    assert duplicates["percentage"] == 20.0

    outliers = client.get(f"/api/datasets/{dataset_id}/outliers")
    assert outliers.status_code == 200
    assert outliers.json()["Score"]["outlier_count"] == 2

    stats = client.get(f"/api/datasets/{dataset_id}/statistics").json()
    assert stats["numerical"]["Score"]["maximum"] == 100
    assert stats["categorical"]["Department"]["unique_values"] == 3

    recommendations = client.get(f"/api/datasets/{dataset_id}/visualization/recommendations").json()["recommendations"]
    assert any(item["chart_type"] == "line" and item["x_column"] == "Joined" for item in recommendations)

    health = client.get(f"/api/datasets/{dataset_id}/health").json()
    assert 0 <= health["overall_score"] <= 100
    assert "scoring" in health

    missing = client.post(
        f"/api/datasets/{dataset_id}/preprocess/missing-values",
        json={"column": "Age", "method": "median"},
    )
    assert missing.status_code == 200, missing.text
    assert missing.json()["missing_before"] == 1
    assert missing.json()["missing_after"] == 0

    removed = client.post(f"/api/datasets/{dataset_id}/preprocess/duplicates", json={"action": "remove"})
    assert removed.json()["rows_after"] == 4
    assert removed.json()["duplicates_removed"] == 1

    renamed = client.post(
        f"/api/datasets/{dataset_id}/columns/rename",
        json={"old_name": "Salary", "new_name": "Annual Salary"},
    )
    assert renamed.status_code == 200, renamed.text

    created = client.post(
        f"/api/datasets/{dataset_id}/columns/create",
        json={"name": "Total", "operation": "Age + Score"},
    )
    assert created.status_code == 200, created.text

    converted = client.post(
        f"/api/datasets/{dataset_id}/preprocess/dtype",
        json={"column": "Age", "dtype": "float"},
    )
    assert converted.status_code == 200, converted.text

    deleted = client.delete(f"/api/datasets/{dataset_id}/columns/Joined")
    assert deleted.status_code == 200, deleted.text
    assert "Joined" not in client.get(f"/api/datasets/{dataset_id}/preview").json()["columns"]

    unsafe = client.post(
        f"/api/datasets/{dataset_id}/columns/create",
        json={"name": "Unsafe", "operation": "__import__('os').system('whoami')"},
    )
    assert unsafe.status_code == 400

    history = client.get(f"/api/datasets/{dataset_id}/history").json()
    assert len(history) == 6
    assert history[0]["operation"] == "Delete column"

    csv_download = client.get(f"/api/datasets/{dataset_id}/download?format=csv")
    assert csv_download.status_code == 200
    downloaded = pd.read_csv(BytesIO(csv_download.content))
    assert "Annual Salary" in downloaded.columns
    assert "Total" in downloaded.columns

    xlsx_download = client.get(f"/api/datasets/{dataset_id}/download?format=xlsx")
    assert xlsx_download.status_code == 200
    assert xlsx_download.content.startswith(b"PK")

    assert hashlib.sha256(upload_path.read_bytes()).hexdigest() == original_hash
    assert upload_path.read_bytes() == original
    assert len(list((tmp_path / "data" / "backups").glob(f"{dataset_id}_*.csv"))) == 6


def test_upload_validation_and_unknown_dataset(client):
    unsupported = client.post("/api/datasets/upload", files={"file": ("data.json", b"{}", "application/json")})
    assert unsupported.status_code == 400
    assert "Unsupported file format" in unsupported.json()["detail"]

    missing = client.get("/api/datasets/not-a-real-id")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Dataset not found"


def test_non_finite_values_are_json_safe(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("non-finite.csv", b"Value\ninf\n1\n", "text/csv")},
    )
    assert response.status_code == 201
    dataset_id = response.json()["dataset_id"]

    preview = client.get(f"/api/datasets/{dataset_id}/preview").json()
    assert preview["rows"][0][0] is None
    outliers = client.get(f"/api/datasets/{dataset_id}/outliers").json()
    assert outliers["Value"]["outlier_count"] == 1