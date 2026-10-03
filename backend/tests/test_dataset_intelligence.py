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


def _upload_dataset(client):
    content = (
        "Department,Age,Score,Salary\n"
        "Sales,25,1,50000\n"
        "HR,30,2,60000\n"
        "Sales,,3,70000\n"
        "HR,30,2,60000\n"
        "IT,40,100,80000\n"
    ).encode()
    response = client.post("/api/datasets/upload", files={"file": ("sales.csv", content, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def test_dataset_profile_includes_per_column_intelligence(client):
    dataset_id = _upload_dataset(client)
    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()

    assert profile["total_rows"] == 5
    assert profile["missing_values"]["total"] == 1
    assert "column_profiles" in profile
    age_profile = next(item for item in profile["column_profiles"] if item["name"] == "Age")
    assert age_profile["null_count"] == 1
    assert age_profile["null_percentage"] == pytest.approx(20.0, abs=0.01)
    assert age_profile["mean"] == pytest.approx(31.25)
    assert age_profile["median"] == 30.0
    assert age_profile["std"] > 0


def test_missing_value_preview_returns_dataset_specific_impact_report(client):
    dataset_id = _upload_dataset(client)
    response = client.post(
        f"/api/datasets/{dataset_id}/preprocess/missing-values/preview",
        json={"column": "Age", "method": "median"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["replacement_value"] == 30
    assert body["rows_affected"] == 1
    assert "before" in body
    assert "after" in body
    assert "impact" in body
    assert body["before"]["missing_count"] == 1
    assert body["after"]["missing_count"] == 0
