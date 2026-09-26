from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.dataset_service import dataset_info, dataset_profile
from app.services.health_service import dataset_health
from app.services.statistics_service import calculate_statistics
from app.services.visualization_service import recommendations


router = APIRouter(prefix="/agent", tags=["agent"])


class AgentTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    history: list[AgentTurn] = Field(default_factory=list, max_length=40)


def _contains_any(message: str, terms: tuple[str, ...]) -> bool:
    return any(term in message for term in terms)


@router.post("/chat")
def chat(request: AgentChatRequest):
    info = dataset_info(request.dataset_id)
    message = request.message.strip().lower()
    tools_used = []

    if _contains_any(message, ("duplicate", "duplicated")):
        profile = dataset_profile(request.dataset_id)
        tools_used.append("dataset_profile")
        answer = f"{profile['duplicate_rows']} duplicate rows were found in {info['filename']}."
    elif _contains_any(message, ("missing", "null", "nan")):
        profile = dataset_profile(request.dataset_id)
        tools_used.append("dataset_profile")
        missing = profile["missing_values"]
        columns = [f"{column}: {count}" for column, count in missing["by_column"].items() if count]
        detail = "; ".join(columns) if columns else "No columns have missing values."
        answer = f"There are {missing['total']} missing values. {detail}"
    elif _contains_any(message, ("quality", "health", "score")):
        report = dataset_health(request.dataset_id)
        tools_used.append("dataset_health")
        answer = (
            f"Data quality score: {report['overall_score']}/100. "
            f"Completeness: {report['completeness']}%, consistency: {report['consistency']}%, "
            f"validity: {report['validity']}%."
        )
    elif _contains_any(message, ("statistic", "average", "mean", "median", "maximum", "minimum")):
        statistics = calculate_statistics(request.dataset_id)
        tools_used.append("calculate_statistics")
        numeric = statistics["numerical"]
        if numeric:
            details = [
                f"{column}: mean {values['mean']}, median {values['median']}, "
                f"min {values['minimum']}, max {values['maximum']}"
                for column, values in list(numeric.items())[:8]
            ]
            answer = "Numerical summary: " + "; ".join(details)
        else:
            answer = "This dataset has no numerical columns to summarize."
    elif _contains_any(message, ("chart", "visualization", "visualisation", "plot", "graph")):
        chart_recommendations = recommendations(request.dataset_id)["recommendations"]
        tools_used.append("visualization_recommendations")
        suggestions = [
            f"{item['chart_type']} ({item['x_column']}"
            + (f" vs {item['y_column']}" if item.get("y_column") else "")
            + ")"
            for item in chart_recommendations[:5]
        ]
        answer = (
            f"I found {len(chart_recommendations)} chart recommendations. "
            + ("Examples: " + "; ".join(suggestions) if suggestions else "There are no chart recommendations for these columns.")
        )
    elif _contains_any(message, ("overview", "summar", "shape", "column", "row", "dataset", "data")):
        tools_used.append("dataset_info")
        columns = ", ".join(info["column_names"])
        answer = (
            f"{info['filename']} has {info['rows']} rows and {info['columns']} columns. "
            f"Columns: {columns}."
        )
    else:
        answer = (
            "I can answer questions about this dataset's overview, missing values, duplicate rows, "
            "data quality, numerical statistics, and chart recommendations. General knowledge and "
            "RAG answers require an LLM agent, which is not configured on this backend."
        )

    return {
        "message": answer,
        "dataset_id": request.dataset_id,
        "citations": [],
        "tools_used": tools_used,
        "operations": [],
        "pending_confirmation": None,
    }