import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import agent, datasets, download, health, history, outliers, preprocessing, statistics, visualization
from app.ai.rag.manager import initialize_agent
from app.database.database import init_db


load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    initialize_agent()
    yield


app = FastAPI(
    title="IBM DataInsight Studio API",
    description="Dataset upload, profiling, preprocessing, and analysis API.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


for api_router in (health.router, datasets.router, agent.router, preprocessing.router, statistics.router,
                   visualization.router, history.router, download.router, outliers.router):
    app.include_router(api_router, prefix="/api")


@app.exception_handler(LookupError)
async def not_found_handler(_: Request, exception: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(exception)})


@app.exception_handler(ValueError)
async def bad_request_handler(_: Request, exception: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exception)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exception: RequestValidationError):
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in exception.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exception: Exception):
    logger.exception("Unexpected API error", exc_info=exception)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
