import logging
import os
from threading import Lock
from pathlib import Path

from app.ai.config import get_config
from app.ai.rag.ingestion import BACKEND_ROOT


logger = logging.getLogger(__name__)


class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._backend = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self.model_name)
                        self._backend = "sentence-transformers"
                    except (ImportError, OSError, RuntimeError) as error:
                        try:
                            from fastembed import TextEmbedding
                        except ImportError as exc:
                            raise RuntimeError("Local embedding dependencies are not available") from exc
                        cache_path = Path(os.getenv("FASTEMBED_CACHE_PATH", "storage/models"))
                        if not cache_path.is_absolute():
                            cache_path = BACKEND_ROOT / cache_path
                        cache_path.mkdir(parents=True, exist_ok=True)
                        try:
                            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache_path))
                        except Exception as exc:
                            raise RuntimeError("The configured local embedding model could not be loaded") from exc
                        self._backend = "fastembed-onnx"
                        logger.warning(
                            "Using local ONNX MiniLM embeddings because sentence-transformers could not initialize (%s)",
                            type(error).__name__,
                        )
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        if self._backend == "fastembed-onnx":
            return [vector.tolist() for vector in model.embed(texts)]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


_embeddings: LocalEmbeddings | None = None


def get_embeddings() -> LocalEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalEmbeddings(get_config().embedding_model)
    return _embeddings


def initialize_embeddings() -> None:
    get_embeddings()._load()