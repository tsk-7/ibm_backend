import os
import re
from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_BASE = BACKEND_ROOT / "knowledge_base"


@dataclass(frozen=True)
class KnowledgeChunk:
    text: str
    metadata: dict[str, str | int]


def knowledge_base_path() -> Path:
    configured = os.getenv("KNOWLEDGE_BASE_PATH")
    return Path(configured).resolve() if configured else DEFAULT_KNOWLEDGE_BASE


def _document_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        heading = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if heading:
            return heading.group(1).strip()
    return path.stem.replace("_", " ").title()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 200:
        raise ValueError("Chunk size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be between zero and chunk size")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start + chunk_size // 2, end)
                    if boundary > start:
                        end = boundary
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap)
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        carry = current[-overlap:].strip() if overlap and current else ""
        current = f"{carry}\n\n{paragraph}" if carry else paragraph
        while len(current) > chunk_size:
            boundary = current.rfind(" ", chunk_size // 2, chunk_size)
            boundary = boundary if boundary > 0 else chunk_size
            chunks.append(current[:boundary].strip())
            current = current[max(0, boundary - overlap):].strip()

    if current:
        chunks.append(current)
    return chunks


def load_knowledge_chunks(
    root: Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[int, list[KnowledgeChunk]]:
    root = root or knowledge_base_path()
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
    overlap = overlap if overlap is not None else int(os.getenv("CHUNK_OVERLAP", "150"))
    if not root.is_dir():
        raise FileNotFoundError("Knowledge base directory is unavailable")

    documents = sorted(root.rglob("*.md"))
    result: list[KnowledgeChunk] = []
    for path in documents:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title = _document_title(content, path)
        relative_path = path.relative_to(root).as_posix()
        category = path.parent.relative_to(root).as_posix()
        if category == ".":
            category = "general"
        for index, text in enumerate(_split_text(content, chunk_size, overlap), start=1):
            result.append(
                KnowledgeChunk(
                    text=text,
                    metadata={
                        "source": f"knowledge_base/{relative_path}",
                        "title": title,
                        "category": category,
                        "file_path": relative_path,
                        "chunk_id": f"{path.stem}_{index:03d}",
                    },
                )
            )
    return len(documents), result