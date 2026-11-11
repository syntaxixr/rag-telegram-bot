"""Загрузка документов и нарезка на фрагменты.

Markdown режем по заголовкам, длинные разделы дополнительно по абзацам,
чтобы фрагмент помещался в контекст и при этом не терял смысл.
"""

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 900


@dataclass
class Chunk:
    source: str   # имя файла
    title: str    # заголовок раздела
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} — {self.title}" if self.title else self.source


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pypdf опционален
            raise RuntimeError("Для PDF поставь pypdf: pip install pypdf") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) > MAX_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


def split_markdown(text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title, buf = "", []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            for piece in _split_long(body):
                chunks.append(Chunk(source=source, title=title, text=piece))

    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            flush()
            title, buf = heading.group(1).strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        chunks.extend(split_markdown(_read(path), path.name))
    return chunks
