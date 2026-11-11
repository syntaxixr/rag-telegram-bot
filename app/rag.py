"""Связка: нашли фрагменты, собрали промпт, спросили модель, вернули ответ с источниками."""

import re
import time
from dataclasses import dataclass, field

from .config import settings
from .loader import Chunk, load_chunks
from .retriever import BM25Retriever

NOT_FOUND = "В документах этого нет. Лучше уточнить у менеджера."

SYSTEM_PROMPT = (
    "Ты помощник поддержки. Отвечай только по фрагментам документов, которые дал пользователь. "
    "Если нужного ответа во фрагментах нет, ответь ровно одной фразой: "
    f"«{NOT_FOUND}» и ничего не придумывай. "
    "После каждого факта ставь номер фрагмента в квадратных скобках, например [2]. "
    "Отвечай коротко, по делу, на русском."
)


@dataclass
class Answer:
    text: str
    # (номер из ответа, фрагмент): только те, на которые модель реально сослалась
    sources: list[tuple[int, Chunk]] = field(default_factory=list)
    used_llm: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @property
    def cost(self) -> float:
        return (self.prompt_tokens * settings.price_in_per_m
                + self.completion_tokens * settings.price_out_per_m) / 1_000_000


def build_user_prompt(question: str, found: list[tuple[Chunk, float]]) -> str:
    blocks = [f"[{i}] ({c.label})\n{c.text}" for i, (c, _) in enumerate(found, 1)]
    return "Фрагменты документов:\n\n" + "\n\n".join(blocks) + f"\n\nВопрос: {question}"


def cited_sources(text: str, found: list[tuple[Chunk, float]]) -> list[tuple[int, Chunk]]:
    """Оставляем только фрагменты, номера которых модель упомянула в ответе: [1], [3]..."""
    ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return [(i, found[i - 1][0]) for i in ids if 1 <= i <= len(found)]


class RAG:
    def __init__(self, llm=None, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_chunks(settings.docs_dir)
        self.retriever = BM25Retriever(self.chunks)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:  # создаём лениво, чтобы поиск работал и без ключа
            from .llm import LLM
            self._llm = LLM()
        return self._llm

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        found = self.retriever.search(question, top_k=settings.top_k)
        if not found or found[0][1] < settings.min_score:
            return Answer(text=NOT_FOUND, seconds=time.perf_counter() - started)

        reply = self.llm.chat(SYSTEM_PROMPT, build_user_prompt(question, found))
        sources = [] if NOT_FOUND in reply.text else cited_sources(reply.text, found)
        return Answer(
            text=reply.text,
            sources=sources,
            used_llm=True,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            seconds=time.perf_counter() - started,
        )
