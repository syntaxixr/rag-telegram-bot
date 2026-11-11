"""Поиск фрагментов по BM25.

Своя реализация на ~40 строк вместо векторной базы: для базы знаний
на десятки-сотни страниц этого хватает, работает без GPU и внешних сервисов.
Для русского обрезаем слова до первых 5 букв: «оплата», «оплату»,
«оплатить» превращаются в одну основу «оплат». С 6 буквами «оплатить»
и «оплату» расходились, и поиск промахивался (проверено на eval/questions.jsonl).
"""

import math
import re
from collections import Counter

from .loader import Chunk

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_STOP = {
    "и", "в", "во", "на", "с", "со", "по", "к", "ко", "о", "об", "от", "до", "за",
    "из", "у", "не", "ни", "а", "но", "или", "что", "как", "это", "то", "же", "ли",
    "бы", "мне", "я", "вы", "мы", "он", "она", "они", "его", "ее", "их", "вас", "нас",
    "можно", "есть", "ли", "если", "для", "при", "так", "уже", "ещё", "еще", "какой",
    "какая", "какие", "сколько", "когда", "где",
}
_STEM_LEN = 5


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if word in _STOP or len(word) < 2:
            continue
        tokens.append(word[:_STEM_LEN])
    return tokens


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Заголовок участвует в поиске: он часто точнее описывает раздел, чем текст.
        self.docs = [tokenize(f"{c.title} {c.source} {c.text}") for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, query: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            freq = tf[term]
            norm = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * freq * (self.k1 + 1) / norm
        return score

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = [(self.chunks[i], self._score(q, i)) for i in range(len(self.chunks))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in scored[:top_k] if s > 0]
