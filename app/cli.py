"""Спросить из терминала: python -m app.cli "Сколько идёт доставка?"

Флаг --search показывает только найденные фрагменты, без запроса к модели.
"""

import sys

from .config import settings
from .rag import RAG


def main() -> None:
    args = sys.argv[1:]
    search_only = "--search" in args
    question = " ".join(a for a in args if a != "--search").strip()
    if not question:
        print('Использование: python -m app.cli [--search] "вопрос"')
        sys.exit(1)

    rag = RAG()
    if search_only:
        for chunk, score in rag.retriever.search(question, top_k=settings.top_k):
            print(f"{score:6.2f}  {chunk.label}")
        return

    answer = rag.ask(question)
    print(answer.text)
    for i, c in answer.sources:
        print(f"  [{i}] {c.label}")
    print(f"\nмодель: {'да' if answer.used_llm else 'нет'}, токены: "
          f"{answer.prompt_tokens}+{answer.completion_tokens}, время: {answer.seconds:.2f} с")


if __name__ == "__main__":
    main()
