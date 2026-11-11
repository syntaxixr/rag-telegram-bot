"""Проверка качества бота на наборе вопросов.

    python -m eval.run_eval                  # поиск + ответы модели
    python -m eval.run_eval --retrieval-only # только поиск, без модели и без затрат

Что считаем:
  поиск   — попал ли нужный документ в первый результат (hit@1) и в top-k (hit@k);
  ответ   — есть ли в ответе нужные факты и ссылка на источник [n];
  отказ   — на вопросы, ответа на которые в документах нет, бот должен честно отказать;
  деньги  — токены, время ответа, стоимость (если в .env заданы цены).
"""

import json
import re
import sys
from pathlib import Path

from app.config import settings
from app.rag import NOT_FOUND, RAG

HERE = Path(__file__).parent


def has_all(text: str, must: list[str]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in group.split("|")) for group in must)


def main() -> None:
    retrieval_only = "--retrieval-only" in sys.argv
    cases = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rag = RAG()

    rows, hit1 = [], 0
    hitk = ans_ok = refuse_ok = cited = 0
    answerable = [c for c in cases if c.get("answerable", True)]
    unanswerable = [c for c in cases if not c.get("answerable", True)]
    tokens_in = tokens_out = 0
    seconds, cost = [], 0.0

    for case in cases:
        q = case["q"]
        found = rag.retriever.search(q, top_k=settings.top_k)
        sources = [c.source for c, _ in found]
        row = {"q": q}

        if case.get("answerable", True):
            h1 = bool(sources) and sources[0] == case["source"]
            hk = case["source"] in sources
            hit1 += h1
            hitk += hk
            row["поиск"] = "✅" if h1 else ("🟡" if hk else "❌")
        else:
            row["поиск"] = "—"

        if not retrieval_only:
            a = rag.ask(q)
            tokens_in += a.prompt_tokens
            tokens_out += a.completion_tokens
            seconds.append(a.seconds)
            cost += a.cost
            if case.get("answerable", True):
                ok = has_all(a.text, case["must"]) and NOT_FOUND not in a.text
                has_cite = bool(re.search(r"\[\d+\]", a.text))
                ans_ok += ok
                cited += has_cite
                row["ответ"] = ("✅" if ok else "❌") + (" [n]" if has_cite else "")
            else:
                ok = NOT_FOUND in a.text
                refuse_ok += ok
                row["ответ"] = ("✅ отказ" if ok else "❌ выдумал") + ("" if a.used_llm else " (без модели)")
            row["текст"] = a.text.replace("\n", " ")[:120]
        rows.append(row)

    n = len(answerable)
    lines = [f"# Отчёт проверки\n", f"Модель: `{settings.llm_model}`, top_k={settings.top_k}, min_score={settings.min_score}\n",
             "| Метрика | Значение |", "|---|---|",
             f"| Поиск: нужный документ первым (hit@1) | {hit1}/{n} |",
             f"| Поиск: нужный документ в top-{settings.top_k} | {hitk}/{n} |"]
    if not retrieval_only:
        avg = sum(seconds) / len(seconds) if seconds else 0
        lines += [f"| Ответ содержит нужные факты | {ans_ok}/{n} |",
                  f"| Ответ со ссылкой на источник | {cited}/{n} |",
                  f"| Честный отказ, когда ответа нет | {refuse_ok}/{len(unanswerable)} |",
                  f"| Среднее время ответа | {avg:.2f} с |",
                  f"| Токены (вход + выход) | {tokens_in} + {tokens_out} |"]
        if cost:
            lines.append(f"| Стоимость прогона | {cost:.4f} |")

    cols = ["q", "поиск"] + ([] if retrieval_only else ["ответ", "текст"])
    lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |" for r in rows]

    report = "\n".join(lines) + "\n"
    print(report)
    if not retrieval_only:
        (HERE / "report.md").write_text(report, encoding="utf-8")
        print(f"Сохранено в {HERE / 'report.md'}")


if __name__ == "__main__":
    main()
