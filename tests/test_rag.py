from pathlib import Path

from app.loader import load_chunks, split_markdown
from app.rag import NOT_FOUND, RAG
from app.llm import LLMReply
from app.retriever import BM25Retriever, tokenize

DOCS = Path(__file__).resolve().parent.parent / "docs"


class FakeLLM:
    def __init__(self, text: str = "Доставка 1–2 дня [1]."):
        self.text = text
        self.calls = 0

    def chat(self, system: str, user: str) -> LLMReply:
        self.calls += 1
        return LLMReply(self.text, prompt_tokens=100, completion_tokens=10)


def test_tokenize_merges_word_forms():
    assert tokenize("доставка")[0] == tokenize("доставку")[0] == tokenize("доставки")[0]


def test_split_by_headings():
    chunks = split_markdown("# A\nтекст а\n## B\nтекст б", "x.md")
    assert [c.title for c in chunks] == ["A", "B"]


def test_retrieval_finds_right_document():
    r = BM25Retriever(load_chunks(DOCS))
    assert r.search("сколько стоит доставка курьером")[0][0].source == "dostavka.md"
    assert r.search("гарантия на электронику")[0][0].source == "garantiya.md"
    assert r.search("вернуть подарочный сертификат")[0][0].source == "vozvrat.md"


def test_no_llm_call_when_nothing_found():
    llm = FakeLLM()
    answer = RAG(llm=llm).ask("кто выиграл чемпионат мира по футболу")
    assert answer.text == NOT_FOUND
    assert llm.calls == 0


def test_answer_has_sources():
    llm = FakeLLM()
    answer = RAG(llm=llm).ask("сколько идёт доставка по Новосибирску")
    assert llm.calls == 1
    assert answer.sources and answer.sources[0][1].source == "dostavka.md"


def test_only_cited_sources_are_shown():
    answer = RAG(llm=FakeLLM("Курьер 300 рублей [2].")).ask("сколько стоит доставка курьером")
    assert [i for i, _ in answer.sources] == [2]


def test_refusal_has_no_sources():
    answer = RAG(llm=FakeLLM(NOT_FOUND)).ask("сколько стоит доставка в Казахстан")
    assert answer.sources == []
