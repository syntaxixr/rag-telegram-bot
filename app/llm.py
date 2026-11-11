"""Клиент к любой OpenAI-совместимой модели (DeepSeek, GLM, Qwen, локальная через vLLM/Ollama)."""

from dataclasses import dataclass

from openai import OpenAI

from .config import settings


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLM:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise RuntimeError("Не задан LLM_API_KEY (см. .env.example)")
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                             timeout=60, max_retries=2)
        self.model = settings.llm_model

    def chat(self, system: str, user: str) -> LLMReply:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        usage = resp.usage
        return LLMReply(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
