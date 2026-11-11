"""Настройки из переменных окружения (.env подхватывается, если есть)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Минимальный парсер .env, чтобы не тянуть лишнюю зависимость."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    docs_dir: Path = Path(os.getenv("DOCS_DIR", str(ROOT / "docs")))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Если лучший фрагмент набрал меньше этого балла, в модель не ходим вообще:
    # ответа в документах почти наверняка нет, а токены стоят денег.
    min_score: float = float(os.getenv("MIN_SCORE", "1.5"))
    # Цена за 1M токенов, чтобы считать стоимость в отчёте (по умолчанию 0 = не считаем).
    price_in_per_m: float = float(os.getenv("PRICE_IN_PER_M", "0"))
    price_out_per_m: float = float(os.getenv("PRICE_OUT_PER_M", "0"))


settings = Settings()
