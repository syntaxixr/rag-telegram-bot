"""Telegram-бот: принимает вопрос, отвечает по документам и показывает источники."""

import asyncio
import html
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message

from .config import settings
from .rag import RAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

rag = RAG()
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Задай вопрос, отвечу по документам компании и покажу, откуда взял ответ."
    )


@dp.message(F.text)
async def question(message: Message) -> None:
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        # Запрос к модели блокирующий, уносим его в поток, чтобы бот не подвисал для других.
        answer = await asyncio.to_thread(rag.ask, message.text)
    except Exception:
        log.exception("Ошибка при ответе")
        await message.answer("Что-то пошло не так, попробуй ещё раз чуть позже.")
        return

    text = html.escape(answer.text)
    if answer.sources:
        lines = [f"[{i}] {html.escape(c.label)}" for i, c in answer.sources]
        text += "\n\n<i>Источники:</i>\n" + "\n".join(lines)
    log.info("q=%r llm=%s tokens=%s+%s %.2fs", message.text, answer.used_llm,
             answer.prompt_tokens, answer.completion_tokens, answer.seconds)
    await message.answer(text)


async def main() -> None:
    if not settings.telegram_token:
        raise RuntimeError("Не задан TELEGRAM_TOKEN (см. .env.example)")
    log.info("Загружено фрагментов: %d", len(rag.chunks))
    bot = Bot(settings.telegram_token, default=DefaultBotProperties(parse_mode="HTML"))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
