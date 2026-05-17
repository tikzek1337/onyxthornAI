# Roadmap: onyxLiteAI Chat

## Этап 1. Smoke test

- Подготовить tokenizer и bins на `data/seed/*.jsonl`.
- Обучить tiny-конфиг 500–2000 шагов.
- Проверить, что модель отвечает в chat-format и не ломает спецтокены.

## Этап 2. Main chat run

- Подготовить данные на `data/generated/*.jsonl`.
- Обучить `configs/onyxliteai_chat_tiny.yaml`.
- Оценить small talk, explanations, practical help, non-coding boundary.

## Этап 3. Quality run

- При наличии VRAM обучить `configs/onyxliteai_chat_small.yaml`.
- Добавить больше реальных очищенных диалогов без персональных данных.
- Сделать evaluation-файл с фиксированными промптами.

## Этап 4. Улучшение поведения

- Добавить больше multi-turn диалогов.
- Добавить предпочтительные ответы и негативные примеры.
- Отдельно усилить честное “не знаю”.
- Отдельно усилить стиль: меньше воды, больше человеческого тона.
