# Windows 11 training guide for onyxLiteAI

Команды ниже рассчитаны на запуск из корня проекта. В примерах используются `/` в путях: Windows PowerShell это нормально понимает, зато в Markdown не появляются управляющие символы.

```powershell
cd C:/onyxLiteAI_chat_project
python --version
python -m pip --version
```

## 1. Установка зависимостей

CPU-вариант:

```powershell
python -m pip install -r requirements.txt
```

Для NVIDIA GPU установи подходящую CUDA-сборку PyTorch с официальной страницы PyTorch, затем:

```powershell
python -m pip install -r requirements.txt
```

## 2. Проверка окружения

```powershell
python scripts/verify_install.py
```

## 3. Быстрый smoke test на seed

```powershell
python scripts/prepare_data.py --jsonl data/seed/*.jsonl --tokenizer tokenizer/onyxliteai_tokenizer.json --out_dir data/processed --vocab_size 12000
python -m onyxliteai.train --config configs/onyxliteai_chat_tiny.yaml
```

В `tiny` уже стоит `max_steps: 4500`. Для еще более короткого теста можно так:

```powershell
python -m onyxliteai.train --config configs/onyxliteai_chat_tiny.yaml --max_steps 4000
```

## 4. Основная подготовка данных

```powershell
python scripts/dataset_stats.py --jsonl data/generated/*.jsonl
python scripts/prepare_data.py --jsonl data/generated/*.jsonl --tokenizer tokenizer/onyxliteai_tokenizer.json --out_dir data/processed --vocab_size 20000
```

## 5. Обучение

Tiny:

```powershell
python -m onyxliteai.train --config configs/onyxliteai_chat_tiny.yaml
```

По умолчанию в `configs/onyxliteai_chat_tiny.yaml` стоит `max_steps: 4500`.

Small, если хватает VRAM:

```powershell
python -m onyxliteai.train --config configs/onyxliteai_chat_small.yaml
```

По умолчанию в `configs/onyxliteai_chat_small.yaml` стоит `max_steps: 5000`.

Поменять steps можно двумя способами:

1. Вручную открыть нужный файл в `configs/` и изменить `train -> max_steps`.
2. Временно переопределить из команды:

```powershell
python -m onyxliteai.train --config configs/onyxliteai_chat_tiny.yaml --max_steps 4000
python -m onyxliteai.train --config configs/onyxliteai_chat_small.yaml --max_steps 5000
```

## 6. Генерация одного ответа

```powershell
python -m onyxliteai.generate --checkpoint runs/onyxliteai_chat_tiny/best.pt --tokenizer tokenizer/onyxliteai_tokenizer.json --prompt "Привет, поболтай со мной" --temperature 0.75 --top_k 50 --top_p 0.95 --max_new_tokens 500
```

## 7. Интерактивный чат

```powershell
python -m onyxliteai.chat --checkpoint runs/onyxliteai_chat_tiny/best.pt --tokenizer tokenizer/onyxliteai_tokenizer.json --temperature 0.75 --top_k 50 --top_p 0.95 --max_new_tokens 600
```

Для chat-модели обычно лучше не ставить температуру слишком низко. Диапазон 0.7–0.9 дает более живой ответ, но при мусоре снижай до 0.55–0.7.
