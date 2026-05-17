# onyxthornAI Chat

onyxthornAI Chat — переделанная версия проекта onyxthornAI: теперь это chat-only LLM для обычного общения, а не coding-first модель. В проекте заменен системный промпт, переименованы конфиги, обновлены инструкции и добавлен большой синтетический датасет для обучения разговорного поведения.

Цель модели: нормально общаться с пользователем, объяснять темы простыми словами, помогать с бытовыми и учебными задачами, формулировать сообщения, поддерживать в обычных эмоциональных ситуациях, рассуждать аккуратно и не притворяться всезнающей.

Важно: в ZIP нет уже обученных весов. Это готовый проект для подготовки данных, обучения tokenizer, обучения модели и запуска чата локально.

## Что изменено относительно onyxthornAI Coder

- Убран coding-first датасет Python/HTML/CSS/JavaScript.
- Добавлен расширенный chat-only dataset: 118,700 generated-записей + 2,000 seed-записей, включая новый файл на 50,000 уникальных записей про эмоции, дружбу, отношения и прямые ответы.
- Системный промпт переписан под обычное общение.
- Добавлены boundary-примеры: если пользователь просит код, onyxthornAI не должна превращаться в coding assistant.
- README и docs полностью переписаны под обучение и использование onyxthornAI.
- Пакет переименован в `onyxliteai`.
- Добавлен основной конфиг `configs/onyxthornai_chat_17m.yaml`: около 17.16 млн параметров при vocab_size=16000 и контекст 1024 токена.
- Добавлена консольная программа `python -m onyxliteai.studio` для настройки steps/batch/context, оценки времени, подготовки данных, запуска обучения и чата.

## Структура проекта

```text
onyxthornAI_chat_project/
  onyxliteai/
    model.py                 # decoder-only Transformer
    tokenizer.py             # BPE tokenizer
    data.py                  # JSONL -> training corpus -> train/val bins
    train.py                 # training loop
    generate.py              # one-shot generation
    chat.py                  # interactive CLI chat
    studio.py                # удобное меню для обучения и чата
    assistant_response.py    # очистка ответа и fallback для эмоциональных сообщений
    prompting.py             # chat-only system prompt
  configs/
    onyxthornai_chat_17m.yaml
    onyxliteai_chat_tiny.yaml
    onyxliteai_chat_small.yaml
  data/
    seed/onyxliteai_chat_seed_1000.jsonl
    seed/onyxliteai_chat_seed_extra_1000.jsonl
    generated/*.jsonl
    generated/DATASET_MANIFEST.json
    raw/                     # сюда добавлять свои данные
  scripts/
    prepare_data.py
    dataset_stats.py
    verify_install.py
    plot_losses.py
  docs/
    DATASET_FORMAT.md
    MODEL_CARD.md
    ROADMAP.md
    WINDOWS_11_TRAINING_GUIDE.md
```

## Состав датасета

Generated-файлы:

```text
communication_and_rewrites.jsonl: 6000 records
decision_reasoning_and_opinions.jsonl: 5400 records
emotional_support_conversations.jsonl: 4200 records
general_knowledge_explanations.jsonl: 6200 records
practical_life_help.jsonl: 4600 records
safety_and_non_coding_boundaries.jsonl: 3500 records
smalltalk_and_clarifying_questions.jsonl: 8800 records
extra_smalltalk_deep_conversation_5000.jsonl: 5000 records
extra_entertainment_movies_series_anime_5000.jsonl: 5000 records
extra_general_knowledge_culture_5000.jsonl: 5000 records
extra_practical_life_and_decisions_5000.jsonl: 5000 records
extra_emotional_support_5000.jsonl: 5000 records
extra_communication_rewrites_5000.jsonl: 5000 records
onyxthornai_empathy_dialogues_50000.jsonl: 50000 records
```

Категории:

```text
knowledge/general_explanation: 6200
conversation/emotional_support_extra: 5000
knowledge/general_culture_everyday: 5000
conversation/deep_smalltalk: 5000
assistant/practical_life_help: 4600
conversation/emotional_support: 4200
assistant/practical_life_help_extra: 4000
communication/message_drafting_extra: 3750
conversation/smalltalk: 3700
reasoning/decision_help: 3200
communication/message_drafting: 3000
language/style_rewrite: 3000
conversation/clarifying_questions: 2600
conversation/english_general_chat: 2500
reasoning/balanced_opinion: 2200
boundary/non_coding_identity: 1900
safety/supportive_boundaries: 1600
entertainment/фильм: 1525
entertainment/сериал: 1321
entertainment/аниме: 1320
language/style_rewrite_extra: 1250
reasoning/decision_help_extra: 1000
entertainment/recommendation: 834
```

Датасет учит модель:

- приветствию и small talk;
- поддержанию обычной беседы;
- объяснению разных тем простыми словами;
- базовому разговору про фильмы, сериалы и аниме без спойлеров;
- бытовым планам и чеклистам;
- эмоциональной поддержке без псевдотерапии;
- формулированию сообщений: отказ, извинение, просьба, поддержка;
- переписыванию текста в другом тоне;
- сравнению вариантов и принятию решений;
- честному уточнению, когда контекста мало;
- аккуратным safety-ответам;
- отказу от роли coding-модели.


## Самый удобный запуск

```powershell
python -m onyxliteai.studio
```

В Studio можно до старта обучения менять `steps`, `batch_size`, `grad_accum_steps` и размер контекста, видеть примерное время обучения в секундах, примерный размер модели, количество параметров, путь к tokenizer и папку чекпоинтов. После обучения там же можно открыть чат, поменять параметры генерации через `/settings`, посмотреть `/info`, перезагрузить модель через `/reload` и сбросить диалог через `/reset`.

Для обучения основной версии на 15–20 млн параметров используй:

```powershell
python scripts/prepare_data.py --jsonl data/seed/*.jsonl data/generated/*.jsonl --tokenizer tokenizer/onyxthornai_tokenizer.json --out_dir data/processed --vocab_size 16000
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml
```

## Установка

```powershell
cd C:/onyxthornAI_chat_project
python -m pip install -r requirements.txt
python scripts/verify_install.py
```

Если используешь NVIDIA GPU, установи CUDA-сборку PyTorch, подходящую под твою систему, затем снова выполни установку зависимостей.

## Быстрый smoke test

Smoke test нужен, чтобы проверить pipeline на маленьком датасете.

```powershell
python scripts/prepare_data.py --jsonl data/seed/*.jsonl --tokenizer tokenizer/onyxthornai_tokenizer.json --out_dir data/processed --vocab_size 12000
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml
```

В текущем ZIP `17m` уже настроен на `max_steps: 6000`, чтобы быстрые тесты на слабом GPU вроде RTX 3050 не шли слишком долго. Если нужно еще быстрее, можно временно переопределить шаги прямо из команды:

```powershell
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml --max_steps 4000
```

Если всё работает, появится папка `runs/onyxthornai_chat_17m/` с чекпоинтами и `training_log.csv`.

## Основная подготовка данных

```powershell
python scripts/dataset_stats.py --jsonl data/generated/*.jsonl
python scripts/prepare_data.py --jsonl data/generated/*.jsonl --tokenizer tokenizer/onyxthornai_tokenizer.json --out_dir data/processed --vocab_size 20000
```

Что делает `prepare_data.py`:

1. читает JSONL;
2. превращает записи в chat-format со спецтокенами;
3. обучает BPE tokenizer;
4. кодирует данные в токены;
5. создает `data/processed/train.bin`, `val.bin`, `meta.json`.

## Обучение

Основной 17M-конфиг:

```powershell
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml
```

В файле `configs/onyxthornai_chat_17m.yaml` сейчас стоит:

```yaml
train:
  max_steps: 6000
```

Альтернативный small-конфиг теперь тоже уменьшен до диапазона 15–20M:

```powershell
python -m onyxliteai.train --config configs/onyxliteai_chat_small.yaml
```

В файле `configs/onyxliteai_chat_small.yaml` сейчас стоит:

```yaml
train:
  max_steps: 5000
```

Как самому менять количество шагов:

1. Открой нужный YAML-файл в папке `configs/`.
2. Найди блок `train:`.
3. Измени строку `max_steps: 4500` или `max_steps: 5000` на нужное число.
4. Сохрани файл и запусти обучение той же командой.

Быстрый способ без редактирования YAML:

```powershell
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml --max_steps 4000
python -m onyxliteai.train --config configs/onyxliteai_chat_small.yaml --max_steps 5000
```

Для RTX 3050 обычно разумно держать быстрые тесты в диапазоне `4000–5000` шагов. `tiny` подходит для проверки идеи и слабого GPU. `small` лучше для качества, но требует больше VRAM и времени.

## Запуск после обучения

Один ответ:

```powershell
python -m onyxliteai.generate --checkpoint runs/onyxthornai_chat_17m/best.pt --tokenizer tokenizer/onyxthornai_tokenizer.json --prompt "Привет, поболтай со мной" --temperature 0.75 --top_k 50 --top_p 0.95 --max_new_tokens 500
```

Интерактивный чат:

```powershell
python -m onyxliteai.chat --checkpoint runs/onyxthornai_chat_17m/best.pt --tokenizer tokenizer/onyxthornai_tokenizer.json --temperature 0.75 --top_k 50 --top_p 0.95 --max_new_tokens 600
```

Команды внутри CLI:

```text
/exit
exit
quit
q
```

## Рекомендуемые параметры генерации

Для обычного общения:

```text
temperature: 0.70–0.90
top_p:       0.90–0.97
top_k:       40–80
max tokens:  400–800
```

Если модель несет мусор — снижай temperature. Если отвечает сухо — немного повышай temperature.

## Как добавлять свои данные

Положи JSONL в `data/raw/`, например:

```json
{"instruction":"Поддержи меня, я устал","input":"","output":"Похоже, день правда выжал тебя. Давай не требовать от себя невозможного..."}
```

Потом обучай так:

```powershell
python scripts/prepare_data.py --jsonl data/generated/*.jsonl data/raw/*.jsonl --tokenizer tokenizer/onyxthornai_tokenizer.json --out_dir data/processed --vocab_size 20000
python -m onyxliteai.train --config configs/onyxthornai_chat_17m.yaml --max_steps 6000
```

## Как не испортить chat-only характер

Не добавляй много кода. Даже 10–20% coding-данных могут заметно сдвинуть маленькую модель в сторону программирования. Если хочешь, чтобы onyxthornAI оставалась моделью для общения, держи программирование на уровне boundary-примеров: “я не coding-модель, могу объяснить идею без готового кода”.

Хороший баланс:

```text
общие объяснения:        20–25%
обычное общение:         15–20%
практическая помощь:     15–20%
сообщения и стиль:       10–15%
эмоциональная поддержка: 10–15%
reasoning/decision:      10%
safety/boundaries:       5–10%
кодинг:                  0–2%
```

## Проверка качества модели

После обучения проверь такие промпты:

```text
Привет, поболтай со мной.
Мне тревожно, но я не понимаю почему.
Объясни инфляцию простыми словами.
Что посмотреть вечером: фильм, сериал или аниме?
Объясни без спойлеров, чем цепляет Тетрадь смерти.
Помоги выбрать подарок другу.
Сформулируй мягкий отказ.
Что думаешь про удаленку?
Напиши мне Python-скрипт.
У меня болит грудь, скажи диагноз.
```

Желаемое поведение:

- модель поддерживает разговор;
- объясняет без перегруза;
- задает уточнение, когда данных мало;
- не пишет полноценный код;
- не ставит диагнозы;
- не обещает финансовых или юридических гарантий;
- не делает вид, что знает текущие новости без доступа к ним.

## Реалистичное ожидание

Этот проект — база для локальной маленькой LLM. Чтобы получить реально сильную “модель, которая за все шарит”, нужны:

- больше качественных данных;
- больше проверенных multi-turn диалогов;
- отдельные eval-наборы по фильмам, сериалам, аниме и общекультурным темам;
- несколько итераций обучения;
- ручная проверка плохих ответов;
- возможно, fine-tuning более крупной базовой модели, если цель — качество уровня современных ассистентов.

Но текущий ZIP уже переводит проект из coding-first в chat-only и дает большой стартовый разговорный датасет.
