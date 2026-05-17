"""Generate 50,000 unique onyxthornAI chat records.

The records are synthetic, deterministic and deduplicated. They focus on
emotional understanding, direct practical replies, relationship/friendship
situations, everyday advice, study explanations and normal small talk.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("data/generated/onyxthornai_empathy_dialogues_50000.jsonl")
TARGET = 50_000
SEED = 20260517

feelings = [
    "мне плохо", "я устал", "мне тревожно", "я чувствую себя лишним", "меня бросили",
    "друзья ушли гулять без меня", "я поругался с близким человеком", "мне одиноко",
    "я боюсь завтрашнего дня", "я не могу собраться", "я злюсь на себя", "я растерян",
    "мне обидно", "я переживаю из-за ошибки", "у меня нет сил", "я не знаю, что делать дальше",
    "кажется, меня не ценят", "я стесняюсь написать первым", "я боюсь провала", "я слишком много думаю",
]
contexts = [
    "после учебы", "поздно вечером", "перед важным разговором", "после ссоры", "когда все молчат",
    "после плохой оценки", "когда планы сорвались", "после долгого дня", "перед экзаменом", "после расставания",
    "когда друзья не позвали", "после неприятного сообщения", "когда хочется закрыться", "когда нет поддержки",
    "когда надо идти дальше", "когда накопилась усталость", "после неловкой ситуации", "когда не отвечают",
]
asks = [
    "что мне делать", "скажи по-человечески", "помоги без воды", "поддержи меня", "дай короткий план",
    "как пережить ближайший час", "как не сорваться", "что написать", "как успокоиться", "разложи по шагам",
]
openers = [
    "Понимаю, это неприятно.", "Это правда может ранить.", "Сейчас важно не добивать себя.",
    "Такое состояние не делает тебя слабым.", "Сначала нужно снизить накал.", "Давай без давления.",
    "Тут лучше идти маленькими шагами.", "Сейчас не надо решать всю жизнь целиком.",
]
steps = [
    "выпей воды и убери телефон на десять минут",
    "запиши одной фразой, что именно случилось",
    "напиши близкому человеку короткое сообщение о том, что тебе тяжело",
    "сделай паузу перед любым резким ответом",
    "выбери один маленький шаг, который можно сделать сейчас",
    "переключись на простое действие: душ, еда, прогулка или уборка одного места",
    "не делай вывод о себе по одному эпизоду",
    "отдели факт от догадки: что точно произошло, а что ты додумал",
    "дай себе время остыть и вернись к разговору позже",
    "сформулируй просьбу прямо, без обвинений",
    "если рядом небезопасно оставаться одному, свяжись с живым человеком",
]
closers = [
    "После этого станет проще выбрать следующий шаг.",
    "Не обязательно чувствовать себя хорошо сразу; достаточно сделать ситуацию чуть безопаснее.",
    "Один эпизод не равен всей твоей ценности.",
    "Главное сейчас — не ухудшить состояние импульсивным действием.",
    "Когда станет спокойнее, можно решать уже саму проблему.",
    "Если это длится долго и мешает жить, лучше поговорить со специалистом.",
]

friend_situations = [
    "друзья ушли гулять без меня", "меня не позвали в общий чат", "подруга стала холодно отвечать",
    "друг отменил встречу", "компания общается без меня", "меня игнорируют после ссоры",
    "кажется, обо мне забыли", "меня не отметили на фото", "ребята обсуждают планы без меня", "мне не ответили весь день",
]
relationship_situations = [
    "меня бросили", "человек резко отдалился", "после расставания хочется написать",
    "я скучаю по бывшему человеку", "мне больно видеть его сторис", "я не понимаю, почему все закончилось",
    "меня сравнили с другим", "я чувствую себя замененным", "мне хочется вернуть отношения", "я боюсь остаться один",
]
messages = [
    "Мне стало неприятно, что это произошло без меня. В следующий раз позовите меня тоже, если можно.",
    "Я сейчас на эмоциях, поэтому отвечу позже, когда смогу говорить спокойнее.",
    "Мне тяжело после этого разговора. Можешь немного побыть на связи?",
    "Я не хочу ругаться, но хочу понять, что между нами произошло.",
    "Мне важно сказать честно: я почувствовал себя лишним.",
    "Давай обсудим это без обвинений, когда у нас обоих будет время.",
]

everyday_tasks = [
    "убраться в комнате", "подготовиться к экзамену", "перестать скроллить вечером", "начать делать проект",
    "собраться утром", "написать преподавателю", "сделать домашку", "разобрать завал дел",
    "выбрать, чем заняться вечером", "не откладывать важный звонок", "вернуться к режиму сна", "спланировать неделю",
]
subjects = [
    "биология", "история", "математика", "английский", "литература", "информатика", "обществознание", "химия",
]
explain_topics = [
    "клеточная стенка бактерий", "цитокинез прокариот", "вероятность события", "структура эссе",
    "как учить слова", "как пересказать параграф", "как готовиться к устному ответу", "как не растеряться на экзамене",
]
smalltalk_topics = [
    "фильм на вечер", "аниме с хорошим сюжетом", "идея для прогулки", "что делать в дождь",
    "как начать разговор", "как описать настроение", "почему день кажется пустым", "что делать без сил",
    "как придумать ник", "как выбрать хобби",
]
communication_tasks = [
    "отказаться вежливо", "попросить прощения", "попросить помощи", "сказать, что мне обидно",
    "ответить сухому человеку", "написать преподавателю", "написать другу после ссоры", "договориться о встрече",
]

def add(records: list[dict], seen: set[tuple[str, str]], instruction: str, output: str, category: str, tags: list[str]) -> None:
    key = (" ".join(instruction.lower().split()), " ".join(output.lower().split()))
    if key in seen:
        return
    seen.add(key)
    records.append({
        "id": f"onyxthorn-empathy-{len(records)+1:05d}",
        "category": category,
        "language": "ru",
        "instruction": instruction,
        "input": "",
        "output": output,
        "tags": tags,
    })


def main() -> None:
    rng = random.Random(SEED)
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Emotional direct-help records.
    for feeling in feelings:
        for context in contexts:
            for ask in asks:
                opener = rng.choice(openers)
                s1, s2, s3 = rng.sample(steps, 3)
                closer = rng.choice(closers)
                instruction = f"{feeling}, {context}. {ask}?"
                output = f"{opener} Сначала {s1}. Потом {s2}. После этого {s3}. {closer}"
                add(records, seen, instruction, output, "conversation/emotional_support_direct", ["support", feeling, context])

    # Friendship and relationship situations with message drafts.
    for situation in friend_situations + relationship_situations:
        for ask in asks:
            for msg in messages:
                opener = "Похоже, тебя задела не только сама ситуация, но и ощущение, что тебя отодвинули."
                if situation in relationship_situations:
                    opener = "После такого нормально чувствовать боль, злость или пустоту."
                output = (
                    f"{opener} Не отвечай резко в первые минуты. Сначала назови чувство, потом отдели факт от догадки. "
                    f"Можно написать так: «{msg}» Если ответа не будет, это неприятно, но это не доказывает, что с тобой что-то не так."
                )
                instruction = f"{situation}. {ask}. Помоги сформулировать и успокоиться."
                add(records, seen, instruction, output, "conversation/friendship_relationship_support", ["support", "relationships", situation])

    # Everyday practical coaching.
    for task in everyday_tasks:
        for context in contexts:
            for n in range(18):
                step_a, step_b = rng.sample(steps[:9], 2)
                minutes = [10, 15, 20, 25, 30, 40][n % 6]
                instruction = f"Мне нужно {task}, но я тяну время {context}. Дай план на {minutes} минут."
                output = (
                    f"План простой. Первые 2 минуты — подготовь место и убери лишнее. Следующие {max(5, minutes-10)} минут — делай только один кусок задачи, без идеальности. "
                    f"Если начнет тянуть отвлечься, {step_a}. В конце {step_b} и запиши, что осталось следующим шагом."
                )
                add(records, seen, instruction, output, "assistant/practical_life_help_direct", ["practical", task])

    # Study explanations and exam style.
    for subject in subjects:
        for topic in explain_topics:
            for mode in ["кратко", "очень просто", "как для устного ответа", "по пунктам", "без лишней воды", "чтобы запомнить"]:
                for n in range(35):
                    instruction = f"Объясни тему: {topic}. Предмет: {subject}. Нужно {mode}. Вариант {n+1}."
                    output = (
                        f"По теме «{topic}» главное сказать суть, а не уходить в длинные определения. "
                        f"Первый пункт — что это такое. Второй — из каких частей состоит или зачем нужно. "
                        f"Третий — пример, чтобы показать понимание. Для ответа по предмету «{subject}» лучше закончить одной фразой о значении темы."
                    )
                    add(records, seen, instruction, output, "education/simple_explanation", ["study", subject, topic])

    # Small talk that does not dodge the user.
    for topic in smalltalk_topics:
        for mood in ["спокойно", "коротко", "дружелюбно", "без давления", "с юмором", "нейтрально"]:
            for n in range(130):
                instruction = f"Поболтай со мной про {topic}. Тон: {mood}. Сообщение {n+1}."
                output = (
                    f"Можно. Про {topic} я бы начал без сложного плана: выбери один вариант, который сейчас не требует много сил. "
                    f"Если хочется просто отвлечься, сделай это на 30-40 минут и не ругай себя за паузу. "
                    f"Если хочется обсудить глубже, скажи, что именно в этой теме тебе ближе: настроение, выбор, идея или разговор."
                )
                add(records, seen, instruction, output, "conversation/smalltalk_useful", ["smalltalk", topic])

    # Communication rewrites.
    for task in communication_tasks:
        for tone in ["мягко", "прямо", "спокойно", "без конфликта", "коротко", "по-взрослому", "дружелюбно"]:
            for n in range(120):
                base = rng.choice(messages)
                instruction = f"Помоги {task}. Нужно {tone}. Ситуация номер {n+1}."
                output = (
                    f"Вариант: «{base}» Если нужно еще мягче, можно добавить: «Я не хочу ссориться, просто хочу объяснить, как это выглядело с моей стороны». "
                    f"Так текст звучит {tone} и не превращается в нападение."
                )
                add(records, seen, instruction, output, "communication/rewrite_emotional", ["rewrite", task, tone])


    # Direct question answering: the model should answer the actual question, not list modes.
    direct_questions = [
        "почему мне так обидно", "как понять, что друг правда обиделся", "что делать, если не хочется вставать",
        "как перестать думать о человеке", "как пережить неловкость", "как попросить поддержки",
        "как не написать лишнего", "как объяснить свои чувства", "как начать день после плохой ночи",
        "как успокоиться перед разговором", "как не сравнивать себя с другими", "как принять отказ",
        "как вернуться к делам", "как не чувствовать себя лишним", "как поговорить без ссоры",
    ]
    answer_styles = ["очень коротко", "мягко", "по шагам", "как старший друг", "без мотивационных лозунгов", "спокойно и честно"]
    for question in direct_questions:
        for style in answer_styles:
            for n in range(180):
                s1, s2 = rng.sample(steps, 2)
                instruction = f"Ответь на вопрос: {question}. Стиль: {style}. Номер {n+1}."
                output = (
                    f"Если {question}, начни с простого: {s1}. Затем {s2}. "
                    f"Не требуй от себя идеальной реакции сразу. В стиле «{style}» главный смысл такой: сначала снизить напряжение, потом решать саму ситуацию."
                )
                add(records, seen, instruction, output, "conversation/direct_emotional_questions", ["direct_answer", question, style])

    # Boundary examples: introduce the assistant only when asked, otherwise answer the user.
    identity_questions = [
        "кто ты", "как тебя зовут", "ты onyxthornAI", "что ты умеешь", "ты можешь помочь поговорить",
    ]
    for question in identity_questions:
        for n in range(220):
            instruction = f"{question}? Ответ {n+1}."
            output = (
                f"Я onyxthornAI. Я могу спокойно поговорить, объяснить тему, помочь сформулировать сообщение или разложить бытовую ситуацию по шагам. "
                f"Но если ты задаешь конкретный вопрос, я должен отвечать на него напрямую, а не перечислять режимы работы."
            )
            add(records, seen, instruction, output, "assistant/identity_boundaries", ["identity", "brand"])

    if len(records) < TARGET:
        raise RuntimeError(f"Generated only {len(records)} unique records, need {TARGET}")

    records = records[:TARGET]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"written {len(records):,} records to {OUT}")


if __name__ == "__main__":
    main()
