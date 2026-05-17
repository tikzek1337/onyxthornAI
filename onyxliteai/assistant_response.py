from __future__ import annotations

import re

BRAND_NAME = "onyxthornAI"

_BAD_FRAGMENTS = (
    "модель для обычного общения",
    "не буду уходить",
    "coding-модель",
    "coding assistant",
    "как не убить вечер",
    "мини-викторина",
    "идея для вечера",
    "Можем идти легко",
    "Ты onyxthornAI",
    "Я onyxthornAI, модель",
)

_DISTRESS_PATTERNS = (
    r"\bмне\s+плохо\b",
    r"\bменя\s+бросил[аио]?\b",
    r"\bбросили\b",
    r"\bрасстал[аи]сь\b",
    r"\bразрыв\b",
    r"\bодинок[ао]?\b",
    r"\bдрузья\b.*\bбез\s+меня\b",
    r"\bушли\s+гулять\b",
    r"\bне\s+позвали\b",
    r"\bплачу\b",
    r"\bтревожн[оа]\b",
    r"\bпаник[ауы]\b",
    r"\bхочу\s+умереть\b",
    r"\bсуицид\b",
)

_CRISIS_PATTERNS = (
    r"\bхочу\s+умереть\b",
    r"\bпокончить\s+с\s+собой\b",
    r"\bсуицид\b",
    r"\bне\s+хочу\s+жить\b",
)


def extract_assistant_text(decoded_text: str) -> str:
    """Return only the last assistant turn from a decoded prompt+completion."""
    marker = "<|assistant|>\n"
    pos = decoded_text.rfind(marker)
    text = decoded_text[pos + len(marker):] if pos != -1 else decoded_text
    for token in ("<|end|>", "<|eos|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>"):
        text = text.replace(token, "")
    # Stop if the model starts hallucinating a new dialog turn.
    text = re.split(r"\n\s*(?:You|User|Пользователь|Human)>", text, maxsplit=1)[0]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def is_distress_message(user_message: str) -> bool:
    return _matches_any(user_message, _DISTRESS_PATTERNS)


def is_crisis_message(user_message: str) -> bool:
    return _matches_any(user_message, _CRISIS_PATTERNS)


def answer_looks_bad(answer: str, user_message: str = "") -> bool:
    stripped = answer.strip()
    if len(stripped) < 18:
        return True
    lowered = stripped.lower()
    if any(fragment.lower() in lowered for fragment in _BAD_FRAGMENTS):
        return True
    if "<|" in stripped or stripped.count("Я onyxthornAI") > 1:
        return True
    if is_distress_message(user_message) and not any(word in lowered for word in ("понима", "тяж", "обид", "больно", "поддерж", "сейчас", "можно")):
        return True
    return False


def fallback_reply(user_message: str) -> str:
    lowered = user_message.lower()
    if is_crisis_message(lowered):
        return (
            "Сейчас важнее всего безопасность. Отойди от всего, чем можно себе навредить, и напиши или позвони живому человеку: другу, родственнику, соседу. "
            "Если есть риск, что ты можешь сделать с собой что-то прямо сейчас, звони в экстренную службу своего региона. "
            "Пока помощь рядом не появилась, сделай одну короткую вещь: сядь, поставь ноги на пол, медленно вдохни и выдохни 5 раз, затем напиши кому-то: «Мне сейчас небезопасно одному, побудь со мной»."
        )
    if "брос" in lowered or "расстал" in lowered or "разрыв" in lowered:
        return (
            "Это правда может сильно выбить. Первое — не пытайся сейчас срочно стать «нормальным» и не пиши человеку десять сообщений подряд на эмоциях. "
            "Сделай ближайшие 30 минут максимально простыми: вода, душ или еда, потом убери телефон на паузу. "
            "Дальше полезно написать в заметки три вещи: что произошло, что ты чувствуешь, что нельзя делать сегодня, чтобы не стало хуже. "
            "Если есть близкий человек, напиши ему прямо: «Меня бросили, мне плохо, можешь немного побыть на связи?»"
        )
    if "друз" in lowered and ("без" in lowered or "не позв" in lowered or "ушли" in lowered):
        return (
            "Это обидно, даже если они не хотели специально ранить. Сначала лучше не додумывать за них самое плохое. "
            "Можно написать спокойно: «Я увидел, что вы пошли без меня. Мне стало неприятно. В следующий раз позовите меня тоже, если можно». "
            "А на сегодня выбери действие, которое вернет контроль: позови кого-то отдельно, выйди прогуляться сам, включи фильм или займись делом на час. "
            "Главное — не превращать один эпизод в вывод, что ты никому не нужен."
        )
    if "мне плохо" in lowered or "плохо" in lowered:
        return (
            "Понимаю. Давай без давления: сейчас не нужно решать всю жизнь целиком. "
            "Сделай три маленьких шага: выпей воды, сядь удобнее и напиши одним предложением, что именно болит сильнее всего — тревога, одиночество, обида, усталость или страх. "
            "После этого можно выбрать один следующий шаг. Если состояние резкое, опасное или ты боишься оставаться один, лучше сразу связаться с близким человеком или экстренной помощью."
        )
    return (
        "Понял. Скажу просто: давай разберем это по шагам. "
        "Сначала опиши ситуацию одним-двумя предложениями, потом отдельно — что ты хочешь получить в итоге: совет, поддержку, план действий или просто разговор."
    )


def polish_answer(answer: str, user_message: str = "") -> str:
    answer = extract_assistant_text(answer)
    if answer_looks_bad(answer, user_message):
        return fallback_reply(user_message)
    # Do not let the model over-introduce itself on every turn.
    answer = re.sub(r"^Я\s+onyxthornAI[,.]?\s*", "", answer).strip()
    return answer
