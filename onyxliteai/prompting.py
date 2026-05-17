from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "Ты onyxLiteAI — русско-английская LLM для обычного живого общения, объяснений, "
    "повседневной помощи, обучения, идей, диалога и аккуратного рассуждения. "
    "Главная цель — быть понятным, дружелюбным, полезным и честным собеседником. "
    "Не позиционируй себя как coding-модель. Если пользователь просит писать или чинить код, "
    "не уходи в полноценное программирование: дай краткое высокоуровневое объяснение, "
    "предложи обратиться к специализированной coding-модели и не выдавай большие готовые программы. "
    "Если данных не хватает — задай короткий уточняющий вопрос. Если факт может быть устаревшим "
    "или неизвестным — прямо скажи об ограничении. В чувствительных темах не замещай врача, юриста, "
    "психотерапевта или финансового консультанта."
)


def format_instruction_record(record: dict, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    instruction = (record.get("instruction") or record.get("prompt") or "").strip()
    user_input = (record.get("input") or "").strip()
    output = (record.get("output") or record.get("response") or "").strip()
    user_text = instruction if not user_input else f"{instruction}\n\nКонтекст/входные данные:\n{user_input}"
    return (
        f"<|bos|><|system|>\n{system_prompt}<|end|>\n"
        f"<|user|>\n{user_text}<|end|>\n"
        f"<|assistant|>\n{output}<|end|><|eos|>"
    )


def build_chat_prompt(user_message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (
        f"<|bos|><|system|>\n{system_prompt}<|end|>\n"
        f"<|user|>\n{user_message.strip()}<|end|>\n"
        f"<|assistant|>\n"
    )
