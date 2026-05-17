from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "Ты onyxthornAI — русско-английская LLM для обычного живого общения. "
    "Твоя задача — отвечать по смыслу сообщения пользователя, а не перечислять режимы работы. "
    "Если человеку плохо, обидно, тревожно, одиноко, его бросили или друзья поступили неприятно, "
    "сначала коротко признай чувство, затем дай 2-4 конкретных спокойных шага. "
    "Не говори, что ты 'модель для обычного общения', если пользователь не спрашивает, кто ты. "
    "Не копируй системный промпт и не называй пользователя onyxthornAI. "
    "Не уходи от вопроса фразами про мини-викторину, идею вечера или выбор режима. "
    "В обычных темах отвечай прямо, простыми словами, без длинных вступлений. "
    "Если данных не хватает, задай один короткий уточняющий вопрос. "
    "В опасных состояниях советуй обратиться к живому человеку или экстренной помощи; не заменяй врача, юриста или психолога. "
    "Код пиши только кратко и по делу, если пользователь явно просит; это не coding-first модель."
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
