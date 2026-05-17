"""Small helper template for adding more onyxthornAI chat records.

This script intentionally creates only a tiny sample. Extend the topic lists and
write outputs manually/with review if you want higher-quality data.
"""
from __future__ import annotations

import json
from pathlib import Path

records = [
    {
        "instruction": "Привет, поболтай со мной",
        "input": "",
        "output": "Привет. Можем спокойно поговорить: про день, планы, настроение, странную мысль или любую тему, которую хочется разобрать.",
        "category": "conversation/custom",
        "language": "ru",
        "tags": ["smalltalk"],
    }
]

out = Path("data/raw/custom_chat_examples.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"written: {out}")
