from __future__ import annotations

import re
from typing import Iterable

# This is not a replacement for professional dataset governance.
# It is a small first-pass filter for obvious junk and dangerous noise.
DEFAULT_BLOCKLIST = [
    r"(?i)password\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)secret[_-]?key\s*=\s*['\"][^'\"]+['\"]",
    r"(?i)BEGIN RSA PRIVATE KEY",
]


def looks_unsafe(text: str, patterns: Iterable[str] = DEFAULT_BLOCKLIST) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()
