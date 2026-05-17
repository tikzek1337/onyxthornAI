from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import OnyxLiteAI
from .prompting import DEFAULT_SYSTEM_PROMPT, build_chat_prompt
from .tokenizer import OnyxTokenizer


def load_model(checkpoint_path: str, tokenizer_path: str | None = None, device: str = "auto"):
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = OnyxLiteAI(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer_path = tokenizer_path or checkpoint.get("extra", {}).get("tokenizer_path") or "tokenizer/onyxliteai_tokenizer.json"
    tokenizer = OnyxTokenizer(tokenizer_path)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with onyxLiteAI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.tokenizer or None, args.device)
    prompt = build_chat_prompt(args.prompt, args.system)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<|eos|>")
    out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eos_token_id=eos_id)
    text = tokenizer.decode(out[0].tolist())
    marker = "<|assistant|>\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.replace("<|end|>", "").replace("<|eos|>", "").strip()
    print(text)


if __name__ == "__main__":
    main()
