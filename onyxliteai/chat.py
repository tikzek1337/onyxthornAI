from __future__ import annotations

import argparse
import torch

from .generate import load_model
from .prompting import DEFAULT_SYSTEM_PROMPT, build_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive onyxLiteAI conversation chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.tokenizer or None, args.device)
    eos_id = tokenizer.token_to_id("<|eos|>")
    print("onyxLiteAI chat. Type /exit to quit.")
    while True:
        user = input("\nYou> ").strip()
        if user.lower() in {"/exit", "exit", "quit", "q"}:
            break
        prompt = build_chat_prompt(user, DEFAULT_SYSTEM_PROMPT)
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eos_token_id=eos_id)
        text = tokenizer.decode(out[0].tolist())
        marker = "<|assistant|>\n"
        answer = text.split(marker, 1)[1] if marker in text else text
        answer = answer.replace("<|end|>", "").replace("<|eos|>", "").strip()
        print(f"\nonyxLiteAI> {answer}")


if __name__ == "__main__":
    main()
