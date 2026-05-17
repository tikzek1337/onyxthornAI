from __future__ import annotations

import argparse

from .generate import generate_reply, load_model
from .prompting import DEFAULT_SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive onyxthornAI conversation chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_fallback", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.tokenizer or None, args.device)
    print("onyxthornAI chat. Commands: /exit, /settings, /info")
    while True:
        user = input("\nYou> ").strip()
        if user.lower() in {"/exit", "exit", "quit", "q"}:
            break
        if user.lower() == "/settings":
            print(
                f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
                f"max_new_tokens={args.max_new_tokens}, repetition_penalty={args.repetition_penalty}"
            )
            continue
        if user.lower() == "/info":
            print(
                f"model=onyxthornAI | device={device} | params={model.parameter_count():,} | "
                f"context={model.config.block_size} tokens | vocab={model.config.vocab_size}"
            )
            continue
        answer = generate_reply(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_message=user,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_fallback=not args.no_fallback,
        )
        print(f"\nonyxthornAI> {answer}")


if __name__ == "__main__":
    main()
