from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .assistant_response import extract_assistant_text, polish_answer
from .config import ModelConfig
from .model import OnyxThornAI
from .prompting import DEFAULT_SYSTEM_PROMPT, build_chat_prompt
from .tokenizer import OnyxTokenizer


def load_model(checkpoint_path: str, tokenizer_path: str | None = None, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = OnyxThornAI(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer_path = tokenizer_path or checkpoint.get("extra", {}).get("tokenizer_path") or "tokenizer/onyxthornai_tokenizer.json"
    if not Path(tokenizer_path).exists() and Path("tokenizer/onyxliteai_tokenizer.json").exists():
        tokenizer_path = "tokenizer/onyxliteai_tokenizer.json"
    tokenizer = OnyxTokenizer(tokenizer_path)
    return model, tokenizer, device


@torch.no_grad()
def generate_reply(
    model: OnyxThornAI,
    tokenizer: OnyxTokenizer,
    device: str,
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float | None = 0.92,
    repetition_penalty: float = 1.08,
    use_fallback: bool = True,
) -> str:
    prompt = build_chat_prompt(user_message, system_prompt)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<|eos|>")
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    )
    decoded = tokenizer.decode(out[0].tolist())
    answer = extract_assistant_text(decoded)
    return polish_answer(answer, user_message) if use_fallback else answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with onyxthornAI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_fallback", action="store_true", help="Disable simple emotional/off-topic fallback repair")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.tokenizer or None, args.device)
    text = generate_reply(
        model=model,
        tokenizer=tokenizer,
        device=device,
        user_message=args.prompt,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_fallback=not args.no_fallback,
    )
    print(text)


if __name__ == "__main__":
    main()
