from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class OnyxTokenizer:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install the 'tokenizers' package first: pip install tokenizers") from exc
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise KeyError(f"Token not found in tokenizer: {token}")
        return int(value)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.token_to_id("<|bos|>")] + ids
        if add_eos:
            ids = ids + [self.token_to_id("<|eos|>")]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)


def train_bpe_tokenizer(
    input_files: list[str | Path],
    out_path: str | Path,
    vocab_size: int = 16000,
    min_frequency: int = 2,
) -> None:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer
        from tokenizers.processors import TemplateProcessing
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install the 'tokenizers' package first: pip install tokenizers") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train([str(p) for p in input_files], trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<|bos|>", tokenizer.token_to_id("<|bos|>")), ("<|eos|>", tokenizer.token_to_id("<|eos|>"))],
    )
    tokenizer.save(str(out_path))
