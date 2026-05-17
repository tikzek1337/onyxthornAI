from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch

from .prompting import format_instruction_record
from .safety import clean_text, looks_unsafe
from .tokenizer import OnyxTokenizer


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def _record_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_corpus_file(jsonl_files: list[str | Path], out_path: str | Path, max_chars_per_record: int = 20000) -> Path:
    """Build tokenizer corpus and drop exact duplicate formatted examples."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as out:
        for file in jsonl_files:
            for record in read_jsonl(file):
                text = format_instruction_record(record)
                text = clean_text(text)
                if not text or looks_unsafe(text):
                    continue
                key = _record_key(text)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                kept += 1
                out.write(text[:max_chars_per_record] + "\n\n")
    print(f"corpus records kept={kept:,}, duplicates skipped={skipped:,}")
    return out_path


def encode_jsonl_to_bins(
    jsonl_files: list[str | Path],
    tokenizer_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.05,
    seed: int = 1337,
) -> None:
    tokenizer = OnyxTokenizer(tokenizer_path)
    records: list[list[int]] = []
    seen: set[str] = set()
    skipped = 0
    for file in jsonl_files:
        for record in read_jsonl(file):
            text = clean_text(format_instruction_record(record))
            if not text or looks_unsafe(text):
                continue
            key = _record_key(text)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            ids = tokenizer.encode(text)
            if len(ids) > 8:
                records.append(ids)

    if len(records) < 2:
        raise ValueError("Need at least two valid training records to create train/val split.")

    rng = random.Random(seed)
    rng.shuffle(records)
    val_count = max(1, int(len(records) * val_fraction))
    val_records = records[:val_count]
    train_records = records[val_count:]

    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_bin(records_: list[list[int]], filename: str) -> int:
        flat = [token for rec in records_ for token in rec]
        arr = np.array(flat, dtype=dtype)
        arr.tofile(out_dir / filename)
        return len(arr)

    train_tokens = write_bin(train_records, "train.bin")
    val_tokens = write_bin(val_records, "val.bin")
    meta = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "duplicates_skipped": skipped,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"encoded records train={len(train_records):,}, val={len(val_records):,}, "
        f"tokens train={train_tokens:,}, val={val_tokens:,}, duplicates skipped={skipped:,}"
    )


class PackedTokenDataset:
    def __init__(self, bin_path: str | Path, block_size: int, device: str, dtype: str = "uint16"):
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.data = np.memmap(bin_path, dtype=np_dtype, mode="r")
        self.block_size = block_size
        self.device = device
        if len(self.data) <= block_size + 1:
            raise ValueError(f"Dataset {bin_path} is too small for block_size={block_size}")

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.block_size - 1
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)
