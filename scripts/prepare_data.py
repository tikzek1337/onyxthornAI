from __future__ import annotations

import argparse
import sys
from pathlib import Path
from glob import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onyxliteai.data import build_corpus_file, encode_jsonl_to_bins
from onyxliteai.tokenizer import train_bpe_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tokenizer and build packed token bins")
    parser.add_argument("--jsonl", nargs="+", required=True, help="One or more instruction JSONL files")
    parser.add_argument("--tokenizer", default="tokenizer/onyxliteai_tokenizer.json")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--corpus", default="data/processed/tokenizer_corpus.txt")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    jsonl_files = []
    for pattern in args.jsonl:
        matches = [Path(x) for x in glob(pattern)]
        jsonl_files.extend(matches if matches else [Path(pattern)])
    jsonl_files = sorted(set(jsonl_files))
    if not jsonl_files:
        raise SystemExit("No JSONL files found.")
    print("JSONL files:")
    for file in jsonl_files:
        print(f"  - {file}")
    corpus_path = build_corpus_file(jsonl_files, args.corpus)
    print(f"corpus written: {corpus_path}")
    train_bpe_tokenizer([corpus_path], args.tokenizer, args.vocab_size, args.min_frequency)
    print(f"tokenizer written: {args.tokenizer}")
    encode_jsonl_to_bins(jsonl_files, args.tokenizer, args.out_dir, args.val_fraction, args.seed)
    print(f"packed train/val bins written: {args.out_dir}")


if __name__ == "__main__":
    main()
