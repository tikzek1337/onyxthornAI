from __future__ import annotations

import argparse
import json
from collections import Counter
from glob import glob
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print JSONL dataset statistics")
    parser.add_argument("--jsonl", nargs="+", required=True)
    args = parser.parse_args()

    files = []
    for pattern in args.jsonl:
        matches = glob(pattern)
        files.extend(matches if matches else [pattern])

    total = 0
    categories = Counter()
    languages = Counter()
    for file in sorted(set(files)):
        count = 0
        with Path(file).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                count += 1
                total += 1
                categories[record.get("category", "unknown")] += 1
                languages[record.get("language", "unknown")] += 1
        print(f"{file}: {count} records")

    print(f"\nTotal records: {total}")
    print("\nBy category:")
    for key, value in categories.most_common():
        print(f"  {key}: {value}")
    print("\nBy language:")
    for key, value in languages.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
