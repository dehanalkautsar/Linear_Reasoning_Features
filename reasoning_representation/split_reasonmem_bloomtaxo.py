import argparse
import json
import os
import random
from typing import List, Tuple


def split_records(records: List[dict], validation_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    validation_size = int(round(len(records) * validation_ratio))
    validation_size = max(1, min(validation_size, len(records) - 1))

    validation_indices = set(indices[:validation_size])
    validation_split = [records[i] for i in range(len(records)) if i in validation_indices]
    test_split = [records[i] for i in range(len(records)) if i not in validation_indices]
    return validation_split, test_split


def split_file(input_path: str, validation_ratio: float = 0.1, seed: int = 8888) -> None:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {input_path}")

    validation_split, test_split = split_records(records, validation_ratio, seed)

    base, ext = os.path.splitext(input_path)
    validation_path = f"{base}_validation{ext}"
    test_path = f"{base}_test{ext}"

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_split, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)

    print(f"Saved validation split: {validation_path} ({len(validation_split)} items)")
    print(f"Saved test split:        {test_path} ({len(test_split)} items)")


def main():
    parser = argparse.ArgumentParser(
        description="Split ReasonMem or BloomTaxo JSON datasets into 10% validation and 90% test splits."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input JSON files to split, e.g. reason_mem_labels_mmlu_ind.json bloom_tax_labels_indo.json",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples to place in validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8888,
        help="Random seed for reproducible sampling (default: 8888)",
    )
    args = parser.parse_args()

    for path in args.paths:
        split_file(path, validation_ratio=args.validation_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
# python split_reasonmem_bloomtaxo.py reason_mem_labels_mmlu_ind.json bloom_tax_labels_indo.json
# python split_reasonmem_bloomtaxo.py reason_mem_labels_mmlu_ind.json --validation-ratio 0.1 --seed 8888