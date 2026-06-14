from __future__ import annotations

import argparse
from datasets import load_dataset
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tinixai/vietnam-real-estates")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i + 1 >= args.max_rows:
            break

    df = pd.DataFrame(rows)

    print("\n===== DATASET SAMPLE =====")
    print(df.head())

    print("\n===== COLUMNS =====")
    for col in df.columns:
        print(col, "|", df[col].dtype)

    if "published_at" in df.columns:
        print("\nPublished time sample:")
        print(df["published_at"].head())

    if "price" in df.columns:
        print("\nPrice sample:")
        print(df["price"].head())


if __name__ == "__main__":
    main()
