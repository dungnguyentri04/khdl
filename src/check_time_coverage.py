"""Check time coverage of crawled Chợ Tốt CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw_chotot_with_date.csv")
    p.add_argument("--date-col", default="NgayDang")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if args.date_col not in df.columns:
        raise ValueError(f"Không có cột {args.date_col}. Columns={list(df.columns)}")
    dt = pd.to_datetime(df[args.date_col], errors="coerce")
    print("Rows:", len(df))
    print("Valid dates:", dt.notna().sum())
    print("Min date:", dt.min())
    print("Max date:", dt.max())
    print("\nRows by month:")
    print(dt.dt.to_period("M").value_counts().sort_index())
    print("\nRows by day:")
    print(dt.dt.to_period("D").value_counts().sort_index())


if __name__ == "__main__":
    main()
