from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

EXPECTED_COLUMNS = [
    "name",
    "description",
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "price",
    "area",
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
    "house_direction",
    "balcony_direction",
    "published_at",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vietnam real-estate price model from Hugging Face dataset")

    parser.add_argument("--dataset", default="tinixai/vietnam-real-estates")
    parser.add_argument("--split-name", default="train")

    parser.add_argument("--province", default="", help='Example: "Hồ Chí Minh". Empty = all provinces.')
    parser.add_argument("--property-type", default="", help='Example: "Căn hộ chung cư". Empty = all types.')

    parser.add_argument("--max-rows", type=int, default=200000, help="Maximum kept rows after filtering.")
    parser.add_argument("--scan-limit", type=int, default=1000000, help="Maximum raw rows to scan from streaming dataset.")
    parser.add_argument("--print-every", type=int, default=50000)

    parser.add_argument("--min-price", type=float, default=100_000_000)
    parser.add_argument("--max-price", type=float, default=100_000_000_000)
    parser.add_argument("--min-area", type=float, default=10)
    parser.add_argument("--max-area", type=float, default=500)
    parser.add_argument("--min-price-per-m2", type=float, default=5_000_000)
    parser.add_argument("--max-price-per-m2", type=float, default=1_000_000_000)

    parser.add_argument("--split", choices=["latest_month", "latest_day", "chronological", "auto"], default="latest_month")
    parser.add_argument("--test-size", type=float, default=0.2, help="Used only for chronological split.")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-test-rows", type=int, default=200)

    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--output-dir", default="outputs_hf")
    parser.add_argument("--save-clean-data", action="store_true")

    return parser.parse_args()


def norm_text(x):
    if x is None:
        return ""
    return str(x).strip()


def row_passes_filter(row, province, property_type):
    if province and norm_text(row.get("province_name")) != province:
        return False
    if property_type and norm_text(row.get("property_type_name")) != property_type:
        return False
    return True


def load_hf_dataframe(args):
    print("\n===== LOADING DATASET FROM HUGGING FACE =====")
    print(f"Dataset      : {args.dataset}")
    print(f"Split        : {args.split_name}")
    print(f"Province     : {args.province or 'ALL'}")
    print(f"Property type: {args.property_type or 'ALL'}")
    print(f"Max kept rows: {args.max_rows}")
    print(f"Scan limit   : {args.scan_limit}")

    ds = load_dataset(args.dataset, split=args.split_name, streaming=True)

    rows = []
    scanned = 0
    kept = 0

    for row in ds:
        scanned += 1

        if row_passes_filter(row, args.province, args.property_type):
            rows.append({col: row.get(col, None) for col in EXPECTED_COLUMNS})
            kept += 1

        if scanned % args.print_every == 0:
            print(f"[LOAD] scanned={scanned:,}, kept={kept:,}")

        if kept >= args.max_rows:
            break
        if scanned >= args.scan_limit:
            break

    if not rows:
        raise ValueError(
            "Không lấy được dòng nào. Hãy kiểm tra lại --province hoặc --property-type. "
            "Có thể chạy inspect_dataset.py để xem giá trị thực tế."
        )

    df = pd.DataFrame(rows)
    print(f"[DONE] scanned={scanned:,}, kept={len(df):,}")
    return df


def clean_dataframe(df, args):
    df = df.copy()

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    numeric_cols = [
        "floor_count",
        "frontage_width",
        "house_depth",
        "road_width",
        "bedroom_count",
        "bathroom_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    df = df.dropna(subset=["published_at", "price", "area"])
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]
    df = df[(df["area"] >= args.min_area) & (df["area"] <= args.max_area)]

    df["price_per_m2"] = df["price"] / df["area"]
    df = df[
        (df["price_per_m2"] >= args.min_price_per_m2)
        & (df["price_per_m2"] <= args.max_price_per_m2)
    ]

    df["posted_year"] = df["published_at"].dt.year
    df["posted_month"] = df["published_at"].dt.month
    df["posted_day"] = df["published_at"].dt.day
    df["posted_dayofweek"] = df["published_at"].dt.dayofweek

    for col in [
        "property_type_name",
        "province_name",
        "district_name",
        "ward_name",
        "street_name",
        "project_name",
        "house_direction",
        "balcony_direction",
    ]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].eq("") | df[col].eq("nan"), col] = "Unknown"

    df = df.sort_values("published_at").reset_index(drop=True)

    print("\n===== CLEANING =====")
    print(f"Before cleaning: {before:,}")
    print(f"After cleaning : {len(df):,}")
    print(f"Time range     : {df['published_at'].min()} -> {df['published_at'].max()}")

    print("\nRows by month:")
    print(df["published_at"].dt.to_period("M").value_counts().sort_index().to_string())

    return df


def split_data(df, args):
    df = df.sort_values("published_at").reset_index(drop=True)

    def latest_month_split(data):
        months = data["published_at"].dt.to_period("M")
        latest = months.max()
        train = data[months < latest].copy()
        test = data[months == latest].copy()
        return train, test, f"latest_month_{latest}"

    def latest_day_split(data):
        days = data["published_at"].dt.date
        latest = days.max()
        train = data[days < latest].copy()
        test = data[days == latest].copy()
        return train, test, f"latest_day_{latest}"

    def chronological_split(data):
        cut = int(len(data) * (1 - args.test_size))
        train = data.iloc[:cut].copy()
        test = data.iloc[cut:].copy()
        return train, test, f"chronological_last_{args.test_size:.0%}"

    if args.split == "latest_month":
        train, test, strategy = latest_month_split(df)
    elif args.split == "latest_day":
        train, test, strategy = latest_day_split(df)
    elif args.split == "chronological":
        train, test, strategy = chronological_split(df)
    else:
        train, test, strategy = latest_month_split(df)
        if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
            train, test, strategy = latest_day_split(df)
        if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
            train, test, strategy = chronological_split(df)

    if len(train) < 1 or len(test) < 1:
        raise ValueError("Không đủ dữ liệu để chia train/test theo thời gian.")

    print("\n===== TIME-BASED SPLIT =====")
    print(f"Strategy: {strategy}")
    print(f"Train: {train['published_at'].min()} -> {train['published_at'].max()} | n={len(train):,}")
    print(f"Test : {test['published_at'].min()} -> {test['published_at'].max()} | n={len(test):,}")

    if len(train) < args.min_train_rows:
        print(f"[WARN] Train rows thấp hơn khuyến nghị: {len(train):,} < {args.min_train_rows:,}")
    if len(test) < args.min_test_rows:
        print(f"[WARN] Test rows thấp hơn khuyến nghị: {len(test):,} < {args.min_test_rows:,}")

    return train, test, strategy


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor():
    numeric_features = [
        "area",
        "floor_count",
        "frontage_width",
        "house_depth",
        "road_width",
        "bedroom_count",
        "bathroom_count",
        "posted_year",
        "posted_month",
        "posted_day",
        "posted_dayofweek",
    ]

    categorical_features = [
        "property_type_name",
        "province_name",
        "district_name",
        "ward_name",
        "street_name",
        "project_name",
        "house_direction",
        "balcony_direction",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE_VND": mae,
        "RMSE_VND": rmse,
        "R2": r2,
        "MAE_BILLION_VND": mae / 1_000_000_000,
        "RMSE_BILLION_VND": rmse / 1_000_000_000,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_hf_dataframe(args)
    df = clean_dataframe(df_raw, args)

    if args.save_clean_data:
        clean_path = output_dir / "clean_data.csv"
        df.to_csv(clean_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Clean data: {clean_path}")

    train_df, test_df, strategy = split_data(df, args)

    preprocessor, numeric_features, categorical_features = build_preprocessor()
    feature_cols = numeric_features + categorical_features

    X_train = train_df[feature_cols]
    y_train = train_df["price"].astype(float)
    X_test = test_df[feature_cols]
    y_test = test_df["price"].astype(float)

    y_train_log = np.log1p(y_train)

    models = {
        "Random_Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        min_samples_leaf=args.min_samples_leaf,
                        random_state=args.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Ridge_Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    rows = []
    fitted_models = {}

    print("\n===== TRAINING =====")
    for name, model in models.items():
        print(f"[TRAIN] {name}")
        model.fit(X_train, y_train_log)
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        pred = np.maximum(pred, 0)

        metrics = evaluate(y_test, pred)
        rows.append({"Model": name, **metrics})
        fitted_models[name] = model

    baseline_pred = np.full(shape=len(y_test), fill_value=float(np.median(y_train)))
    baseline_metrics = evaluate(y_test, baseline_pred)
    rows.append({"Model": "Baseline_Median", **baseline_metrics})

    metrics_df = pd.DataFrame(rows).sort_values("MAE_VND")
    print("\n===== METRICS =====")
    print(metrics_df.to_string(index=False))

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    best_model_name = metrics_df.iloc[0]["Model"]
    if best_model_name == "Baseline_Median":
        best_model_name = "Random_Forest"

    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, output_dir / "best_model.joblib")

    best_pred = np.expm1(best_model.predict(X_test))
    pred_df = test_df[
        [
            "published_at",
            "property_type_name",
            "province_name",
            "district_name",
            "ward_name",
            "street_name",
            "project_name",
            "area",
            "bedroom_count",
            "bathroom_count",
            "price",
        ]
    ].copy()
    pred_df["predicted_price"] = best_pred
    pred_df["absolute_error"] = np.abs(pred_df["price"] - pred_df["predicted_price"])
    pred_df["absolute_error_billion"] = pred_df["absolute_error"] / 1_000_000_000
    pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": args.dataset,
        "province_filter": args.province,
        "property_type_filter": args.property_type,
        "split_strategy": strategy,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df["published_at"].min()),
        "train_end": str(train_df["published_at"].max()),
        "test_start": str(test_df["published_at"].min()),
        "test_end": str(test_df["published_at"].max()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "log1p(price), reported back in VND",
        "best_model": best_model_name,
        "notes": [
            "name and description are excluded by default to reduce leakage risk because listing text may contain price.",
            "target price is log-transformed during training.",
            "metrics are computed after converting predictions back to VND.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
