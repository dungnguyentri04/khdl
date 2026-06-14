"""
Full pipeline: chuẩn hóa dữ liệu Chợ Tốt về schema Hugging Face, gộp dữ liệu,
và huấn luyện/đánh giá mô hình dự đoán giá căn hộ chung cư theo thời gian.

Chạy ví dụ:

1) Chỉ chuẩn hóa và gộp dữ liệu:
python src/merge_chotot_hf_full_pipeline.py ^
  --chotot-csv data/raw_chotot_with_date.csv ^
  --output-dir data_merged ^
  --mode prepare_only

2) Train trên Hugging Face, test ngoài nguồn trên Chợ Tốt:
python src/merge_chotot_hf_full_pipeline.py ^
  --chotot-csv data/raw_chotot_with_date.csv ^
  --output-dir outputs_external_chotot ^
  --mode external_test ^
  --province "Hồ Chí Minh" ^
  --property-type "Căn hộ chung cư" ^
  --hf-max-rows 300000 ^
  --hf-scan-limit 4000000

3) Gộp Hugging Face + Chợ Tốt rồi chia theo tháng mới nhất:
python src/merge_chotot_hf_full_pipeline.py ^
  --chotot-csv data/raw_chotot_with_date.csv ^
  --output-dir outputs_merged ^
  --mode merged_train ^
  --province "Hồ Chí Minh" ^
  --property-type "Căn hộ chung cư" ^
  --hf-max-rows 300000 ^
  --hf-scan-limit 4000000
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


# ============================================================
# 1. Schema chung dùng cho cả Hugging Face và Chợ Tốt
# ============================================================

COMMON_COLUMNS = [
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
    "source",
]

HF_COLUMNS = [
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


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "project_name",
    "house_direction",
    "balcony_direction",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 2. Hàm chuẩn hóa text, địa danh, thời gian
# ============================================================

def clean_text(x, default: str = "Unknown") -> str:
    if pd.isna(x):
        return default
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x if x else default


def normalize_province(x) -> str:
    x = clean_text(x)
    x = x.replace("TP.", "TP")
    mapping = {
        "Tp Hồ Chí Minh": "Hồ Chí Minh",
        "TP Hồ Chí Minh": "Hồ Chí Minh",
        "TP. Hồ Chí Minh": "Hồ Chí Minh",
        "Thành phố Hồ Chí Minh": "Hồ Chí Minh",
        "TP.HCM": "Hồ Chí Minh",
        "HCM": "Hồ Chí Minh",
    }
    return mapping.get(x, x)


def normalize_district(x) -> str:
    x = clean_text(x)

    replacements = {
        "Q.": "Quận ",
        "Q ": "Quận ",
        "Tp.": "Thành phố ",
        "TP.": "Thành phố ",
    }
    for old, new in replacements.items():
        x = x.replace(old, new)

    prefixes = [
        "Quận ",
        "Huyện ",
        "Thành phố ",
        "Thị xã ",
    ]

    for prefix in prefixes:
        if x.startswith(prefix):
            x = x.replace(prefix, "", 1).strip()

    return x if x else "Unknown"


def normalize_ward(x) -> str:
    x = clean_text(x)

    prefixes = [
        "Phường ",
        "Xã ",
        "Thị trấn ",
    ]

    for prefix in prefixes:
        if x.startswith(prefix):
            x = x.replace(prefix, "", 1).strip()

    return x if x else "Unknown"


def normalize_direction(x) -> str:
    x = clean_text(x)

    # Chợ Tốt đôi khi trả mã số cho hướng. Không có bảng giải mã thì để Unknown.
    if re.fullmatch(r"\d+(\.0)?", x):
        return "Unknown"

    mapping = {
        "Đông": "Đông",
        "Tây": "Tây",
        "Nam": "Nam",
        "Bắc": "Bắc",
        "Đông Nam": "Đông Nam",
        "Đông Bắc": "Đông Bắc",
        "Tây Nam": "Tây Nam",
        "Tây Bắc": "Tây Bắc",
    }

    return mapping.get(x, "Unknown" if x.lower() in ["nan", "none", ""] else x)


def parse_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def parse_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# 3. Chuẩn hóa Hugging Face dataset
# ============================================================

def row_passes_filter(row: dict, province: str, property_type: str) -> bool:
    if province:
        if normalize_province(row.get("province_name")) != province:
            return False

    if property_type:
        if clean_text(row.get("property_type_name")) != property_type:
            return False

    return True


def load_huggingface_dataset(args) -> pd.DataFrame:
    print("\n===== LOADING HUGGING FACE DATASET =====")
    print(f"Dataset      : {args.hf_dataset}")
    print(f"Split        : {args.hf_split}")
    print(f"Province     : {args.province or 'ALL'}")
    print(f"Property type: {args.property_type or 'ALL'}")
    print(f"Max rows     : {args.hf_max_rows:,}")
    print(f"Scan limit   : {args.hf_scan_limit:,}")

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)

    rows = []
    scanned = 0
    kept = 0

    for row in ds:
        scanned += 1

        if row_passes_filter(row, args.province, args.property_type):
            item = {col: row.get(col, None) for col in HF_COLUMNS}
            item["source"] = "huggingface"
            rows.append(item)
            kept += 1

        if scanned % args.print_every == 0:
            print(f"[HF] scanned={scanned:,}, kept={kept:,}")

        if kept >= args.hf_max_rows:
            break

        if scanned >= args.hf_scan_limit:
            break

    if not rows:
        raise ValueError("Không lấy được dòng nào từ Hugging Face. Kiểm tra lại province/property-type.")

    df = pd.DataFrame(rows)
    df = standardize_common_schema(df, source_name="huggingface")

    print(f"[HF DONE] rows={len(df):,}")
    return df


# ============================================================
# 4. Chuẩn hóa dữ liệu Chợ Tốt sang schema chung
# ============================================================

def convert_chotot_to_common_schema(chotot_csv: str | Path) -> pd.DataFrame:
    chotot_csv = Path(chotot_csv)

    if not chotot_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file Chợ Tốt: {chotot_csv}")

    print("\n===== LOADING CHOTOT CSV =====")
    print(f"File: {chotot_csv}")

    raw = pd.read_csv(chotot_csv)

    out = pd.DataFrame()

    # Text fields
    out["name"] = raw["Subject"] if "Subject" in raw.columns else "Unknown"

    # Không nên dùng description/name làm feature chính vì có thể chứa giá.
    # Tuy nhiên vẫn lưu lại để truy vết và báo cáo.
    out["description"] = raw["Subject"] if "Subject" in raw.columns else "Unknown"

    # Property fields
    out["property_type_name"] = "Căn hộ chung cư"

    out["province_name"] = raw["Region"] if "Region" in raw.columns else "Hồ Chí Minh"
    out["district_name"] = raw["District"] if "District" in raw.columns else "Unknown"
    out["ward_name"] = raw["Ward"] if "Ward" in raw.columns else "Unknown"
    out["street_name"] = raw["Address"] if "Address" in raw.columns else "Unknown"

    # Dữ liệu crawl hiện chưa có project_name chuẩn. Để Unknown.
    # Có thể viết thêm rule extract tên dự án từ Subject sau.
    out["project_name"] = "Unknown"

    # Numeric fields
    out["price"] = raw["PriceVND"] if "PriceVND" in raw.columns else np.nan
    out["area"] = raw["AreaM2"] if "AreaM2" in raw.columns else np.nan
    out["floor_count"] = raw["Floor"] if "Floor" in raw.columns else np.nan
    out["frontage_width"] = np.nan
    out["house_depth"] = np.nan
    out["road_width"] = np.nan
    out["bedroom_count"] = raw["BedRooms"] if "BedRooms" in raw.columns else np.nan
    out["bathroom_count"] = raw["BathRooms"] if "BathRooms" in raw.columns else np.nan

    # Direction fields
    out["house_direction"] = raw["MainDirection"] if "MainDirection" in raw.columns else "Unknown"
    out["balcony_direction"] = raw["BalconyDirection"] if "BalconyDirection" in raw.columns else "Unknown"

    # Time field
    if "NgayDang" in raw.columns:
        out["published_at"] = raw["NgayDang"]
    elif "CrawlDate" in raw.columns:
        out["published_at"] = raw["CrawlDate"]
    else:
        out["published_at"] = pd.NaT

    out["source"] = "chotot"

    # Metadata useful for debugging, not for model training
    if "AdId" in raw.columns:
        out["source_ad_id"] = raw["AdId"]
    if "ListId" in raw.columns:
        out["source_list_id"] = raw["ListId"]
    if "Link" in raw.columns:
        out["source_link"] = raw["Link"]

    out = standardize_common_schema(out, source_name="chotot")

    # Drop duplicated Chợ Tốt listings
    dedup_cols = ["name", "price", "area", "district_name", "ward_name"]
    before = len(out)
    out = out.drop_duplicates(subset=[c for c in dedup_cols if c in out.columns]).reset_index(drop=True)
    after = len(out)

    print(f"[CHOTOT DONE] rows={after:,} | removed_duplicates={before - after:,}")
    return out


# ============================================================
# 5. Chuẩn hóa schema chung và làm sạch dữ liệu
# ============================================================

def standardize_common_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    # Ensure columns exist
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Text normalization
    df["name"] = df["name"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)
    df["property_type_name"] = df["property_type_name"].apply(clean_text)
    df["province_name"] = df["province_name"].apply(normalize_province)
    df["district_name"] = df["district_name"].apply(normalize_district)
    df["ward_name"] = df["ward_name"].apply(normalize_ward)
    df["street_name"] = df["street_name"].apply(clean_text)
    df["project_name"] = df["project_name"].apply(clean_text)
    df["house_direction"] = df["house_direction"].apply(normalize_direction)
    df["balcony_direction"] = df["balcony_direction"].apply(normalize_direction)

    # Numeric normalization
    for col in [
        "price",
        "area",
        "floor_count",
        "frontage_width",
        "house_depth",
        "road_width",
        "bedroom_count",
        "bathroom_count",
    ]:
        df[col] = parse_numeric(df[col])

    # Time normalization
    df["published_at"] = parse_datetime(df["published_at"])

    # Source
    df["source"] = source_name

    return df


def clean_for_model(
    df: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_area: float,
    max_area: float,
    min_price_per_m2: float,
    max_price_per_m2: float,
) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df.dropna(subset=["price", "area", "published_at"])

    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
    df = df[(df["area"] >= min_area) & (df["area"] <= max_area)]

    df["price_per_m2"] = df["price"] / df["area"]
    df = df[
        (df["price_per_m2"] >= min_price_per_m2)
        & (df["price_per_m2"] <= max_price_per_m2)
    ]

    # Time features
    df["posted_year"] = df["published_at"].dt.year
    df["posted_month"] = df["published_at"].dt.month
    df["posted_day"] = df["published_at"].dt.day
    df["posted_dayofweek"] = df["published_at"].dt.dayofweek

    df = df.sort_values("published_at").reset_index(drop=True)

    print("\n===== CLEAN FOR MODEL =====")
    print(f"Before: {before:,}")
    print(f"After : {len(df):,}")
    print(f"Time  : {df['published_at'].min()} -> {df['published_at'].max()}")

    print("\nRows by source:")
    print(df["source"].value_counts().to_string())

    print("\nRows by month:")
    print(df["published_at"].dt.to_period("M").value_counts().sort_index().to_string())

    return df


# ============================================================
# 6. Split theo thời gian
# ============================================================

def split_latest_month(df: pd.DataFrame):
    df = df.sort_values("published_at").reset_index(drop=True)
    months = df["published_at"].dt.to_period("M")
    latest_month = months.max()

    train_df = df[months < latest_month].copy()
    test_df = df[months == latest_month].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Không đủ dữ liệu để chia latest_month. Cần ít nhất 2 tháng dữ liệu.")

    return train_df, test_df, f"latest_month_{latest_month}"


def split_chronological(df: pd.DataFrame, test_size: float):
    df = df.sort_values("published_at").reset_index(drop=True)
    cut = int(len(df) * (1 - test_size))

    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Không đủ dữ liệu để chia chronological.")

    return train_df, test_df, f"chronological_last_{test_size:.0%}"


def make_external_split(hf_df: pd.DataFrame, chotot_df: pd.DataFrame):
    train_df = hf_df.copy()
    test_df = chotot_df.copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("External test cần train Hugging Face và test Chợ Tốt đều có dữ liệu.")

    return train_df, test_df, "external_test_hf_train_chotot_test"


# ============================================================
# 7. Model training
# ============================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor():
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
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


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


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, args, split_strategy: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== TIME SPLIT =====")
    print(f"Strategy: {split_strategy}")
    print(f"Train: {train_df['published_at'].min()} -> {train_df['published_at'].max()} | n={len(train_df):,}")
    print(f"Test : {test_df['published_at'].min()} -> {test_df['published_at'].max()} | n={len(test_df):,}")

    print("\nTrain source distribution:")
    print(train_df["source"].value_counts().to_string())

    print("\nTest source distribution:")
    print(test_df["source"].value_counts().to_string())

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["price"].astype(float)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["price"].astype(float)

    # Train target on log scale
    y_train_log = np.log1p(y_train)

    models = {
        "Random_Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
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
                ("preprocess", build_preprocessor()),
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

    # Baseline median
    baseline_pred = np.full(len(y_test), float(np.median(y_train)))
    baseline_metrics = evaluate(y_test, baseline_pred)
    rows.append({"Model": "Baseline_Median", **baseline_metrics})

    metrics_df = pd.DataFrame(rows).sort_values("MAE_VND").reset_index(drop=True)

    print("\n===== METRICS =====")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    best_model_name = metrics_df.iloc[0]["Model"]
    if best_model_name == "Baseline_Median":
        best_model_name = "Random_Forest"

    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, output_dir / "best_model.joblib")

    # Save prediction file
    best_pred = np.expm1(best_model.predict(X_test))
    best_pred = np.maximum(best_pred, 0)

    pred_df = test_df[
        [
            "published_at",
            "source",
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
        "mode": args.mode,
        "split_strategy": split_strategy,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df["published_at"].min()),
        "train_end": str(train_df["published_at"].max()),
        "test_start": str(test_df["published_at"].min()),
        "test_end": str(test_df["published_at"].max()),
        "train_source_distribution": train_df["source"].value_counts().to_dict(),
        "test_source_distribution": test_df["source"].value_counts().to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns_note": [
            "price is target and not used as feature.",
            "price_per_m2 is computed from price and is used only for outlier filtering, not as feature.",
            "name and description are not used by default because listing text may contain price and cause leakage.",
            "source is kept for analysis, not used as feature to avoid platform-bias learning.",
        ],
        "target_transform": "model trains on log1p(price), metrics are reported in VND after expm1.",
        "best_model": best_model_name,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] Outputs saved to: {output_dir}")


# ============================================================
# 8. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Normalize Chợ Tốt, merge with Hugging Face, and train models.")

    parser.add_argument(
        "--mode",
        choices=["prepare_only", "hf_only", "external_test", "merged_train"],
        default="prepare_only",
        help=(
            "prepare_only: chỉ chuẩn hóa và lưu CSV. "
            "hf_only: train/test chỉ Hugging Face theo latest_month. "
            "external_test: train Hugging Face, test Chợ Tốt. "
            "merged_train: gộp HF + Chợ Tốt rồi split latest_month."
        ),
    )

    parser.add_argument("--chotot-csv", default="data/raw_chotot_with_date.csv")
    parser.add_argument("--output-dir", default="outputs_merge")

    # Hugging Face
    parser.add_argument("--hf-dataset", default="tinixai/vietnam-real-estates")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--province", default="Hồ Chí Minh")
    parser.add_argument("--property-type", default="Căn hộ chung cư")
    parser.add_argument("--hf-max-rows", type=int, default=300000)
    parser.add_argument("--hf-scan-limit", type=int, default=4000000)
    parser.add_argument("--print-every", type=int, default=50000)

    # Cleaning thresholds
    parser.add_argument("--min-price", type=float, default=100_000_000)
    parser.add_argument("--max-price", type=float, default=100_000_000_000)
    parser.add_argument("--min-area", type=float, default=10)
    parser.add_argument("--max-area", type=float, default=500)
    parser.add_argument("--min-price-per-m2", type=float, default=5_000_000)
    parser.add_argument("--max-price-per-m2", type=float, default=1_000_000_000)

    # Split
    parser.add_argument("--split", choices=["latest_month", "chronological"], default="latest_month")
    parser.add_argument("--test-size", type=float, default=0.2)

    # Model params
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def save_dataset_outputs(output_dir: Path, hf_df: pd.DataFrame | None, chotot_df: pd.DataFrame | None, merged_df: pd.DataFrame | None):
    output_dir.mkdir(parents=True, exist_ok=True)

    if hf_df is not None:
        hf_df.to_csv(output_dir / "hf_clean_schema.csv", index=False, encoding="utf-8-sig")

    if chotot_df is not None:
        chotot_df.to_csv(output_dir / "chotot_clean_schema.csv", index=False, encoding="utf-8-sig")

    if merged_df is not None:
        merged_df.to_csv(output_dir / "merged_hf_chotot.csv", index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chợ Tốt is always loaded because the user's current purpose is conversion/merge.
    chotot_df = convert_chotot_to_common_schema(args.chotot_csv)
    chotot_df = clean_for_model(
        chotot_df,
        args.min_price,
        args.max_price,
        args.min_area,
        args.max_area,
        args.min_price_per_m2,
        args.max_price_per_m2,
    )

    hf_df = None
    merged_df = None

    if args.mode in ["hf_only", "external_test", "merged_train", "prepare_only"]:
        # For prepare_only, still load HF because the point is to create merged_hf_chotot.csv.
        hf_df = load_huggingface_dataset(args)
        hf_df = clean_for_model(
            hf_df,
            args.min_price,
            args.max_price,
            args.min_area,
            args.max_area,
            args.min_price_per_m2,
            args.max_price_per_m2,
        )

    # Keep only common/model-safe columns + metadata
    keep_cols = list(dict.fromkeys(COMMON_COLUMNS + ["price_per_m2", "posted_year", "posted_month", "posted_day", "posted_dayofweek"]))
    hf_keep = hf_df[keep_cols].copy() if hf_df is not None else None
    chotot_keep = chotot_df[keep_cols].copy()

    if hf_keep is not None:
        merged_df = pd.concat([hf_keep, chotot_keep], ignore_index=True)
        merged_df = merged_df.sort_values("published_at").reset_index(drop=True)

        # Remove exact duplicate after merge
        merged_df = merged_df.drop_duplicates(
            subset=["name", "price", "area", "district_name", "ward_name", "published_at", "source"],
            keep="first",
        ).reset_index(drop=True)

    save_dataset_outputs(output_dir, hf_keep, chotot_keep, merged_df)

    print("\n===== SAVED CLEAN DATASETS =====")
    if hf_keep is not None:
        print(output_dir / "hf_clean_schema.csv")
    print(output_dir / "chotot_clean_schema.csv")
    if merged_df is not None:
        print(output_dir / "merged_hf_chotot.csv")

    if args.mode == "prepare_only":
        print("\n[DONE] prepare_only: chỉ chuẩn hóa và lưu file, chưa train model.")
        return

    if args.mode == "hf_only":
        train_df, test_df, strategy = split_latest_month(hf_keep) if args.split == "latest_month" else split_chronological(hf_keep, args.test_size)

    elif args.mode == "external_test":
        train_df, test_df, strategy = make_external_split(hf_keep, chotot_keep)

    elif args.mode == "merged_train":
        train_df, test_df, strategy = split_latest_month(merged_df) if args.split == "latest_month" else split_chronological(merged_df, args.test_size)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    train_and_evaluate(train_df, test_df, output_dir, args, strategy)


if __name__ == "__main__":
    main()
