from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. SCHEMA CHUNG
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

MOGI_COLUMNS = [
    "name",
    "description",
    "property_type_name",
    "province_name",
    "district_name",
    "ward_name",
    "street_name",
    "price",
    "area",
    "bedroom_count",
    "bathroom_count",
    "published_at",
    "url",
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
# 2. CHUẨN HÓA TEXT / ĐỊA DANH / GIÁ / DIỆN TÍCH
# ============================================================

def clean_text(x, default: str = "Unknown") -> str:
    if pd.isna(x):
        return default
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x if x else default


def normalize_province(x) -> str:
    x = clean_text(x)

    mapping = {
        "Tp Hồ Chí Minh": "Hồ Chí Minh",
        "TP Hồ Chí Minh": "Hồ Chí Minh",
        "TP. Hồ Chí Minh": "Hồ Chí Minh",
        "Thành phố Hồ Chí Minh": "Hồ Chí Minh",
        "TPHCM": "Hồ Chí Minh",
        "TP.HCM": "Hồ Chí Minh",
        "HCM": "Hồ Chí Minh",
        "Hồ Chí Minh cũ": "Hồ Chí Minh",
    }

    return mapping.get(x, x)


def normalize_district(x) -> str:
    x = clean_text(x)

    # Bỏ phần ngoặc như "Quận 2 (TP. Thủ Đức)" -> "Quận 2"
    x = re.sub(r"\s*\(.*?\)\s*", "", x).strip()

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

    x = re.sub(r"\s*\(.*?\)\s*", "", x).strip()

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

    if re.fullmatch(r"\d+(\.0)?", x):
        return "Unknown"

    valid = {
        "Unknown",
        "Đông",
        "Tây",
        "Nam",
        "Bắc",
        "Đông Nam",
        "Đông Bắc",
        "Tây Nam",
        "Tây Bắc",
    }

    return x if x in valid else "Unknown"


def parse_number_vn(text: str) -> float:
    """
    Parse số kiểu Việt Nam trong giá/diện tích.
    Ví dụ:
    - "3,780" -> 3.780
    - "3.780" -> 3.780 nếu chỉ có 3 chữ số sau dấu, dùng trong "3.780 tỷ"
    - "11,5" -> 11.5
    """
    text = str(text).strip()

    # Giữ lại số, dấu phẩy, dấu chấm
    text = re.sub(r"[^0-9,.]", "", text)

    if not text:
        return np.nan

    # Nếu có cả "." và "," thì giả định "." là ngăn cách nghìn, "," là thập phân
    if "." in text and "," in text:
        text = text.replace(".", "").replace(",", ".")
        return float(text)

    # Nếu chỉ có "," thì đổi thành "."
    if "," in text:
        return float(text.replace(",", "."))

    # Nếu chỉ có "." thì giữ nguyên.
    # Với dữ liệu giá dạng 3.780tỷ, đây là 3.780 tỷ.
    return float(text)


def parse_price_vnd(value):
    """
    Chuyển giá text từ Mogi/Chợ Tốt sang VNĐ.

    Hỗ trợ:
    - "6 tỷ 600 triệu" -> 6_600_000_000
    - "3 tỷ 780 triệu" -> 3_780_000_000
    - "2 tỷ 40 triệu" -> 2_040_000_000
    - "50 tỷ" -> 50_000_000_000
    - "3.780tỷ" -> 3_780_000_000
    - "1,650 tỷ" -> 1_650_000_000
    - "4.6 tỷ" -> 4_600_000_000
    - số VNĐ sẵn -> giữ nguyên
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if value > 0 else np.nan

    s = str(value).lower().strip()
    s = s.replace("tỉ", "tỷ").replace("ty", "tỷ")
    s = s.replace("triệu", "triệu").replace("trieu", "triệu")
    s = s.replace("vnd", "").replace("vnđ", "").replace("đ", "")
    s = re.sub(r"\s+", " ", s)

    total = 0.0

    # Lấy phần tỷ đầu tiên
    billion_match = re.search(r"(\d+(?:[,.]\d+)?)\s*tỷ", s)
    if billion_match:
        billion_value = parse_number_vn(billion_match.group(1))
        if not pd.isna(billion_value):
            total += billion_value * 1_000_000_000

        # Lấy phần triệu sau chữ tỷ nếu có
        after = s[billion_match.end():]
        million_match = re.search(r"(\d+(?:[,.]\d+)?)\s*triệu", after)
        if million_match:
            million_value = parse_number_vn(million_match.group(1))
            if not pd.isna(million_value):
                total += million_value * 1_000_000

        return total if total > 0 else np.nan

    # Nếu chỉ có triệu
    million_match = re.search(r"(\d+(?:[,.]\d+)?)\s*triệu", s)
    if million_match:
        million_value = parse_number_vn(million_match.group(1))
        return million_value * 1_000_000 if not pd.isna(million_value) else np.nan

    # Nếu là số thuần
    num_match = re.search(r"\d+(?:[,.]\d+)?", s)
    if num_match:
        num = parse_number_vn(num_match.group(0))
        if pd.isna(num):
            return np.nan

        # Nếu số đã là VNĐ
        if num >= 100_000_000:
            return num

        # Nếu số nhỏ, giả định là tỷ
        if num < 1000:
            return num * 1_000_000_000

    return np.nan


def parse_area_m2(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if value > 0 else np.nan

    s = str(value).lower()
    s = s.replace("m²", "").replace("m2", "").replace("㎡", "")
    s = s.strip()

    match = re.search(r"\d+(?:[,.]\d+)?", s)
    if not match:
        return np.nan

    return parse_number_vn(match.group(0))


def parse_datetime_series(series, source_name: str):
    if source_name == "mogi":
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

    return pd.to_datetime(series, errors="coerce", yearfirst=True)


# ============================================================
# 3. CHUẨN HÓA SCHEMA CHUNG
# ============================================================

def standardize_common_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

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

    df["price"] = df["price"].apply(parse_price_vnd)
    df["area"] = df["area"].apply(parse_area_m2)

    for col in [
        "floor_count",
        "frontage_width",
        "house_depth",
        "road_width",
        "bedroom_count",
        "bathroom_count",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["published_at"] = parse_datetime_series(df["published_at"], source_name)
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

    df["posted_year"] = df["published_at"].dt.year
    df["posted_month"] = df["published_at"].dt.month
    df["posted_day"] = df["published_at"].dt.day
    df["posted_dayofweek"] = df["published_at"].dt.dayofweek

    df = df.sort_values("published_at").reset_index(drop=True)

    print("\n===== CLEAN FOR MODEL =====")
    print(f"Before: {before:,}")
    print(f"After : {len(df):,}")
    if len(df) > 0:
        print(f"Time  : {df['published_at'].min()} -> {df['published_at'].max()}")
        print("\nRows by source:")
        print(df["source"].value_counts().to_string())
        print("\nRows by month:")
        print(df["published_at"].dt.to_period("M").value_counts().sort_index().to_string())

    return df


# ============================================================
# 4. LOAD HUGGING FACE
# ============================================================

def load_hf_from_cache(cache_path: str | Path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Không tìm thấy HF cache: {cache_path}")

    print("\n===== LOADING HF CACHE =====")
    print(cache_path)

    df = pd.read_csv(cache_path)
    df = standardize_common_schema(df, source_name="huggingface")
    return df


def load_hf_from_huggingface(args) -> pd.DataFrame:
    from datasets import load_dataset

    print("\n===== LOADING HUGGING FACE DATASET =====")
    print(f"Dataset      : {args.hf_dataset}")
    print(f"Province     : {args.province}")
    print(f"Property type: {args.property_type}")
    print(f"Max rows     : {args.hf_max_rows:,}")
    print(f"Scan limit   : {args.hf_scan_limit:,}")

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)

    rows = []
    scanned = 0
    kept = 0

    for row in ds:
        scanned += 1

        province = normalize_province(row.get("province_name"))
        prop_type = clean_text(row.get("property_type_name"))

        if province == args.province and prop_type == args.property_type:
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
        raise ValueError("Không lấy được dòng HF nào. Kiểm tra lại tham số.")

    df = pd.DataFrame(rows)
    df = standardize_common_schema(df, source_name="huggingface")
    return df


# ============================================================
# 5. LOAD CHỢ TỐT
# ============================================================

def load_chotot_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[SKIP] Không tìm thấy Chợ Tốt CSV: {path}")
        return pd.DataFrame(columns=COMMON_COLUMNS)

    print("\n===== LOADING CHOTOT CSV =====")
    print(path)

    raw = pd.read_csv(path)

    out = pd.DataFrame()
    out["name"] = raw["Subject"] if "Subject" in raw.columns else "Unknown"
    out["description"] = raw["Subject"] if "Subject" in raw.columns else "Unknown"
    out["property_type_name"] = "Căn hộ chung cư"
    out["province_name"] = raw["Region"] if "Region" in raw.columns else "Hồ Chí Minh"
    out["district_name"] = raw["District"] if "District" in raw.columns else "Unknown"
    out["ward_name"] = raw["Ward"] if "Ward" in raw.columns else "Unknown"
    out["street_name"] = raw["Address"] if "Address" in raw.columns else "Unknown"
    out["project_name"] = "Unknown"

    out["price"] = raw["PriceVND"] if "PriceVND" in raw.columns else np.nan
    out["area"] = raw["AreaM2"] if "AreaM2" in raw.columns else np.nan
    out["floor_count"] = raw["Floor"] if "Floor" in raw.columns else np.nan
    out["frontage_width"] = np.nan
    out["house_depth"] = np.nan
    out["road_width"] = np.nan
    out["bedroom_count"] = raw["BedRooms"] if "BedRooms" in raw.columns else np.nan
    out["bathroom_count"] = raw["BathRooms"] if "BathRooms" in raw.columns else np.nan

    out["house_direction"] = raw["MainDirection"] if "MainDirection" in raw.columns else "Unknown"
    out["balcony_direction"] = raw["BalconyDirection"] if "BalconyDirection" in raw.columns else "Unknown"

    if "NgayDang" in raw.columns:
        out["published_at"] = raw["NgayDang"]
    elif "CrawlDate" in raw.columns:
        out["published_at"] = raw["CrawlDate"]
    else:
        out["published_at"] = pd.NaT

    out["source"] = "chotot"

    if "Link" in raw.columns:
        out["source_link"] = raw["Link"]

    out = standardize_common_schema(out, source_name="chotot")

    before = len(out)
    out = out.drop_duplicates(
        subset=["name", "price", "area", "district_name", "ward_name"],
        keep="first",
    ).reset_index(drop=True)

    print(f"[CHOTOT] rows={len(out):,}, duplicates_removed={before - len(out):,}")

    return out


# ============================================================
# 6. LOAD MOGI
# ============================================================

def load_mogi_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[SKIP] Không tìm thấy Mogi CSV: {path}")
        return pd.DataFrame(columns=COMMON_COLUMNS)

    print("\n===== LOADING MOGI CSV =====")
    print(path)

    rows = []
    skipped = 0

    # File hiện tại có thể không có header và có một số dòng rác đầu file.
    # Vì vậy dùng csv.reader và chỉ giữ dòng có đúng 13 cột.
    with open(path, "r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 13:
                # Nếu là header thì bỏ qua
                if str(row[0]).strip().lower() in ["name", "title", "tên tin"]:
                    continue
                rows.append(row)
            else:
                skipped += 1

    if not rows:
        raise ValueError(
            "Không đọc được dòng Mogi hợp lệ. File cần có 13 cột: "
            "name, description, property_type_name, province_name, district_name, "
            "ward_name, street_name, price, area, bedroom_count, bathroom_count, published_at, url"
        )

    raw = pd.DataFrame(rows, columns=MOGI_COLUMNS)

    out = pd.DataFrame()
    out["name"] = raw["name"]
    out["description"] = raw["description"]
    out["property_type_name"] = raw["property_type_name"]
    out["province_name"] = raw["province_name"]
    out["district_name"] = raw["district_name"]
    out["ward_name"] = raw["ward_name"]
    out["street_name"] = raw["street_name"]
    out["project_name"] = "Unknown"

    out["price"] = raw["price"]
    out["area"] = raw["area"]
    out["floor_count"] = np.nan
    out["frontage_width"] = np.nan
    out["house_depth"] = np.nan
    out["road_width"] = np.nan
    out["bedroom_count"] = raw["bedroom_count"]
    out["bathroom_count"] = raw["bathroom_count"]

    out["house_direction"] = "Unknown"
    out["balcony_direction"] = "Unknown"
    out["published_at"] = raw["published_at"]
    out["source"] = "mogi"
    out["source_link"] = raw["url"]

    out = standardize_common_schema(out, source_name="mogi")

    before = len(out)
    out = out.drop_duplicates(
        subset=["name", "price", "area", "district_name", "ward_name", "published_at"],
        keep="first",
    ).reset_index(drop=True)

    print(f"[MOGI] valid_rows={len(out):,}, skipped_rows={skipped:,}, duplicates_removed={before - len(out):,}")

    return out


# ============================================================
# 7. SPLIT TRAIN 06/2025-03/2026, TEST 04/2026-05/2026
# ============================================================

def split_train_test_by_date(df: pd.DataFrame, args):
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    train_df = df[
        (df["published_at"] >= train_start)
        & (df["published_at"] <= train_end)
    ].copy()

    test_df = df[
        (df["published_at"] >= test_start)
        & (df["published_at"] <= test_end)
    ].copy()

    if len(train_df) == 0:
        raise ValueError("Train rỗng. Kiểm tra mốc train_start/train_end.")
    if len(test_df) == 0:
        raise ValueError("Test rỗng. Kiểm tra mốc test_start/test_end hoặc dữ liệu Mogi/Chợ Tốt.")

    strategy = f"fixed_train_{args.train_start}_to_{args.train_end}__test_{args.test_start}_to_{args.test_end}"

    return train_df, test_df, strategy


# ============================================================
# 8. MODEL
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


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, args, strategy: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== FIXED TIME SPLIT =====")
    print(f"Strategy: {strategy}")
    print(f"Train: {train_df['published_at'].min()} -> {train_df['published_at'].max()} | n={len(train_df):,}")
    print(f"Test : {test_df['published_at'].min()} -> {test_df['published_at'].max()} | n={len(test_df):,}")

    print("\nTrain source distribution:")
    print(train_df["source"].value_counts().to_string())

    print("\nTest source distribution:")
    print(test_df["source"].value_counts().to_string())

    print("\nTrain rows by month:")
    print(train_df["published_at"].dt.to_period("M").value_counts().sort_index().to_string())

    print("\nTest rows by month:")
    print(test_df["published_at"].dt.to_period("M").value_counts().sort_index().to_string())

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["price"].astype(float)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["price"].astype(float)

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

    baseline_pred = np.full(len(y_test), float(np.median(y_train)))
    rows.append({"Model": "Baseline_Median", **evaluate(y_test, baseline_pred)})

    metrics_df = pd.DataFrame(rows).sort_values("MAE_VND").reset_index(drop=True)

    print("\n===== METRICS =====")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    best_model_name = metrics_df.iloc[0]["Model"]
    if best_model_name == "Baseline_Median":
        best_model_name = "Random_Forest"

    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, output_dir / "best_model.joblib")

    best_pred = np.expm1(best_model.predict(X_test))
    best_pred = np.maximum(best_pred, 0)

    pred_cols = [
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
    pred_df = test_df[pred_cols].copy()
    pred_df["predicted_price"] = best_pred
    pred_df["absolute_error"] = np.abs(pred_df["price"] - pred_df["predicted_price"])
    pred_df["absolute_error_billion"] = pred_df["absolute_error"] / 1_000_000_000

    pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "split_strategy": strategy,
        "train_start": str(train_df["published_at"].min()),
        "train_end": str(train_df["published_at"].max()),
        "test_start": str(test_df["published_at"].min()),
        "test_end": str(test_df["published_at"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_source_distribution": train_df["source"].value_counts().to_dict(),
        "test_source_distribution": test_df["source"].value_counts().to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_transform": "log1p(price), metrics reported after expm1 to VND",
        "best_model": best_model_name,
        "excluded_columns_note": [
            "price is target, not feature.",
            "price_per_m2 is computed from price and used only for outlier filtering, not feature.",
            "name and description are excluded because listing text may contain price and cause leakage.",
            "source is kept for analysis but not used as model feature.",
        ],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] Outputs saved to: {output_dir}")


# ============================================================
# 9. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HuggingFace + Chợ Tốt + Mogi and train with fixed split: train 06/2025-03/2026, test 04/2026-05/2026."
    )

    parser.add_argument("--output-dir", default="outputs_mogi_fixed_split")

    # Data sources
    parser.add_argument("--mogi-csv", default="data/mogi_dataset.csv")
    parser.add_argument("--chotot-csv", default="data/raw_chotot_with_date.csv")
    parser.add_argument("--hf-cache", default="outputs_merged/hf_clean_schema.csv")
    parser.add_argument("--use-hf-cache", action="store_true", help="Dùng file HF đã clean sẵn để khỏi tải lại Hugging Face.")

    # Hugging Face loading if cache is not used
    parser.add_argument("--hf-dataset", default="tinixai/vietnam-real-estates")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--province", default="Hồ Chí Minh")
    parser.add_argument("--property-type", default="Căn hộ chung cư")
    parser.add_argument("--hf-max-rows", type=int, default=300000)
    parser.add_argument("--hf-scan-limit", type=int, default=4000000)
    parser.add_argument("--print-every", type=int, default=50000)

    # Fixed split
    parser.add_argument("--train-start", default="2025-06-01")
    parser.add_argument("--train-end", default="2026-03-31")
    parser.add_argument("--test-start", default="2026-04-01")
    parser.add_argument("--test-end", default="2026-05-31")

    # Cleaning thresholds
    parser.add_argument("--min-price", type=float, default=100_000_000)
    parser.add_argument("--max-price", type=float, default=100_000_000_000)
    parser.add_argument("--min-area", type=float, default=10)
    parser.add_argument("--max-area", type=float, default=500)
    parser.add_argument("--min-price-per-m2", type=float, default=5_000_000)
    parser.add_argument("--max-price-per-m2", type=float, default=1_000_000_000)

    # Model params
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load Hugging Face
    if args.use_hf_cache:
        hf_df = load_hf_from_cache(args.hf_cache)
    else:
        hf_df = load_hf_from_huggingface(args)

    # 2) Load Chợ Tốt
    chotot_df = load_chotot_csv(args.chotot_csv)

    # 3) Load Mogi
    mogi_df = load_mogi_csv(args.mogi_csv)

    # 4) Gộp nguồn
    raw_merged = pd.concat(
        [hf_df, chotot_df, mogi_df],
        ignore_index=True,
        sort=False,
    )

    # 5) Clean chung
    merged = clean_for_model(
        raw_merged,
        min_price=args.min_price,
        max_price=args.max_price,
        min_area=args.min_area,
        max_area=args.max_area,
        min_price_per_m2=args.min_price_per_m2,
        max_price_per_m2=args.max_price_per_m2,
    )

    # 6) Xóa trùng sau gộp
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=["name", "price", "area", "district_name", "ward_name", "published_at", "source"],
        keep="first",
    ).reset_index(drop=True)
    print(f"\nDuplicates removed after merge: {before - len(merged):,}")

    # 7) Lưu dữ liệu sạch
    merged.to_csv(output_dir / "merged_all_clean.csv", index=False, encoding="utf-8-sig")
    hf_df.to_csv(output_dir / "hf_schema_raw.csv", index=False, encoding="utf-8-sig")
    chotot_df.to_csv(output_dir / "chotot_schema_raw.csv", index=False, encoding="utf-8-sig")
    mogi_df.to_csv(output_dir / "mogi_schema_raw.csv", index=False, encoding="utf-8-sig")

    # 8) Split cố định train 06/2025-03/2026, test 04/2026-05/2026
    train_df, test_df, strategy = split_train_test_by_date(merged, args)

    train_df.to_csv(output_dir / "train_data.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test_data.csv", index=False, encoding="utf-8-sig")

    # 9) Train và evaluate
    train_and_evaluate(train_df, test_df, output_dir, args, strategy)


if __name__ == "__main__":
    main()
