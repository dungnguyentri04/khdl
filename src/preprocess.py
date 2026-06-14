"""Preprocessing utilities for Chợ Tốt apartment price prediction.

The original project used many text columns directly and random train/test split.
This version converts raw Vietnamese real-estate fields into stable numerical and
categorical features, then lets the training script split data by time.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else np.nan


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def to_float_number(value) -> Optional[float]:
    """Parse Vietnamese-style number strings.

    Examples:
    - "62 m²" -> 62
    - "32,26 triệu/m²" -> 32.26
    - "1.234,5" -> 1234.5
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).lower().strip()
    text = text.replace("m²", "").replace("m2", "")
    text = text.replace("triệu", "").replace("trieu", "")
    text = text.replace("tỷ", "").replace("ty", "")
    text = text.replace("/", " ")
    match = re.search(r"[-+]?\d[\d\.,]*", text)
    if not match:
        return np.nan
    num = match.group(0)
    # Vietnamese decimal comma. If both . and , exist, assume . is thousand and , is decimal.
    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num:
        num = num.replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return np.nan


def parse_price_vnd(value) -> Optional[float]:
    """Parse Chợ Tốt price text to VND.

    Examples:
    - "2 tỷ- 62 m2đ" -> 2_000_000_000
    - "2,58 tỷ" -> 2_580_000_000
    - "850 triệu" -> 850_000_000
    - numeric API price is treated as VND.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        val = float(value)
        return val if val > 0 else np.nan

    text = str(value).lower().strip()
    text = text.split("-")[0]
    text = text.replace("đ", "").replace("vnd", "").strip()

    number = to_float_number(text)
    if pd.isna(number):
        return np.nan

    no_accent = strip_accents(text)
    if "ty" in no_accent:
        return number * 1_000_000_000
    if "trieu" in no_accent:
        return number * 1_000_000

    # If number is small, it is probably billion in shorthand.
    if number < 1000:
        return number * 1_000_000_000
    return number


def extract_district(address) -> Optional[str]:
    if pd.isna(address):
        return np.nan
    text = normalize_text(address)
    if pd.isna(text):
        return np.nan
    patterns = [
        r"(Quận\s+[\wÀ-ỹ\s]+)",
        r"(Q\.\s*\d+)",
        r"(Huyện\s+[\wÀ-ỹ\s]+)",
        r"(Thành phố\s+Thủ Đức)",
        r"(TP\.\s*Thủ Đức)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return normalize_text(m.group(1))
    return np.nan


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_chotot_dataframe(raw_df: pd.DataFrame, date_col: str = "NgayDang") -> pd.DataFrame:
    """Clean raw Chợ Tốt data from either old notebook CSV or new API crawler."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize date.
    if date_col not in df.columns:
        fallback = first_existing_column(df, ["CrawlDate", "crawl_date", "created_at", "date"])
        if fallback is None:
            raise ValueError(
                f"Không tìm thấy cột ngày '{date_col}'. Hãy crawl lại bằng src/crawl_chotot.py để có NgayDang/CrawlDate."
            )
        date_col = fallback
    df["NgayDang"] = pd.to_datetime(df[date_col], errors="coerce")

    # Target price.
    price_col = first_existing_column(df, ["PriceVND", "price", "Gia", "price_vnd"])
    if price_col is None:
        raise ValueError("Không tìm thấy cột giá. Cần có một trong các cột: PriceVND, price, Gia, price_vnd.")
    df["PriceVND"] = df[price_col].apply(parse_price_vnd)

    # Area.
    area_col = first_existing_column(df, ["AreaM2", "DienTich", "area", "size", "area_m2"])
    if area_col is not None:
        df["AreaM2"] = df[area_col].apply(to_float_number)
    else:
        df["AreaM2"] = np.nan

    # Rooms.
    bed_col = first_existing_column(df, ["BedRooms", "Phongngu", "bedrooms", "rooms"])
    bath_col = first_existing_column(df, ["BathRooms", "PhongTam", "bathrooms", "toilets"])
    floor_col = first_existing_column(df, ["Floor", "SoTang", "floor"])

    df["BedRooms"] = df[bed_col].apply(to_float_number) if bed_col else np.nan
    df["BathRooms"] = df[bath_col].apply(to_float_number) if bath_col else np.nan
    df["Floor"] = df[floor_col].apply(to_float_number) if floor_col else np.nan

    # Location and categorical fields.
    address_col = first_existing_column(df, ["Address", "DiaChi", "address"])
    district_col = first_existing_column(df, ["District", "district", "area_name"])
    ward_col = first_existing_column(df, ["Ward", "ward", "ward_name"])

    if district_col:
        df["District"] = df[district_col].apply(normalize_text)
    elif address_col:
        df["District"] = df[address_col].apply(extract_district)
    else:
        df["District"] = np.nan

    df["Ward"] = df[ward_col].apply(normalize_text) if ward_col else np.nan

    rename_candidates = {
        "PropertyStatus": ["PropertyStatus", "TinhTrangBDS", "property_status", "condition_ad"],
        "Type": ["Type", "Loai", "type", "category_name"],
        "LegalStatus": ["LegalStatus", "GiayTo", "legal_status"],
        "Interior": ["Interior", "TinhTrangNoiThat", "interior"],
        "MainDirection": ["MainDirection", "HuongCuaChinh", "direction"],
        "BalconyDirection": ["BalconyDirection", "HuongBanCong", "balcony_direction"],
        "Characteristic": ["Characteristic", "DacDiem", "characteristic"],
        "SellerType": ["SellerType", "seller_type", "account_type"],
    }

    for new_col, candidates in rename_candidates.items():
        old_col = first_existing_column(df, candidates)
        df[new_col] = df[old_col].apply(normalize_text) if old_col else np.nan

    # Time features known at prediction time.
    df["PostedYear"] = df["NgayDang"].dt.year
    df["PostedMonth"] = df["NgayDang"].dt.month

    # Basic filters.
    df = df.dropna(subset=["NgayDang", "PriceVND"]).copy()
    df = df[df["PriceVND"] > 0]

    # Remove extreme invalid values, not aggressive outlier removal.
    df = df[(df["PriceVND"] >= 100_000_000) & (df["PriceVND"] <= 200_000_000_000)].copy()
    if df["AreaM2"].notna().any():
        df = df[(df["AreaM2"].isna()) | ((df["AreaM2"] >= 10) & (df["AreaM2"] <= 1000))].copy()

    df = df.sort_values("NgayDang").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame):
    numeric_features = [
        "AreaM2",
        "BedRooms",
        "BathRooms",
        "Floor",
        "PostedYear",
        "PostedMonth",
    ]
    categorical_features = [
        "District",
        "Ward",
        "PropertyStatus",
        "Type",
        "LegalStatus",
        "Interior",
        "MainDirection",
        "BalconyDirection",
        "Characteristic",
        "SellerType",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns and df[c].notna().any()]
    categorical_features = [c for c in categorical_features if c in df.columns and df[c].notna().any()]
    return numeric_features, categorical_features
