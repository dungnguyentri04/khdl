"""Crawl Chợ Tốt/Nhà Tốt apartment listings with time columns.

This script uses the public ad-listing endpoint commonly used by Chợ Tốt pages.
Because Chợ Tốt can change parameters or rate limits, keep max-pages moderate and
increase sleep if requests fail.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

API_URL = "https://gateway.chotot.com/v1/public/ad-listing"


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl Chợ Tốt apartment data with date columns")
    parser.add_argument("--output", default="data/raw_chotot_with_date.csv", help="Output CSV path")
    parser.add_argument("--region-v2", default="13000", help="Region code. 13000 is commonly used for TP.HCM")
    parser.add_argument("--category", default="1010", help="Category code. 1010 is commonly used for apartment listings")
    parser.add_argument("--max-pages", type=int, default=20, help="Number of pages to crawl")
    parser.add_argument("--limit", type=int, default=50, help="Ads per page")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between requests")
    parser.add_argument("--api-url", default=API_URL, help="API endpoint")
    parser.add_argument("--extra-param", action="append", default=[], help="Extra API params in key=value format")
    return parser.parse_args()


def parse_extra_params(items: List[str]) -> Dict[str, str]:
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-param '{item}'. Use key=value format.")
        key, value = item.split("=", 1)
        params[key] = value
    return params


def timestamp_to_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = float(value)
        # Chợ Tốt list_time often uses milliseconds.
        if ts > 10_000_000_000:
            ts = ts / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def extract_param_value(ad: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Extract value from flexible Chợ Tốt ad structures."""
    for key in keys:
        if key in ad and ad[key] not in [None, ""]:
            return ad[key]

    # API sometimes stores details in ad_params / parameters.
    for container_name in ["ad_params", "parameters", "params"]:
        params = ad.get(container_name)
        if isinstance(params, list):
            for item in params:
                if not isinstance(item, dict):
                    continue
                item_keys = [
                    str(item.get("id", "")).lower(),
                    str(item.get("name", "")).lower(),
                    str(item.get("label", "")).lower(),
                    str(item.get("key", "")).lower(),
                ]
                for wanted in keys:
                    if wanted.lower() in item_keys:
                        return item.get("value") or item.get("value_name") or item.get("label")
    return None


def build_ad_url(ad: Dict[str, Any]) -> Optional[str]:
    for key in ["ad_link", "webp_image", "url"]:
        val = ad.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    list_id = ad.get("list_id") or ad.get("ad_id")
    if list_id:
        return f"https://www.chotot.com/{list_id}.htm"
    return None


def flatten_ad(ad: Dict[str, Any], crawl_time: str) -> Dict[str, Any]:
    list_time = ad.get("list_time") or ad.get("date_timestamp") or ad.get("created_at")
    ngay_dang = timestamp_to_datetime(list_time)

    row = {
        "AdId": ad.get("ad_id"),
        "ListId": ad.get("list_id"),
        "Subject": ad.get("subject"),
        "Address": ad.get("address"),
        "Ward": ad.get("ward_name") or ad.get("ward"),
        "District": ad.get("area_name") or ad.get("area"),
        "Region": ad.get("region_name") or ad.get("region"),
        "PriceVND": ad.get("price"),
        "PriceString": ad.get("price_string"),
        "AreaM2": extract_param_value(ad, ["size", "area", "living_size", "m2"]),
        "BedRooms": extract_param_value(ad, ["rooms", "bedrooms", "apartment_bedroom"]),
        "BathRooms": extract_param_value(ad, ["toilets", "bathrooms", "apartment_toilet"]),
        "Floor": extract_param_value(ad, ["floors", "floor", "apartment_floor"]),
        "PropertyStatus": extract_param_value(ad, ["condition_ad", "property_status", "condition"]),
        "Type": extract_param_value(ad, ["apartment_type", "type", "category"]),
        "LegalStatus": extract_param_value(ad, ["property_legal_document", "legal_status", "legal"]),
        "Interior": extract_param_value(ad, ["furnishing_sell", "interior", "furnishing"]),
        "MainDirection": extract_param_value(ad, ["direction", "house_direction"]),
        "BalconyDirection": extract_param_value(ad, ["balcony_direction"]),
        "SellerType": ad.get("account_type") or ad.get("seller_type"),
        "PostDateText": ad.get("date"),
        "ListTimeRaw": list_time,
        "NgayDang": ngay_dang,
        "CrawlDate": crawl_time,
        "Link": build_ad_url(ad),
    }
    return row


def request_page(session: requests.Session, api_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nhatot.com/",
    }
    response = session.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    extra = parse_extra_params(args.extra_param)
    crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    with requests.Session() as session:
        for page in range(args.max_pages):
            offset = page * args.limit
            params = {
                "region_v2": args.region_v2,
                "cg": args.category,
                "limit": args.limit,
                "o": offset,
                "st": "s,k",
            }
            params.update(extra)

            try:
                payload = request_page(session, args.api_url, params)
            except Exception as exc:
                print(f"[WARN] Page {page + 1} failed: {exc}")
                time.sleep(args.sleep)
                continue

            ads = payload.get("ads") or payload.get("data") or []
            if not ads:
                print(f"[INFO] No ads at page {page + 1}; stop.")
                break

            for ad in ads:
                if isinstance(ad, dict):
                    rows.append(flatten_ad(ad, crawl_time))

            print(f"[OK] page={page + 1}, ads={len(ads)}, total_rows={len(rows)}")
            time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] Không crawl được dữ liệu. Hãy thử giảm max-pages, tăng sleep, hoặc kiểm tra lại category/region/API.")
    else:
        df = df.drop_duplicates(subset=["AdId", "ListId", "Subject"], keep="first")
        df.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"[DONE] Saved {len(df)} rows to {output}")
        print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
