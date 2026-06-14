from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one Vietnam real-estate price")

    parser.add_argument("--model-dir", default="outputs_hcm_apartment")

    parser.add_argument("--area", type=float, required=True)
    parser.add_argument("--floor-count", type=float, default=np.nan)
    parser.add_argument("--frontage-width", type=float, default=np.nan)
    parser.add_argument("--house-depth", type=float, default=np.nan)
    parser.add_argument("--road-width", type=float, default=np.nan)
    parser.add_argument("--bedrooms", type=float, default=np.nan)
    parser.add_argument("--bathrooms", type=float, default=np.nan)

    parser.add_argument("--property-type", default="Căn hộ chung cư")
    parser.add_argument("--province", default="Hồ Chí Minh")
    parser.add_argument("--district", default="")
    parser.add_argument("--ward", default="")
    parser.add_argument("--street", default="")
    parser.add_argument("--project-name", default="")
    parser.add_argument("--house-direction", default="Unknown")
    parser.add_argument("--balcony-direction", default="Unknown")

    parser.add_argument("--published-year", type=int, default=2026)
    parser.add_argument("--published-month", type=int, default=6)
    parser.add_argument("--published-day", type=int, default=1)
    parser.add_argument("--published-dayofweek", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "best_model.joblib"
    summary_path = model_dir / "summary.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {model_path}. Hãy train model trước.")
    if not summary_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {summary_path}. Hãy train model trước.")

    model = joblib.load(model_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    numeric_features = summary["numeric_features"]
    categorical_features = summary["categorical_features"]
    feature_cols = numeric_features + categorical_features

    row = {
        "area": args.area,
        "floor_count": args.floor_count,
        "frontage_width": args.frontage_width,
        "house_depth": args.house_depth,
        "road_width": args.road_width,
        "bedroom_count": args.bedrooms,
        "bathroom_count": args.bathrooms,
        "posted_year": args.published_year,
        "posted_month": args.published_month,
        "posted_day": args.published_day,
        "posted_dayofweek": args.published_dayofweek,

        "property_type_name": args.property_type,
        "province_name": args.province,
        "district_name": args.district,
        "ward_name": args.ward,
        "street_name": args.street,
        "project_name": args.project_name,
        "house_direction": args.house_direction,
        "balcony_direction": args.balcony_direction,
    }

    df = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}])

    pred_log = model.predict(df)[0]
    pred_vnd = float(np.expm1(pred_log))

    print("\n===== INPUT =====")
    for col in feature_cols:
        print(f"{col}: {df.loc[0, col]}")

    print("\n===== PREDICTION =====")
    print(f"Predicted price: {pred_vnd:,.0f} VND")
    print(f"Predicted price: {pred_vnd / 1_000_000_000:.3f} billion VND")


if __name__ == "__main__":
    main()
