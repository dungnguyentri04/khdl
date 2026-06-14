from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"


def clean_text(x, default: str = "Unknown") -> str:
    if x is None:
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

    for prefix in ["Quận ", "Huyện ", "Thành phố ", "Thị xã "]:
        if x.startswith(prefix):
            x = x.replace(prefix, "", 1).strip()

    return x if x else "Unknown"


def normalize_ward(x) -> str:
    x = clean_text(x)
    for prefix in ["Phường ", "Xã ", "Thị trấn "]:
        if x.startswith(prefix):
            x = x.replace(prefix, "", 1).strip()
    return x if x else "Unknown"


def normalize_direction(x) -> str:
    x = clean_text(x)

    # Nếu dữ liệu là mã số hướng từ Chợ Tốt mà chưa có bảng decode thì để Unknown
    if re.fullmatch(r"\d+(\.0)?", x):
        return "Unknown"

    valid = {
        "Unknown", "Đông", "Tây", "Nam", "Bắc",
        "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc",
    }
    return x if x in valid else "Unknown"


def resolve_model_dir(model_dir: str) -> Path:
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = PROJECT_ROOT / model_dir_path
    return model_dir_path


def load_model_bundle(model_dir: str):
    model_dir_path = resolve_model_dir(model_dir)

    model_path = model_dir_path / "best_model.joblib"
    summary_path = model_dir_path / "summary.json"
    metrics_path = model_dir_path / "metrics.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Không tìm thấy summary.json: {summary_path}")

    model = joblib.load(model_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = None
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)

        # Ưu tiên lấy chỉ số Random Forest vì đây là model chính trong báo cáo
        if "Model" in metrics_df.columns and (metrics_df["Model"] == "Random_Forest").any():
            metrics = metrics_df[metrics_df["Model"] == "Random_Forest"].iloc[0].to_dict()
        else:
            metrics = metrics_df.iloc[0].to_dict()

    return model, summary, metrics, model_dir_path


def parse_float(value, default=np.nan):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=1):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except ValueError:
        return default


def create_input_dataframe(form, summary):
    numeric_features = summary["numeric_features"]
    categorical_features = summary["categorical_features"]
    feature_cols = numeric_features + categorical_features

    published_year = parse_int(form.get("published_year"), 2026)
    published_month = parse_int(form.get("published_month"), 5)
    published_day = parse_int(form.get("published_day"), 1)

    try:
        published_dayofweek = date(published_year, published_month, published_day).weekday()
    except ValueError:
        published_dayofweek = 0

    row = {
        "area": parse_float(form.get("area")),
        "floor_count": parse_float(form.get("floor_count")),
        "frontage_width": parse_float(form.get("frontage_width")),
        "house_depth": parse_float(form.get("house_depth")),
        "road_width": parse_float(form.get("road_width")),
        "bedroom_count": parse_float(form.get("bedrooms")),
        "bathroom_count": parse_float(form.get("bathrooms")),

        "posted_year": published_year,
        "posted_month": published_month,
        "posted_day": published_day,
        "posted_dayofweek": published_dayofweek,

        "property_type_name": clean_text(form.get("property_type", "Căn hộ chung cư")),
        "province_name": normalize_province(form.get("province", "Hồ Chí Minh")),
        "district_name": normalize_district(form.get("district", "")),
        "ward_name": normalize_ward(form.get("ward", "")),
        "street_name": clean_text(form.get("street", "")),
        "project_name": clean_text(form.get("project_name", "")),
        "house_direction": normalize_direction(form.get("house_direction", "Unknown")),
        "balcony_direction": normalize_direction(form.get("balcony_direction", "Unknown")),
    }

    # Chỉ lấy đúng các cột model đã train
    df = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}])
    return df


def predict_price(model, input_df):
    # Model train trên log1p(price), nên cần chuyển ngược bằng expm1
    pred_log = model.predict(input_df)[0]
    pred_vnd = float(np.expm1(pred_log))
    return max(pred_vnd, 0)


def format_vnd(value):
    return f"{value:,.0f} VND"


def format_billion(value):
    return f"{value / 1_000_000_000:.3f} tỷ VNĐ"


def get_float_metric(metrics, key):
    if metrics is None or key not in metrics:
        return None
    try:
        return float(metrics[key])
    except Exception:
        return None


def create_app(model_dir: str):
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR)
    )

    model, summary, metrics, model_dir_path = load_model_bundle(model_dir)

    @app.route("/", methods=["GET", "POST"])
    def index():
        prediction = None
        error = None

        # Mặc định theo kịch bản outputs_merged:
        # Train đến 04/2026, test/predict tháng 05/2026
        input_values = {
            "area": "70",
            "bedrooms": "2",
            "bathrooms": "2",
            "floor_count": "15",
            "frontage_width": "",
            "house_depth": "",
            "road_width": "",
            "province": "Hồ Chí Minh",
            "district": "7",
            "ward": "Tân Phong",
            "street": "",
            "project_name": "Sunrise City",
            "property_type": "Căn hộ chung cư",
            "house_direction": "Unknown",
            "balcony_direction": "Unknown",
            "published_year": "2026",
            "published_month": "5",
            "published_day": "1",
        }

        if request.method == "POST":
            input_values.update({k: v for k, v in request.form.items()})

            try:
                input_df = create_input_dataframe(request.form, summary)
                pred_vnd = predict_price(model, input_df)

                mae_vnd = get_float_metric(metrics, "MAE_VND")
                rmse_vnd = get_float_metric(metrics, "RMSE_VND")
                r2 = get_float_metric(metrics, "R2")

                lower = None
                upper = None
                if mae_vnd is not None:
                    lower = max(pred_vnd - mae_vnd, 0)
                    upper = pred_vnd + mae_vnd

                prediction = {
                    "price_vnd": format_vnd(pred_vnd),
                    "price_billion": format_billion(pred_vnd),

                    "mae_vnd": format_vnd(mae_vnd) if mae_vnd is not None else None,
                    "mae_billion": format_billion(mae_vnd) if mae_vnd is not None else None,

                    "rmse_vnd": format_vnd(rmse_vnd) if rmse_vnd is not None else None,
                    "rmse_billion": format_billion(rmse_vnd) if rmse_vnd is not None else None,

                    "r2": f"{r2:.3f}" if r2 is not None else None,

                    "lower": format_billion(lower) if lower is not None else None,
                    "upper": format_billion(upper) if upper is not None else None,

                    "normalized_input": input_df.iloc[0].to_dict(),
                }

            except Exception as exc:
                error = str(exc)

        return render_template(
            "index.html",
            prediction=prediction,
            error=error,
            input_values=input_values,
            model_dir=model_dir,
        )

    @app.route("/health", methods=["GET"])
    def health():
        return {
            "status": "ok",
            "model_dir": str(model_dir_path),
            "template_dir": str(TEMPLATE_DIR),
            "static_dir": str(STATIC_DIR),
            "best_model": summary.get("best_model", "Unknown"),
            "split_strategy": summary.get("split_strategy", "Unknown"),
            "train_rows": summary.get("train_rows", None),
            "test_rows": summary.get("test_rows", None),
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default="outputs_merged",
        help="Folder chứa best_model.joblib, summary.json, metrics.csv"
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.model_dir)

    print()
    print("===== WEB APP STARTED =====")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Template dir : {TEMPLATE_DIR}")
    print(f"Static dir   : {STATIC_DIR}")
    print(f"Model dir    : {args.model_dir}")
    print(f"Open browser : http://{args.host}:{args.port}")
    print(f"Health check : http://{args.host}:{args.port}/health")
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
