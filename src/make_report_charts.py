from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CẤU HÌNH CHÍNH
# ============================================================

RUN_DIR = Path("outputs_mogi_train_jun_to_feb_test_mar_to_may")
OUTPUT_DIR = Path("report_charts_train_jun_to_feb_test_mar_to_may")

PRED_PATH = RUN_DIR / "predictions.csv"
METRICS_PATH = RUN_DIR / "metrics.csv"
TRAIN_PATH = RUN_DIR / "train_data.csv"
TEST_PATH = RUN_DIR / "test_data.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

# Giới hạn để biểu đồ dễ nhìn hơn
MAX_PRICE_BILLION = 50
MAX_AREA = 300


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def save_fig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_prediction_data():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {PRED_PATH}")

    df = pd.read_csv(PRED_PATH)

    required_cols = [
        "price",
        "predicted_price",
        "absolute_error",
        "area",
        "district_name",
        "source",
        "published_at",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"predictions.csv thiếu các cột: {missing_cols}")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")
    df["absolute_error"] = pd.to_numeric(df["absolute_error"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    df = df.dropna(
        subset=[
            "price",
            "predicted_price",
            "absolute_error",
            "area",
            "published_at",
        ]
    )

    df["price_billion"] = df["price"] / 1_000_000_000
    df["predicted_billion"] = df["predicted_price"] / 1_000_000_000
    df["error_billion"] = df["absolute_error"] / 1_000_000_000
    df["month"] = df["published_at"].dt.to_period("M").astype(str)

    return df


def load_split_data():
    train_df = None
    test_df = None

    if TRAIN_PATH.exists():
        train_df = pd.read_csv(TRAIN_PATH)
        train_df["published_at"] = pd.to_datetime(
            train_df["published_at"],
            errors="coerce"
        )
        train_df["split"] = "Train"

    if TEST_PATH.exists():
        test_df = pd.read_csv(TEST_PATH)
        test_df["published_at"] = pd.to_datetime(
            test_df["published_at"],
            errors="coerce"
        )
        test_df["split"] = "Test"

    return train_df, test_df


# ============================================================
# VẼ BIỂU ĐỒ EDA TRÊN TẬP TEST
# ============================================================

def plot_price_distribution(df):
    plot_df = df[df["price_billion"] <= MAX_PRICE_BILLION]

    plt.figure(figsize=(8, 5))
    plt.hist(plot_df["price_billion"], bins=50)
    plt.xlabel("Giá căn hộ (tỷ VNĐ)")
    plt.ylabel("Số lượng")
    plt.title("Phân phối giá căn hộ trong tập kiểm thử")
    save_fig("hinh_4_1_phan_phoi_gia.png")


def plot_area_distribution(df):
    plot_df = df[df["area"] <= MAX_AREA]

    plt.figure(figsize=(8, 5))
    plt.hist(plot_df["area"], bins=50)
    plt.xlabel("Diện tích (m²)")
    plt.ylabel("Số lượng")
    plt.title("Phân phối diện tích căn hộ trong tập kiểm thử")
    save_fig("hinh_4_2_phan_phoi_dien_tich.png")


def plot_district_counts(df):
    district_counts = df["district_name"].astype(str).value_counts().head(15)

    plt.figure(figsize=(10, 6))
    district_counts.sort_values().plot(kind="barh")
    plt.xlabel("Số lượng tin đăng")
    plt.ylabel("Quận/Huyện")
    plt.title("Top 15 quận/huyện có nhiều tin đăng nhất trong tập kiểm thử")
    save_fig("hinh_4_3_so_luong_theo_quan.png")


def plot_boxplot_by_district(df):
    plot_df = df[df["price_billion"] <= MAX_PRICE_BILLION].copy()

    top_districts = (
        plot_df["district_name"]
        .astype(str)
        .value_counts()
        .head(10)
        .index
    )

    box_df = plot_df[
        plot_df["district_name"].astype(str).isin(top_districts)
    ]

    data = [
        box_df[box_df["district_name"].astype(str) == district]["price_billion"].dropna()
        for district in top_districts
    ]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=top_districts, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Giá căn hộ (tỷ VNĐ)")
    plt.title("Boxplot giá căn hộ theo quận/huyện trong tập kiểm thử")
    save_fig("hinh_4_4_boxplot_gia_theo_quan.png")


def plot_area_vs_price(df):
    plot_df = df[
        (df["price_billion"] <= MAX_PRICE_BILLION)
        & (df["area"] <= MAX_AREA)
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["area"], plot_df["price_billion"], alpha=0.35, s=10)
    plt.xlabel("Diện tích (m²)")
    plt.ylabel("Giá căn hộ (tỷ VNĐ)")
    plt.title("Quan hệ giữa diện tích và giá căn hộ trong tập kiểm thử")
    save_fig("hinh_4_5_scatter_dien_tich_gia.png")


def plot_actual_vs_predicted(df):
    plot_df = df[
        (df["price_billion"] <= MAX_PRICE_BILLION)
        & (df["predicted_billion"] <= MAX_PRICE_BILLION)
    ]

    plt.figure(figsize=(7, 7))
    plt.scatter(
        plot_df["price_billion"],
        plot_df["predicted_billion"],
        alpha=0.35,
        s=10
    )

    min_val = min(
        plot_df["price_billion"].min(),
        plot_df["predicted_billion"].min()
    )

    max_val = max(
        plot_df["price_billion"].max(),
        plot_df["predicted_billion"].max()
    )

    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Giá thực tế (tỷ VNĐ)")
    plt.ylabel("Giá dự đoán (tỷ VNĐ)")
    plt.title("So sánh giá thực tế và giá dự đoán")
    save_fig("hinh_4_6_thuc_te_vs_du_doan.png")


def plot_error_distribution(df):
    q99 = df["error_billion"].quantile(0.99)
    plot_df = df[df["error_billion"] <= q99]

    plt.figure(figsize=(8, 5))
    plt.hist(plot_df["error_billion"], bins=50)
    plt.xlabel("Sai số tuyệt đối (tỷ VNĐ)")
    plt.ylabel("Số lượng")
    plt.title("Phân phối sai số dự đoán")
    save_fig("hinh_4_7_phan_phoi_sai_so.png")


def plot_test_source_distribution(df):
    source_counts = df["source"].astype(str).value_counts()

    plt.figure(figsize=(7, 5))
    source_counts.plot(kind="bar")
    plt.xlabel("Nguồn dữ liệu")
    plt.ylabel("Số lượng bản ghi")
    plt.title("Phân bố nguồn dữ liệu trong tập kiểm thử")
    plt.xticks(rotation=0)
    save_fig("hinh_4_8_phan_bo_nguon_du_lieu_test.png")


def plot_test_month_distribution(df):
    monthly_counts = df["month"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    monthly_counts.plot(kind="bar")
    plt.xlabel("Tháng")
    plt.ylabel("Số lượng bản ghi")
    plt.title("Số lượng bản ghi kiểm thử theo tháng")
    plt.xticks(rotation=45, ha="right")
    save_fig("hinh_4_9_so_luong_test_theo_thang.png")


# ============================================================
# BIỂU ĐỒ TRAIN/TEST THEO THÁNG VÀ NGUỒN
# ============================================================

def plot_train_test_by_month(train_df, test_df):
    if train_df is None or test_df is None:
        print("[SKIP] Không tìm thấy train_data.csv hoặc test_data.csv")
        return

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.dropna(subset=["published_at"])

    all_df["month"] = all_df["published_at"].dt.to_period("M").astype(str)

    pivot = (
        all_df
        .groupby(["month", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, figsize=(11, 5))
    plt.xlabel("Tháng")
    plt.ylabel("Số lượng bản ghi")
    plt.title("Phân chia train/test theo tháng")
    plt.xticks(rotation=45, ha="right")
    save_fig("hinh_4_10_train_test_theo_thang.png")


def plot_source_by_month(train_df, test_df):
    if train_df is None or test_df is None:
        print("[SKIP] Không tìm thấy train_data.csv hoặc test_data.csv")
        return

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.dropna(subset=["published_at"])

    all_df["month"] = all_df["published_at"].dt.to_period("M").astype(str)

    pivot = (
        all_df
        .groupby(["month", "source"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, figsize=(11, 5))
    plt.xlabel("Tháng")
    plt.ylabel("Số lượng bản ghi")
    plt.title("Số lượng bản ghi theo tháng và nguồn dữ liệu")
    plt.xticks(rotation=45, ha="right")
    save_fig("hinh_4_11_so_luong_theo_thang_va_nguon.png")


# ============================================================
# BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ============================================================

def plot_model_metrics():
    if not METRICS_PATH.exists():
        print(f"Không tìm thấy file metrics: {METRICS_PATH}")
        return

    metrics = pd.read_csv(METRICS_PATH)

    required_cols = [
        "Model",
        "MAE_BILLION_VND",
        "RMSE_BILLION_VND",
        "R2",
    ]

    missing_cols = [col for col in required_cols if col not in metrics.columns]
    if missing_cols:
        raise ValueError(f"metrics.csv thiếu các cột: {missing_cols}")

    # Hình 5.1: MAE
    metrics_mae = metrics.sort_values("MAE_BILLION_VND")

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_mae["Model"], metrics_mae["MAE_BILLION_VND"])
    plt.ylabel("MAE (tỷ VNĐ)")
    plt.title("So sánh MAE giữa các mô hình")
    plt.xticks(rotation=20, ha="right")

    for i, value in enumerate(metrics_mae["MAE_BILLION_VND"]):
        plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    save_fig("hinh_5_1_so_sanh_mae.png")

    # Hình 5.2: RMSE
    metrics_rmse = metrics.sort_values("RMSE_BILLION_VND")

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_rmse["Model"], metrics_rmse["RMSE_BILLION_VND"])
    plt.ylabel("RMSE (tỷ VNĐ)")
    plt.title("So sánh RMSE giữa các mô hình")
    plt.xticks(rotation=20, ha="right")

    for i, value in enumerate(metrics_rmse["RMSE_BILLION_VND"]):
        plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    save_fig("hinh_5_2_so_sanh_rmse.png")

    # Hình 5.3: R²
    metrics_r2 = metrics.sort_values("R2", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_r2["Model"], metrics_r2["R2"])
    plt.ylabel("R²")
    plt.title("So sánh R² giữa các mô hình")
    plt.xticks(rotation=20, ha="right")

    for i, value in enumerate(metrics_r2["R2"]):
        plt.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    save_fig("hinh_5_3_so_sanh_r2.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("===== MAKE REPORT CHARTS =====")
    print(f"Run dir      : {RUN_DIR}")
    print(f"Prediction  : {PRED_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Train data  : {TRAIN_PATH}")
    print(f"Test data   : {TEST_PATH}")
    print(f"Output dir  : {OUTPUT_DIR}")

    df = load_prediction_data()
    train_df, test_df = load_split_data()

    print("\n===== TEST DATA INFO =====")
    print(f"Rows: {len(df):,}")
    print(
        f"Price range: "
        f"{df['price_billion'].min():.3f} -> {df['price_billion'].max():.3f} tỷ"
    )
    print(
        f"Area range : "
        f"{df['area'].min():.3f} -> {df['area'].max():.3f} m²"
    )

    print("\nRows by test month:")
    print(df["month"].value_counts().sort_index().to_string())

    print("\nRows by test source:")
    print(df["source"].value_counts().to_string())

    plot_price_distribution(df)
    plot_area_distribution(df)
    plot_district_counts(df)
    plot_boxplot_by_district(df)
    plot_area_vs_price(df)
    plot_actual_vs_predicted(df)
    plot_error_distribution(df)
    plot_test_source_distribution(df)
    plot_test_month_distribution(df)

    plot_train_test_by_month(train_df, test_df)
    plot_source_by_month(train_df, test_df)

    plot_model_metrics()

    print("\n[DONE] All charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()