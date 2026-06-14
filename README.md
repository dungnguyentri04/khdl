# Dự đoán giá căn hộ chung cư TP.HCM bằng Machine Learning

Project này xây dựng mô hình dự đoán giá căn hộ chung cư tại TP.HCM dựa trên dữ liệu bất động sản Việt Nam từ Hugging Face Dataset `tinixai/vietnam-real-estates`.

Mục tiêu chính của project là sử dụng dữ liệu bất động sản trong quá khứ để dự đoán giá ở giai đoạn tương lai. Vì vậy, project không chia train/test ngẫu nhiên mà sử dụng cách chia theo thời gian (`time-based split`).

---

## 1. Mục tiêu project

Project thực hiện các công việc chính:

- Thu thập dữ liệu bất động sản từ Hugging Face.
- Lọc dữ liệu căn hộ chung cư tại TP.HCM.
- Tiền xử lý dữ liệu: xử lý dữ liệu thiếu, dữ liệu bất thường, biến thời gian và biến phân loại.
- Huấn luyện mô hình dự đoán giá bất động sản.
- Đánh giá mô hình bằng MAE, RMSE và R².
- Tạo giao diện web để người dùng nhập thông tin căn hộ và nhận kết quả dự đoán.
- Tạo biểu đồ phục vụ báo cáo.

---

## 2. Dataset sử dụng

Dataset được sử dụng:

```python
from datasets import load_dataset

ds = load_dataset("tinixai/vietnam-real-estates")
```

Các trường dữ liệu chính được sử dụng trong project gồm:

| Nhóm dữ liệu | Cột dữ liệu |
|---|---|
| Giá và diện tích | `price`, `area` |
| Vị trí | `province_name`, `district_name`, `ward_name`, `street_name`, `project_name` |
| Đặc điểm căn hộ | `property_type_name`, `floor_count`, `bedroom_count`, `bathroom_count` |
| Kích thước bổ sung | `frontage_width`, `house_depth`, `road_width` |
| Hướng | `house_direction`, `balcony_direction` |
| Thời gian | `published_at` |

Trong phiên bản chính, project tập trung vào:

```text
province_name = Hồ Chí Minh
property_type_name = Căn hộ chung cư
```

Lý do là để mô hình tập trung vào một phân khúc cụ thể, tránh nhiễu từ nhiều loại bất động sản khác nhau như đất nền, nhà phố hoặc biệt thự.

---

## 3. Cấu trúc thư mục

```text
VietnamRealEstate_HF_TimeSplit_Project
│
├── src
│   ├── inspect_dataset.py
│   ├── train_hf_real_estate_timesplit.py
│   ├── predict_one_hf.py
│   ├── make_report_charts.py
│   └── web_app.py
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
├── outputs_hcm_apartment_full
│   ├── best_model.joblib
│   ├── metrics.csv
│   ├── predictions.csv
│   └── summary.json
│
├── report_charts
│   ├── hinh_4_1_phan_phoi_gia.png
│   ├── hinh_4_2_phan_phoi_dien_tich.png
│   ├── hinh_4_3_so_luong_theo_quan.png
│   ├── hinh_4_4_boxplot_gia_theo_quan.png
│   ├── hinh_4_5_scatter_dien_tich_gia.png
│   ├── hinh_4_6_thuc_te_vs_du_doan.png
│   └── hinh_4_7_phan_phoi_sai_so.png
│
├── requirements.txt
├── requirements_web.txt
└── README.md
```

---

## 4. Cài đặt môi trường

Nên sử dụng môi trường ảo để tránh ảnh hưởng đến các project Python khác.

### Bước 1: Tạo môi trường ảo

```bash
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo

Trên Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi quyền chạy script, dùng:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Khi kích hoạt thành công, terminal sẽ có dạng:

```text
(.venv) PS D:\VietnamRealEstate_HF_TimeSplit_Project>
```

### Bước 3: Cài thư viện

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu dùng giao diện web, cài thêm Flask:

```bash
python -m pip install Flask
```

Hoặc:

```bash
python -m pip install -r requirements_web.txt
```

Nếu muốn tạo biểu đồ cho báo cáo, cần có `matplotlib`:

```bash
python -m pip install matplotlib
```

---

## 5. Kiểm tra nhanh dataset

Chạy lệnh sau để xem thử dữ liệu và các cột trong dataset:

```bash
python src/inspect_dataset.py --max-rows 20
```

Lệnh này giúp kiểm tra dataset có tải được không và các cột dữ liệu có đúng như mong muốn không.

---

## 6. Huấn luyện mô hình

### 6.1. Train bản căn hộ chung cư TP.HCM

Lệnh khuyến nghị:

```bash
python src/train_hf_real_estate_timesplit.py --province "Hồ Chí Minh" --property-type "Căn hộ chung cư" --max-rows 300000 --scan-limit 4000000 --split latest_month --output-dir outputs_hcm_apartment_full
```

Ý nghĩa các tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--province "Hồ Chí Minh"` | Chỉ lấy dữ liệu tại TP.HCM |
| `--property-type "Căn hộ chung cư"` | Chỉ lấy dữ liệu căn hộ chung cư |
| `--max-rows 300000` | Số dòng tối đa giữ lại sau khi lọc |
| `--scan-limit 4000000` | Số dòng tối đa quét từ dataset |
| `--split latest_month` | Lấy tháng mới nhất làm test, các tháng trước làm train |
| `--output-dir outputs_hcm_apartment_full` | Thư mục lưu model và kết quả |

### 6.2. Train bản nhẹ để test nhanh

Nếu máy yếu hoặc muốn test nhanh trước:

```bash
python src/train_hf_real_estate_timesplit.py --province "Hồ Chí Minh" --property-type "Căn hộ chung cư" --max-rows 80000 --scan-limit 500000 --split latest_month --output-dir outputs_hcm_apartment
```

### 6.3. Train toàn bộ dữ liệu không lọc tỉnh/thành

Có thể train toàn bộ dữ liệu, nhưng thời gian chạy sẽ lâu hơn và mô hình có thể nhiễu hơn:

```bash
python src/train_hf_real_estate_timesplit.py --max-rows 300000 --scan-limit 1000000 --split latest_month --output-dir outputs_all
```

---

## 7. Cách chia train/test

Project sử dụng `time-based split`, không dùng chia ngẫu nhiên.

Ví dụ với dữ liệu nhiều tháng:

```text
Train: các tháng trong quá khứ
Test : tháng mới nhất
```

Cách chia này phù hợp với yêu cầu dự đoán tương lai. Mô hình chỉ được học từ dữ liệu quá khứ và được kiểm thử trên dữ liệu xảy ra sau đó.

Ví dụ:

```text
Train: dữ liệu trước tháng 03/2026
Test : dữ liệu tháng 03/2026
```

Cách đánh giá này thực tế hơn so với chia ngẫu nhiên, vì tránh việc dữ liệu tương lai bị trộn vào tập huấn luyện.

---

## 8. Kết quả mô hình

Kết quả thực nghiệm tốt nhất hiện tại:

| Model | MAE VNĐ | RMSE VNĐ | R² | MAE tỷ VNĐ | RMSE tỷ VNĐ |
|---|---:|---:|---:|---:|---:|
| Random Forest | 1,112,624,942 | 2,440,688,265 | 0.898 | 1.113 | 2.441 |
| Ridge Regression | 1,255,380,490 | 5,654,427,251 | 0.454 | 1.255 | 5.654 |
| Baseline Median | 3,868,690,685 | 7,978,038,148 | -0.087 | 3.869 | 7.978 |

Random Forest là mô hình tốt nhất trong các mô hình đã thử nghiệm.

Ý nghĩa kết quả:

- `MAE = 1.113 tỷ VNĐ`: trung bình mỗi dự đoán lệch khoảng 1.113 tỷ VNĐ.
- `RMSE = 2.441 tỷ VNĐ`: mô hình vẫn có một số dự đoán sai lớn.
- `R² = 0.898`: mô hình giải thích được khoảng 89.8% biến động giá trong tập kiểm thử.

Lưu ý: Đây là bài toán hồi quy nên không dùng chỉ số accuracy như bài toán phân loại.

---

## 9. Dự đoán một căn hộ bằng command line

Sau khi train xong, có thể dự đoán một căn hộ mới bằng file:

```text
src/predict_one_hf.py
```

Ví dụ dự đoán căn hộ 70m², 2 phòng ngủ, 2 phòng tắm tại Quận 7, dự án Sunrise City:

```bash
python src/predict_one_hf.py --model-dir outputs_hcm_apartment_full --area 70 --bedrooms 2 --bathrooms 2 --floor-count 15 --province "Hồ Chí Minh" --district "7" --ward "Tân Phong" --property-type "Căn hộ chung cư" --project-name "Sunrise City" --published-year 2026 --published-month 5 --published-day 1
```

Ví dụ kết quả:

```text
Predicted price: 3,700,606,776 VND
Predicted price: 3.701 billion VND
```

Dự đoán này chỉ là giá tham khảo. Sai số trung bình kỳ vọng có thể dựa trên MAE của mô hình.

---

## 10. Chạy giao diện web

Project có giao diện web Flask để nhập form thông tin căn hộ và nhận giá dự đoán.

### 10.1. Cài Flask

```bash
python -m pip install Flask
```

### 10.2. Chạy web

```bash
python src/web_app.py --model-dir outputs_hcm_apartment_full --host 127.0.0.1 --port 5000
```

Sau đó mở trình duyệt:

```text
http://127.0.0.1:5000
```

### 10.3. Thông tin có thể nhập trên web

Web cho phép nhập các thông tin:

- Diện tích.
- Số phòng ngủ.
- Số phòng tắm.
- Số tầng.
- Tỉnh/Thành.
- Quận/Huyện.
- Phường/Xã.
- Tên đường.
- Tên dự án.
- Loại bất động sản.
- Hướng nhà.
- Hướng ban công.
- Năm/tháng/ngày dự đoán.

Sau khi bấm nút dự đoán, web hiển thị:

- Giá dự đoán.
- Giá theo tỷ VNĐ.
- Sai số trung bình tham khảo theo MAE.
- Khoảng giá tham khảo.
- R² của mô hình.

---

## 11. Tạo biểu đồ cho báo cáo

Sau khi train xong và có file:

```text
outputs_hcm_apartment_full/predictions.csv
outputs_hcm_apartment_full/metrics.csv
```

chạy lệnh:

```bash
python src/make_report_charts.py
```

Các biểu đồ sẽ được lưu trong thư mục:

```text
report_charts
```

Các hình phục vụ báo cáo gồm:

| File | Nội dung |
|---|---|
| `hinh_4_1_phan_phoi_gia.png` | Phân phối giá căn hộ |
| `hinh_4_2_phan_phoi_dien_tich.png` | Phân phối diện tích căn hộ |
| `hinh_4_3_so_luong_theo_quan.png` | Số lượng tin đăng theo quận/huyện |
| `hinh_4_4_boxplot_gia_theo_quan.png` | Boxplot giá căn hộ theo quận/huyện |
| `hinh_4_5_scatter_dien_tich_gia.png` | Quan hệ giữa diện tích và giá |
| `hinh_4_6_thuc_te_vs_du_doan.png` | So sánh giá thực tế và giá dự đoán |
| `hinh_4_7_phan_phoi_sai_so.png` | Phân phối sai số dự đoán |
| `hinh_5_1_so_sanh_mae.png` | So sánh MAE giữa các mô hình |
| `hinh_5_2_so_sanh_r2.png` | So sánh R² giữa các mô hình |

---

## 12. Một số bộ thông tin để test web

### Test 1: Căn hộ Quận 7

```text
Diện tích: 70
Số phòng ngủ: 2
Số phòng tắm: 2
Tầng: 15
Tỉnh/Thành: Hồ Chí Minh
Quận/Huyện: 7
Phường/Xã: Tân Phong
Tên dự án: Sunrise City
Loại BĐS: Căn hộ chung cư
Năm: 2026
Tháng: 5
Ngày: 1
```

### Test 2: Căn hộ Bình Thạnh

```text
Diện tích: 55
Số phòng ngủ: 1
Số phòng tắm: 1
Tầng: 12
Tỉnh/Thành: Hồ Chí Minh
Quận/Huyện: Bình Thạnh
Phường/Xã: 22
Tên dự án: Vinhomes Central Park
Loại BĐS: Căn hộ chung cư
Năm: 2026
Tháng: 5
Ngày: 5
```

### Test 3: Căn hộ Thủ Đức

```text
Diện tích: 85
Số phòng ngủ: 3
Số phòng tắm: 2
Tầng: 20
Tỉnh/Thành: Hồ Chí Minh
Quận/Huyện: Thủ Đức
Phường/Xã: An Phú
Tên dự án: Masteri An Phú
Loại BĐS: Căn hộ chung cư
Năm: 2026
Tháng: 5
Ngày: 10
```

### Test 4: Căn hộ Quận 1

```text
Diện tích: 120
Số phòng ngủ: 4
Số phòng tắm: 3
Tầng: 30
Tỉnh/Thành: Hồ Chí Minh
Quận/Huyện: 1
Phường/Xã: Bến Nghé
Tên dự án: The Marq
Loại BĐS: Căn hộ chung cư
Năm: 2026
Tháng: 5
Ngày: 25
```

---

## 13. Giải thích các chỉ số đánh giá

### MAE

MAE là sai số tuyệt đối trung bình giữa giá thực tế và giá dự đoán.

```text
MAE càng thấp thì mô hình càng tốt.
```

### RMSE

RMSE là căn bậc hai của sai số bình phương trung bình.

```text
RMSE phạt mạnh các dự đoán sai lớn.
```

Nếu RMSE cao hơn MAE nhiều, nghĩa là mô hình có một số dự đoán lệch lớn.

### R²

R² cho biết mô hình giải thích được bao nhiêu phần biến động của giá.

```text
R² càng gần 1 thì mô hình càng tốt.
```

Ví dụ:

```text
R² = 0.898
```

có thể hiểu là mô hình giải thích được khoảng 89.8% biến động giá trên tập kiểm thử.

---

## 14. Lưu ý khi trình bày báo cáo

Không nên nói:

```text
Mô hình có độ chính xác 89.8%
```

Nên nói:

```text
Mô hình đạt R² = 0.898, cho thấy khả năng giải thích biến động giá tốt.
```

Không nên nói giá dự đoán là giá chính xác tuyệt đối.

Nên nói:

```text
Giá dự đoán là mức giá tham khảo. Sai số trung bình của mô hình khoảng 1.113 tỷ VNĐ theo MAE.
```

---

## 15. Hướng phát triển

Project có thể được cải thiện theo các hướng:

- Cập nhật thêm dữ liệu mới theo thời gian.
- Thử nghiệm XGBoost, LightGBM hoặc CatBoost.
- Bổ sung tọa độ địa lý.
- Bổ sung khoảng cách đến trung tâm, trường học, bệnh viện, metro.
- Dùng thêm dữ liệu nội dung tiêu đề/mô tả sau khi kiểm soát rò rỉ dữ liệu.
- Triển khai web lên server để người dùng truy cập trực tuyến.
- Tạo khoảng tin cậy dự đoán thay vì chỉ đưa ra một giá trị duy nhất.

---

## 16. Tóm tắt

Project đã xây dựng được pipeline hoàn chỉnh:

```text
Load dataset
→ Lọc dữ liệu căn hộ chung cư TP.HCM
→ Tiền xử lý dữ liệu
→ Chia train/test theo thời gian
→ Train mô hình
→ Đánh giá bằng MAE, RMSE, R²
→ Dự đoán bằng command line
→ Triển khai web form
→ Tạo biểu đồ báo cáo
```

Mô hình tốt nhất hiện tại là Random Forest với:

```text
MAE  = 1.113 tỷ VNĐ
RMSE = 2.441 tỷ VNĐ
R²   = 0.898
```

Kết quả này cho thấy mô hình có khả năng dự đoán giá tương đối tốt và phù hợp để dùng làm công cụ tham khảo giá căn hộ chung cư tại TP.HCM.
