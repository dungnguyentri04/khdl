# Dự đoán giá căn hộ chung cư TP.HCM bằng Machine Learning

Project này xây dựng mô hình dự đoán giá căn hộ chung cư tại TP.HCM dựa trên dữ liệu bất động sản Việt Nam từ Hugging Face Dataset `tinixai/vietnam-real-estates`, chợ tốt và mogi

---

## 1. Mục tiêu project

Project thực hiện các công việc chính:

- Thu thập dữ liệu bất động sản từ Hugging Face, chợ tốt và mogi
- Chuẩn hóa dữ liệu
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

| Nhóm dữ liệu       | Cột dữ liệu                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| Giá và diện tích   | `price`, `area`                                                              |
| Vị trí             | `province_name`, `district_name`, `ward_name`, `street_name`, `project_name` |
| Đặc điểm căn hộ    | `property_type_name`, `floor_count`, `bedroom_count`, `bathroom_count`       |
| Kích thước bổ sung | `frontage_width`, `house_depth`, `road_width`                                |
| Hướng              | `house_direction`, `balcony_direction`                                       |
| Thời gian          | `published_at`                                                               |

Trong phiên bản chính, project tập trung vào:

```text
province_name = Hồ Chí Minh
property_type_name = Căn hộ chung cư
```

Lý do là để mô hình tập trung vào một phân khúc cụ thể, tránh nhiễu từ nhiều loại bất động sản khác nhau như đất nền, nhà phố hoặc biệt thự.

---

Crawl dữ liệu Chợ Tốt

Chạy lệnh:

```bash
python src/crawl_chotot.py --max-pages 20 --limit 50 --output data/raw_chotot_with_date.csv
```

...

## 3. Cài đặt môi trường

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

## 4. Huấn luyện mô hình

```bash
python src/train_merge_hf_chotot_mogi_fixed_split.py --use-hf-cache --hf-cache outputs_merged/hf_clean_schema.csv --chotot-csv data/raw_chotot_with_date.csv --mogi-csv data/mogi_dataset.csv --output-dir outputs_mogi_train_jun_to_feb_test_mar_to_may --train-start 2025-06-01 --train-end 2026-02-28 --test-start 2026-03-01 --test-end 2026-05-31

```

---

## 5. Cách chia train/test

Project sử dụng `time-based split`, không dùng chia ngẫu nhiên.

Ví dụ với dữ liệu nhiều tháng:

```text
Train: từ tháng 6 năm 2025 đến tháng 2 năm 2026
Test : từ tháng 3 năm 2026 đến tháng 5 năm 2026
```

Cách chia này phù hợp với yêu cầu dự đoán tương lai. Mô hình chỉ được học từ dữ liệu quá khứ và được kiểm thử trên dữ liệu xảy ra sau đó.

## 6. Chạy giao diện web

Project có giao diện web Flask để nhập form thông tin căn hộ và nhận giá dự đoán.

### 6.1. Cài Flask

```bash
python -m pip install Flask
```

### 6.2. Chạy web

```bash
 python src/web_app.py --model-dir outputs_mogi_train_jun_to_feb_test_mar_to_may --host 127.0.0.1 --port 5000
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

## Một số bộ thông tin để test web

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
