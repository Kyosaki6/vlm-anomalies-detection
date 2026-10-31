# Chương 6: Phụ lục kỹ thuật

## 6.1. Bảng cấu hình phần cứng và phân công tải tính toán

Bảng dưới đây thống kê chi tiết cấu hình các nút tính toán của các thành viên trong nhóm tham gia thực nghiệm:

| Thành viên | Phần cứng (GPU / CPU / RAM) | Độ chính xác tính toán (Dtype) | Khối lượng tải tính toán phân bổ | Ghi chú kỹ thuật |
|---|---|---|---|---|
| **Phước** | NVIDIA GeForce RTX 3050 6GB VRAM | float16 | Đảm nhiệm Tầng 2 Generative VLM (lượng tử hóa 4-bit) + Shard 1 tiền tính toán embedding (khoảng 40% số video) | Kiến trúc Ampere, hỗ trợ xử lý nửa chính xác (fp16) mượt mà |
| **Huy** | NVIDIA GeForce RTX 3050 4GB VRAM | float16 | Shard 2 tiền tính toán embedding (khoảng 35% số video) + Thu thập và gắn nhãn 7 clip demo (iStock preview, chưa mua licence) | Kiến trúc Ampere, chạy suy luận nhúng SigLIP 2 và CLIP ViT-B/32 |
| **Nam** | NVIDIA GeForce GTX 1060 3GB VRAM | **float32** (chính xác đơn) | Shard 3 tiền tính toán embedding (khoảng 25% số video) + Xây dựng đồ thị chính sách Alert Policy Graph | Kiến trúc Pascal (sm_61). Không chạy float16 do tốc độ phần cứng suy giảm |
| **Thịnh** | Không có GPU rời; CPU + Google Colab T4 (dự phòng) | float16 (trên Colab) / float32 (trên CPU) | Nút dự phòng cho các shard embedding trễ + Phát triển toàn bộ bảng điều khiển Streamlit + Viết báo cáo | Đảm nhiệm kiểm chứng tính tái lập trên môi trường phi GPU |

## 6.2. Kiểm tra tính nhất quán giữa các thiết bị (Cross-Card Consistency)

Do sử dụng nhiều dòng GPU khác nhau (Ampere fp16 và Pascal fp32), hệ thống áp dụng quy trình kiểm tra chéo độ tương đồng véc-tơ trước khi ghép dữ liệu cache:
- Một video ngẫu nhiên được chạy trích xuất đặc trưng độc lập trên cả hai máy của Phước (RTX 3050 fp16) và Nam (GTX 1060 fp32).
- Độ tương đồng cosine trung bình giữa hai mảng véc-tơ nhúng trích xuất ra phải thỏa mãn điều kiện:
  $$\cos(v_{\text{Ampere}}, v_{\text{Pascal}}) > 0.999$$
- Thử nghiệm này xác nhận việc khác biệt về vi kiến trúc và kiểu dữ liệu không gây sai lệch không gian biểu diễn chung của mô hình.

## 6.3. Hướng dẫn cài đặt và lệnh tái lập thực nghiệm

Mọi câu lệnh dưới đây được thiết kế để chạy từ thư mục gốc của kho mã nguồn.

### 6.3.1. Thiết lập môi trường
```bash
# Cài đặt các gói phụ thuộc cơ bản và thư viện kiểm thử
pip install -r requirements-dev.txt

# Cài đặt thêm Streamlit cho bảng điều khiển
pip install -r requirements-app.txt
```

### 6.3.2. Chạy bộ kiểm thử tự động (Unit Tests)
Hệ thống tích hợp sẵn bộ kiểm tra tự động <!-- TODO(TH8): điền số test thật ở D6 --> bao phủ toàn bộ logic toán học, chuẩn hóa chuỗi thời gian, đồ thị chính sách và hợp đồng dữ liệu. Các bài kiểm tra sử dụng mô hình giả lập (`FakeEmbedder`, `FakeVerifier`) nên có thể chạy hoàn chỉnh trên CPU mà không cần tải trọng số nặng:
```bash
python -m pytest -q
```

### 6.3.3. Tiền tính toán véc-tơ nhúng (Precomputing Embeddings)
Lệnh trích xuất đặc trưng hỗ trợ chạy thử nghiệm khô (dry-run) để kiểm tra đường dẫn mà không nạp mô hình:
```bash
# Kiểm tra manifest và phân chia shard mà không tốn GPU
python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 2/4 --dry-run

# Chạy tính toán nhúng thật cho Shard 1 trên 4 shard
python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 1/4
```

### 6.3.4. Chạy kiểm chuẩn và trích xuất bảng số liệu
```bash
# Chạy kiểm chuẩn cấu hình A4c trên tập UCF-Crime test (290 video)
python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A4c \
    --calib-manifest manifests/ucf_calib_normal.csv --out-dir results
```

### 6.3.5. Kiểm chứng tầng 2 bằng Generative VLM
```bash
# Chạy kiểm chứng trên các candidate window (hỗ trợ phân chia shard i/N và tự động chia chunk <= 30s)
python scripts/run_verify.py --manifest manifests/ucf_test.csv --windows-dir results/windows \
    --out-dir results/verify --k 4 --verifier-model Qwen/Qwen2.5-VL-3B-Instruct

# Chạy thử nghiệm khô (dry-run) đếm số chunk-call mà không nạp model
python scripts/run_verify.py --manifest manifests/ucf_test.csv --windows-dir results/windows \
    --dry-run
```

### 6.3.6. Khởi chạy bảng điều khiển giám sát (Streamlit Dashboard)
```bash
# Khởi chạy giao diện trực quan ở chế độ Replay Mode
streamlit run app/streamlit_app.py
```
Giao diện sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.
