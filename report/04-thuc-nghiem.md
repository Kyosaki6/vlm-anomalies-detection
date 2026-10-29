# Chương 4: Thực nghiệm và Đánh giá

## 4.1. Thiết lập thực nghiệm

### 4.1.1. Môi trường phần cứng và phân chia tính toán
Hệ thống được phát triển và kiểm thử trên các cấu hình phần cứng phân tán của các thành viên trong nhóm:
- Máy 1 (Phước): GPU NVIDIA GeForce RTX 3050 6GB VRAM (nửa chính xác float16), đảm nhiệm chạy mô hình Generative VLM (Tầng 2) và tính toán Shard 1 của bộ nhớ đệm nhúng.
- Máy 2 (Huy): GPU NVIDIA GeForce RTX 3050 4GB VRAM (float16), đảm nhiệm Shard 2 của bộ nhớ đệm nhúng.
- Máy 3 (Nam): GPU NVIDIA GeForce GTX 1060 3GB VRAM (độ chính xác đơn float32 do kiến trúc Pascal xử lý float16 chậm), đảm nhiệm Shard 3.
- Máy 4 (Thịnh): CPU nội bộ kèm môi trường đám mây Google Colab T4 16GB VRAM (dùng làm nút dự phòng và kiểm chứng chéo).

Để bảo đảm tính nhất quán giữa các máy, cơ chế chia shard sử dụng hàm băm SHA-256 trên tên video (`vad/datasets/sharding.py`). Mọi máy trạm đều dùng chung một bộ trọng số mô hình, cùng tần số trích mẫu khung hình và cùng định dạng chuẩn hóa vector.

### 4.1.2. Tập dữ liệu thực nghiệm

1. **UCF-Crime Test Split (Tập đánh giá chuẩn chính):**
   - Gồm 290 video thực tế từ camera an ninh với tổng thời lượng hàng chục giờ, độ phân giải đa dạng.
   - 140 video chứa hành vi bất thường thuộc 13 nhóm tội phạm (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism).
   - 150 video sinh hoạt bình thường.
   - Nhãn mặt đất (Ground Truth) được biểu diễn dưới dạng các đoạn khung hình bất thường `[start_frame, end_frame]` (chuẩn hóa ở tốc độ 30 fps) theo tệp chú thích chính thức của UCF-Crime.
2. **UCF-Crime Calib Normal (Tập hiệu chỉnh ngưỡng):**
   - Gồm khoảng 60 video bình thường lấy từ tập huấn luyện (train normal).
   - Được dùng riêng biệt để tính toán phân phối điểm nền và xác định ngưỡng báo động $\tau$ ứng với mức ngân sách báo động giả mục tiêu (ví dụ FPR = 5%, mức này là giả định chưa chốt). Tuyệt đối không chạm vào tập kiểm thử để chọn ngưỡng.
3. **ShanghaiTech Campus (Mục tiêu mở rộng - Stretch Goal):**
   - Gồm 107 video kiểm thử tại 13 góc quay khuôn viên trường đại học.
   - Nếu điều kiện thời gian cho phép, tập này được đánh giá theo quy ước "training-free kết hợp khung hình tham chiếu bình thường". Nếu thiếu thời gian, báo cáo ghi rõ giới hạn kỹ thuật này.
4. **Tập clip đường phố Việt Nam (Minh họa demo):**
   - Khoảng 5 clip ngắn (10 đến 30 giây) quay tại bối cảnh đường phố trong nước, gán nhãn thủ công mốc thời gian, dùng cho kiểm thử trực quan trên bảng điều khiển.

### 4.1.3. Giao thức đánh giá (Evaluation Protocol)

- **Frame-level Micro ROC-AUC:** Điểm bất thường sau khi nội suy về tốc độ khung hình gốc (`expand_to_frames`) được nối liền thành một chuỗi nhị phân liên tục trên toàn bộ 290 video kiểm thử. Diện tích dưới đường cong ROC (ROC-AUC) được tính trên toàn bộ chuỗi khung hình này theo đúng giao thức của LAVAD (`notes/lavad-protocol.md`: gộp mọi khung hình của mọi video rồi tính một AUC, không lấy trung bình theo video). Khác biệt cần lưu ý: LAVAD lấy mẫu mỗi 16 khung rồi lặp lại điểm, còn hệ thống này lấy mẫu 1 fps rồi dùng `expand_to_frames`.
- **Khoảng tin cậy Bootstrap 95% theo Video:** Để phản ánh đúng sự biến động giữa các cảnh quay khác nhau, hệ thống thực hiện lấy mẫu lặp lại có hoàn lại 1000 lần trên cấp độ video (resampling over videos), sau đó tính khoảng tin cậy 95% CI cho ROC-AUC.
- **Điểm chuẩn cổng kiểm chứng A0 (Validity Gate):** Chạy mô hình OpenAI CLIP ViT-B/32 với hai prompt cơ sở. Kết quả phải đạt xấp xỉ mốc 53.2% (+/- 3%) được công bố bởi LAVAD để chứng minh quy trình nạp nhãn và tính toán toán học không mắc lỗi lệch khung hình (off-by-one).

## 4.2. Kết quả thực nghiệm và Phân tích thành phần (Ablation Study)

Bảng dưới đây tổng hợp tiến trình cải thiện độ chính xác qua từng thành phần thiết kế. Mọi ô số liệu đang chờ kết quả chạy thực nghiệm từ các máy trạm và được đánh dấu `TBD`.

<!-- TODO(TH8): Cập nhật số liệu A0 từ nhiệm vụ TH4 -->
<!-- TODO(TH8): Cập nhật số liệu A1 đến A5 từ nhiệm vụ TH5 -->
<!-- TODO(TH8): Cập nhật số liệu A6 từ nhiệm vụ TH7 và A5 -->

| Mã | Biến thể thực nghiệm | Kiến trúc trích xuất | Kỹ thuật áp dụng | ROC-AUC (%) | 95% Bootstrap CI | Ghi chú kỹ thuật |
|---|---|---|---|---|---|---|
| **A0** | Baseline CLIP B/32 | CLIP ViT-B/32 | 2 prompt cơ bản, chấm điểm từng khung độc lập | TBD <!-- TODO(TH8): TH4 --> | [TBD, TBD] <!-- TODO(TH8): TH4 --> | Cổng tái lập mốc ~53.2% của LAVAD |
| **A1** | SigLIP 2 đơn khung | SigLIP 2 Base | 2 prompt cơ bản, chấm điểm từng khung độc lập | TBD <!-- TODO(TH8): TH5 --> | [TBD, TBD] <!-- TODO(TH8): TH5 --> | So sánh trực tiếp backbone SigLIP 2 vs CLIP |
| **A2** | + Prompt Ensemble | SigLIP 2 Base | Tổ hợp 4 prompt bình thường + 4 prompt bất thường | TBD <!-- TODO(TH8): TH5 --> | [TBD, TBD] <!-- TODO(TH8): TH5 --> | Giảm độ nhạy với cách diễn đạt câu lệnh |
| **A3** | + Chuẩn hóa video | SigLIP 2 Base | Thêm `robust_normalize` (median và MAD nội tại video) | TBD <!-- TODO(TH8): TH5 --> | [TBD, TBD] <!-- TODO(TH8): TH5 --> | Cân bằng mức điểm giữa các góc quay camera |
| **A4** | + Xử lý thời gian | SigLIP 2 Base | Thêm gom cụm top-k và làm mịn EMA/Median | TBD <!-- TODO(TH8): TH5 --> | [TBD, TBD] <!-- TODO(TH8): TH5 --> | Triệt tiêu gai nhiễu tức thời của khung đơn |
| **A5** | + Bộ nhớ cảnh tĩnh | SigLIP 2 Base | Kết hợp điểm khoảng cách Normal Memory Bank | TBD <!-- TODO(TH8): TH5 --> | [TBD, TBD] <!-- TODO(TH8): TH5 --> | Tận dụng đặc trưng các khung hình bình thường |
| **A6** | + Thẩm định VLM | SigLIP 2 + VLM | Tầng 1 (A4/A5) + Kiểm chứng Generative VLM Tầng 2 | TBD <!-- TODO(TH8): TH7, A5 --> | [TBD, TBD] <!-- TODO(TH8): TH7, A5 --> | Đánh giá đóng góp của VLM trên các cửa sổ nghi vấn |

## 4.3. Đánh giá phân loại sự kiện (Event Typing)

Khác với tác vụ phát hiện nhị phân, bộ phân loại sự kiện zero-shot 13 lớp (`vad/score/typing.py`) được đo lường riêng trên các đoạn video đã xác định có bất thường theo nhãn mặt đất.

<!-- TODO(TH8): Cập nhật số liệu Event Typing từ nhiệm vụ C3 -->

| Phương pháp định danh | Số lớp | Mức ngẫu nhiên (Chance Level) | Top-1 Accuracy (%) | Ghi chú |
|---|---|---|---|---|
| Zero-shot Text Typer (Cosine Mean Window) | 13 lớp | 7.69% (1/13) | TBD <!-- TODO(TH8): C3 --> | So khớp vector nhúng trung bình cửa sổ với 13 prompt |
| Generative VLM Typing (JSON output) | 13 lớp | 7.69% (1/13) | TBD <!-- TODO(TH8): A5 --> | Dự đoán trực tiếp từ trường `event_type` của VLM |

## 4.4. Các chỉ số vận hành thực tế (Operational Metrics)

Để đánh giá mức độ khả thi khi triển khai hệ thống vào trung tâm điều hành camera thực tế, hệ thống đo lường các chỉ số vận hành tại ngưỡng $\tau$ được hiệu chỉnh trên tập `calib_normal` (ngưỡng tương ứng ngân sách FPR = 5%):

1. **Báo động giả mỗi giờ (False Alarms per Hour - FA/hour):**
   Số lượng cảnh báo sai phát sinh trên mỗi giờ video hoàn toàn bình thường từ tập kiểm thử:
   $$\text{FA/hour} = \frac{\text{Tổng số lần phát cảnh báo sai trên video bình thường}}{\text{Tổng thời lượng video bình thường (giờ)}}$$
2. **Độ trễ phát hiện (Detection Latency):**
   Khoảng thời gian tính bằng giây từ thời điểm bắt đầu xảy ra sự kiện bất thường theo nhãn mặt đất ($t_{GT\_start}$) đến thời điểm hệ thống phát ra cảnh báo đầu tiên ($t_{alert}$):
   $$\text{Latency} = \max(0, t_{alert} - t_{GT\_start})$$
3. **Tỉ lệ bao phủ sự cố (Incident-level Recall):**
   Tỉ lệ phần trăm các sự kiện thực tế có ít nhất một cảnh báo chính xác rơi vào khoảng thời gian xảy ra sự cố.

<!-- TODO(TH8): Cập nhật số liệu vận hành từ nhiệm vụ TH7 -->

| Chỉ số vận hành | Ngưỡng hiệu chỉnh $\tau$ (tại FPR calib = 5%) | Giá trị đo được | Đơn vị tính | Nguồn đo kiểm |
|---|---|---|---|---|
| Báo động giả mỗi giờ (FA/hour) | TBD <!-- TODO(TH8): TH5 --> | TBD <!-- TODO(TH8): TH7 --> | sự cố / giờ | Đo trên 150 video bình thường tập test |
| Độ trễ phát hiện trung bình (Mean Latency) | TBD <!-- TODO(TH8): TH5 --> | TBD <!-- TODO(TH8): TH7 --> | giây | Đo trên 140 video bất thường tập test |
| Tỉ lệ phát hiện sự cố (Incident Recall) | TBD <!-- TODO(TH8): TH5 --> | TBD <!-- TODO(TH8): TH7 --> | % | Cửa sổ cảnh báo giao với nhãn mặt đất |
| Tỉ lệ cửa sổ chuyển tiếp lên Tầng 2 | TBD <!-- TODO(TH8): TH5 --> | TBD <!-- TODO(TH8): A5 --> | % tổng số cửa sổ | Đo tải suy luận chuyển lên VLM |
| Thời gian xử lý trung bình mỗi cửa sổ VLM | - | TBD <!-- TODO(TH8): A1, A5 --> | giây / cửa sổ | Đo trên GPU RTX 3050 6GB (4-bit) |
