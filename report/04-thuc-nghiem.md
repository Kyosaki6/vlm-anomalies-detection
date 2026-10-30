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

Bảng dưới đây tổng hợp tiến trình cải thiện độ chính xác qua từng thành phần thiết kế. Các ô A0 - A3 là số liệu thật (chạy trên cache embedding, 290 video, 1.111.808 khung hình; `results/ablation_A*.json`). Các ô còn lại đánh dấu `TBD` chờ kết quả chạy.

<!-- TODO(TH8): Cập nhật số liệu A6 từ nhiệm vụ TH7 và A5 -->

| Mã | Biến thể thực nghiệm | Kiến trúc trích xuất | Kỹ thuật áp dụng | ROC-AUC (%) | 95% Bootstrap CI | Ghi chú kỹ thuật |
|---|---|---|---|---|---|---|
| **A0** | Baseline CLIP B/32 | CLIP ViT-B/32 | 2 prompt cơ bản, chấm điểm từng khung độc lập | 48.77 | [38.30, 59.40] | Cổng tái lập mốc 53.16% của LAVAD: mốc nằm trong CI nhưng thấp hơn dải chấp nhận 50.2 - 56.2 (xem 4.2.1) |
| **A1** | SigLIP 2 đơn khung | SigLIP 2 Base | 2 prompt cơ bản, chấm điểm từng khung độc lập | 65.87 | [57.93, 73.44] | So sánh trực tiếp backbone SigLIP 2 vs CLIP (+17.1 điểm so với A0) |
| **A2** | + Prompt Ensemble | SigLIP 2 Base | Tổ hợp 4 prompt bình thường + 4 prompt bất thường | 72.39 | [64.14, 80.39] | Giảm độ nhạy với cách diễn đạt câu lệnh (+6.5 điểm so với A1) |
| **A3** | + Chuẩn hóa video | SigLIP 2 Base | Thêm `robust_normalize` (median và MAD nội tại video) | 58.99 | [54.89, 63.24] | Micro AUC GIẢM 13.4 điểm so với A2 dù macro AUC giữ nguyên 64.12 (xem 4.2.2) |
| **A4** | + Xử lý thời gian | SigLIP 2 Base | Thêm gom cụm top-k và làm mịn EMA/Median | 58.41 | [53.51, 63.42] | Macro 63.35. Micro giảm 0.6 và macro giảm 0.8 so với A3 (58.99 / 64.12), cả hai nằm trong nhiễu. Top-k và EMA chưa cứu được micro AUC bị chuẩn hóa theo video làm mất (xem 4.2.2) |
| **A5** | + Bộ nhớ cảnh tĩnh | SigLIP 2 Base | Kết hợp điểm khoảng cách Normal Memory Bank | 57.08 | [52.14, 62.28] | Macro 58.87. Memory bank từ `calib_normal` (trọng số 0.5) KHÔNG cải thiện: micro -1.3, macro -4.5 so với A4. Báo cáo trung thực là kết quả âm |
| **A3c** | A3 chuẩn hóa theo `calib_normal` | SigLIP 2 Base | Như A3 nhưng median/MAD cố định tính trên 60 video `calib_normal` (gộp khung hình) | 72.39 | [64.14, 80.39] | Bằng đúng A2: phép biến đổi affine toàn cục là đơn điệu nên không đổi AUC. Giữ được chênh lệch giữa các video (xem 4.2.2) |
| **A4c** | A4 chuẩn hóa theo `calib_normal` | SigLIP 2 Base | Như A4 với thống kê calib cố định | 73.00 | [64.45, 81.15] | Macro 63.35. Top-k và EMA cho +0.6 điểm so với A3c, nằm trong nhiễu |
| **A5c** | A5 chuẩn hóa theo `calib_normal` | SigLIP 2 Base | Như A5; thống kê memory tính leave-one-video-out trên `calib_normal` | 64.95 | [55.19, 74.61] | Macro 57.87. Memory bank vẫn làm giảm (-8.1 điểm so với A4c): kết quả âm |
| **A6** | + Thẩm định VLM | SigLIP 2 + VLM | Tầng 1 (A4/A5) + Kiểm chứng Generative VLM Tầng 2 | TBD <!-- TODO(TH8): TH7, A5 --> | [TBD, TBD] <!-- TODO(TH8): TH7, A5 --> | Đánh giá đóng góp của VLM trên các cửa sổ nghi vấn |

### 4.2.1. Kết quả cổng kiểm chứng A0

A0 đạt 48.77% (CI 95% theo video [38.30, 59.40]), thấp hơn mốc 53.16% của LAVAD 4.4 điểm và nằm ngoài dải chấp nhận 50.2 - 56.2 đã đặt trước khi chạy. Mốc 53.16% vẫn nằm trong khoảng tin cậy. Các kiểm tra sai sót: đảo nhãn (loại trừ, một scorer đảo sẽ cho ~51), dịch nhãn ±1, ±16, +30 khung hình (AUC chỉ dao động 48.56 - 48.91, nên căn khung và quy ước biên đoạn là đúng), `logit_scale` 1/10/100 (cùng 48.77), pooling micro (macro 49.57), và prompt trùng văn bản của bài báo. Pha lấy mẫu đã kiểm tra trên cả 290 video (lấy mỗi 16 khung: 48.43 so với 48.77 ở 1 fps), không phải nguyên nhân. Chưa kiểm tra: tiền xử lý ảnh của CLIP HF so với pipeline của tác giả. Chi tiết: `notes/a0-gate.md`. Báo cáo ghi nhận đây là độ lệch so với bài báo gốc, không điều chỉnh dải chấp nhận sau khi có kết quả.

### 4.2.2. Ảnh hưởng của chuẩn hóa theo video (A2 so với A3)

Chuẩn hóa robust z-score theo từng video là phép đơn điệu trong mỗi video nên AUC theo từng video không đổi (macro 64.12 ở cả A2 và A3). Tuy nhiên micro AUC gộp khung hình giữa các video, và chuẩn hóa xóa mức chênh điểm giữa video bất thường và video bình thường, nên micro AUC giảm từ 72.39 xuống 58.99. Điều này cũng có nghĩa tỉ lệ cảnh báo sai trên giờ (FA/hour) khi dùng chuẩn hóa theo video phản ánh chính phép chuẩn hóa hơn là hệ thống, vì mọi video, kể cả video bình thường, bị ép về cùng phân phối.

**Quyết định (b): chuẩn hóa bằng thống kê cố định từ `calib_normal`.** Median và MAD của điểm được tính một lần trên toàn bộ khung hình của 60 video `calib_normal` (`PipelineConfig.norm_ref="calib"`, các biến thể A3c, A4c, A5c), rồi áp cho mọi video. Kết quả: A3c = 72.39, đúng bằng A2, như lý thuyết dự đoán (phép biến đổi affine chung cho mọi khung hình không đổi thứ hạng). Khác biệt là điểm sau chuẩn hóa có ý nghĩa tuyệt đối giữa các video, nên ngưỡng hiệu chỉnh trên `calib_normal` và FA/hour đo hệ thống thay vì đo phép chuẩn hóa. Với memory, mỗi video calib được chấm bằng bank dựng từ các video calib còn lại (leave-one-video-out), nếu không điểm memory của video nằm trong bank của chính nó bằng 0 và MAD bị đánh sập. Các biến thể chuẩn hóa theo video (A3, A4, A5) vẫn được giữ làm đối chứng.

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

Ngưỡng $\tau$ = 1.1686 chọn bằng phân vị 95% của điểm A4c gộp trên 60 video `calib_normal` (`scripts/calibrate_threshold.py`, `results/threshold_A4c.json`), FPR đạt được trên calib là 4.99%. Trên các khung hình bình thường của tập test, cùng ngưỡng cho FPR 10.57%, gấp đôi ngân sách: `calib_normal` lấy từ tập huấn luyện nên khác camera và cảnh so với tập test (lệch phân phối), không phải lỗi tính toán. Các số liệu dưới đây là của riêng tầng 1; tầng 2 chỉ có thể loại bớt cửa sổ nên cải thiện FA/hour chứ không tăng recall.

| Chỉ số vận hành | Ngưỡng hiệu chỉnh $\tau$ (tại FPR calib = 5%) | Giá trị đo được | Đơn vị tính | Nguồn đo kiểm |
|---|---|---|---|---|
| Báo động giả mỗi giờ (FA/hour) | 1.1686 (A4c, tier 1) | 20.64 (tier 1, chưa có VLM) <!-- TODO(TH8): cập nhật sau tầng 2 (TH7) --> | sự cố / giờ | Đo trên 150 video bình thường tập test |
| Độ trễ phát hiện trung bình (Mean Latency) | 1.1686 | 2.03 (chỉ tính 81 sự cố được phát hiện; trung vị 0.0) | giây | Đo trên 140 video bất thường tập test |
| Tỉ lệ phát hiện sự cố (Incident Recall) | 1.1686 | 51.9 (81/156, tier 1) | % | Cửa sổ cảnh báo giao với nhãn mặt đất |
| Tỉ lệ cửa sổ chuyển tiếp lên Tầng 2 | 1.1686 (307 cửa sổ, 1.06/video) | TBD <!-- TODO(TH8): A5 --> | % tổng số cửa sổ | Đo tải suy luận chuyển lên VLM |
| Thời gian xử lý trung bình mỗi cửa sổ VLM | - | TBD <!-- TODO(TH8): A1, A5 --> | giây / cửa sổ | Đo trên GPU RTX 3050 6GB (4-bit) |
