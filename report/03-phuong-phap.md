# Chương 3: Phương pháp đề xuất

## 3.1. Tổng quan kiến trúc hai tầng (Two-Tier Architecture)

Hệ thống được thiết kế theo mô hình hai tầng nhằm tối ưu hóa giữa tốc độ xử lý luồng video và độ tin cậy của cảnh báo:

```
Video gốc ──> Trích xuất khung hình (1 - 2 fps)
                    │
                    ▼
          Trích xuất đặc trưng (SigLIP 2 / CLIP B/32) ──> Bộ nhớ đệm (Cache .npy fp16)
                    │
                    ▼
          TẦNG 1: Chấm điểm nhanh từng khung hình
          - Điểm văn bản (Generic Prompt Ensemble): Softmax(cos(ảnh, bất thường) vs cos(ảnh, bình thường))
          - Điểm bộ nhớ bình thường (Normal Memory): 1 - max cos(ảnh, memory bank)
          - Kết hợp điểm số
                    │
                    ▼
          Xử lý chuỗi thời gian (Temporal Operations)
          - Chuẩn hóa nội tại video (robust_normalize: median / MAD)
          - Gom cụm cửa sổ (top-k pooling)
          - Làm mịn chuỗi điểm (EMA + Median filter)
          - Trích xuất cửa sổ ứng viên (Candidate Windows: score > ngưỡng tau)
                    │
                    ▼
          TẦNG 2: Kiểm chứng chuyên sâu (Generative VLM Verifier)
          - Lấy k khung hình phân bố đều trong cửa sổ ứng viên
          - VLM phân tích ngữ cảnh, trả về JSON có cấu trúc
          - Xác nhận bất thường, sinh câu tóm tắt (caption) và loại sự kiện
                    │
                    ▼
          Định danh sự kiện & Đồ thị chính sách (Policy Graph)
          - Event Typer (Zero-shot 13 lớp trên vector nhúng trung bình của cửa sổ)
          - Alert Policy Graph: Zone + Khung giờ + Leo thang + Vệ sinh cảnh báo (Cooldown, Merge)
                    │
                    ▼
          Bảng điều khiển (Streamlit Dashboard Replay Mode)
```

## 3.2. Tầng 1: Trích xuất đặc trưng và tính điểm thô

### 3.2.1. Trích xuất đặc trưng và lưu đệm (Embedding & Atomic Cache)
- Bộ trích xuất khung hình (`vad/io/video.py`) duyệt tuần tự video bằng OpenCV grab theo chu kỳ cố định (mặc định 1 đến 2 fps), giúp tiết kiệm thời gian giải mã so với việc đọc toàn bộ từng frame vào bộ nhớ RAM.
- Bộ mã hóa (`vad/embed/siglip.py`) thiết kế hỗ trợ đa mô hình (model-generic), nạp được cả `google/siglip2-base-patch16-384` (độ dài padding 64) và `openai/clip-vit-base-patch32` (độ dài token 77).
- Cơ chế tiền xử lý cho phép trích xuất đặc trưng cho cả SigLIP 2 và CLIP ViT-B/32 chỉ qua một lượt giải mã video duy nhất.
- Vector đặc trưng được chuẩn hóa chuẩn L2 và lưu trữ nguyên tử (atomic write) dưới định dạng nhị phân `.npy` (nửa chính xác float16) tại thư mục `cache/embeddings/`. Khóa tệp được định danh duy nhất theo cú pháp: `{model_id}_{video_id}_{sample_fps}.npy`. Cơ chế này hỗ trợ dừng và chạy lại (resumable) mà không phải tính toán lặp lại.

### 3.2.2. Chấm điểm bằng Prompt khái quát (Generic-Prompt Scoring)
Để phát hiện bất thường mà không làm rò rỉ thông tin phân lớp, prompt phát hiện tuyệt đối không chứa tên của 13 loại hành vi trong tập kiểm thử. Điểm số văn bản được tính dựa trên độ tương đồng cosine giữa vector ảnh $v$ và các vector văn bản:
- Tập prompt bình thường ($T_{norm}$): gồm các mô tả khái quát như "a normal everyday scene", "a surveillance camera view of a peaceful scene".
- Tập prompt bất thường ($T_{abn}$): gồm các mô tả như "a scene with suspicious or potentially criminal activities", "a surveillance camera view of a dangerous or violent incident".
- Điểm dị biệt văn bản là xác suất softmax hai lớp giữa trung bình cosine của tập bất thường so với tập bình thường:
  $$s_{text}(t) = \frac{\exp(\cos(v_t, \bar{T}_{abn}) / \tau_{temp})}{\exp(\cos(v_t, \bar{T}_{abn}) / \tau_{temp}) + \exp(\cos(v_t, \bar{T}_{norm}) / \tau_{temp})}$$

### 3.2.3. Chấm điểm qua bộ nhớ cảnh bình thường (Normal Memory Bank)
Với các camera cố định, hệ thống duy trì một ngân hàng vector đặc trưng đại diện cho các khung hình bình thường được ghi nhận trước đó ($M = \{m_1, m_2, \dots, m_K\}$). Điểm dị biệt bộ nhớ được tính bằng khoảng cách cosine đến khung hình bình thường giống nhất:
$$s_{mem}(t) = 1 - \max_{j} \cos(v_t, m_j)$$
Điểm tổng hợp Tầng 1 là tổ hợp tuyến tính giữa điểm văn bản và điểm bộ nhớ:
$$s_{raw}(t) = \alpha \cdot s_{text}(t) + (1 - \alpha) \cdot s_{mem}(t)$$

## 3.3. Xử lý chuỗi thời gian (Temporal Operations)

Điểm thô $s_{raw}(t)$ thường chứa nhiều gai nhiễu do dao động ánh sáng tức thời hoặc chuyển động ngẫu nhiên của cảm biến. Module `vad/temporal/ops.py` thực hiện chuỗi biến đổi tuần tự:

1. **Chuẩn hóa nội tại theo video (`robust_normalize`):**
   Thay vì dùng giá trị cực đại min-max toàn cục dễ bị phá vỡ bởi các điểm ngoại lai (outliers), hệ thống chuẩn hóa bằng trung vị (median) và độ lệch tuyệt đối trung vị (MAD - Median Absolute Deviation):
   $$\hat{s}(t) = \frac{s_{raw}(t) - \text{median}(s)}{\text{MAD}(s) + \epsilon}$$
   Biến đổi này đưa phân phối điểm của các video có độ sáng và nền cảnh khác nhau về cùng một thước đo so sánh.
2. **Gom cụm theo cửa sổ thời gian (Top-k Window Pooling):**
   Trong một cửa sổ quan sát kích thước $W$ khung hình quanh thời điểm $t$, hệ thống lấy giá trị trung bình của $k$ giá trị cao nhất ($k \le W$). Kỹ thuật này giữ được tín hiệu bất thường kéo dài và triệt tiêu xung nhiễu xuất hiện ở đúng một khung hình duy nhất.
3. **Làm mịn chuỗi thời gian (Median Filter & EMA):**
   Áp dụng bộ lọc trung vị với kích thước hạt nhỏ để loại bỏ các điểm nhảy đột ngột, tiếp nối bằng lọc trung bình trượt hàm mũ (Exponential Moving Average - EMA) nhằm tạo đường cong điểm số liên tục, mượt mà.
4. **Trích xuất cửa sổ ứng viên (Candidate Windows):**
   Các đoạn thời gian liên tục có điểm số vượt ngưỡng $\tau$ (với thời lượng tối thiểu $L_{min}$) được đóng gói thành các cửa sổ ứng viên `[start_sec, end_sec]`. Các cửa sổ gần nhau trong khoảng cách cho phép được gộp lại để tránh phân mảnh cảnh báo.

## 3.4. Tầng 2: Kiểm chứng bằng Generative VLM

Các cửa sổ ứng viên từ Tầng 1 được chuyển tiếp đến Tầng 2 (`vad/verify/vlm.py`):
- Hệ thống trích xuất $k$ khung hình phân bố đều (thường $k = 4$) trong khoảng thời gian của cửa sổ nghi vấn.
- Các khung hình này được đưa vào mô hình VLM sinh ngôn ngữ (được lượng tử hóa 4-bit để vận hành trong ngưỡng 6GB VRAM, ví dụ dòng Qwen2-VL hoặc SmolVLM2).
- VLM nhận câu lệnh yêu cầu phân tích khách quan và bắt buộc trả về kết quả định dạng JSON duy nhất:
  ```json
  {
    "is_anomaly": true,
    "event_type": "Fighting",
    "caption": "Hai nguoi dan ong dang xo xat tren via he gan xe may.",
    "confidence": 0.88
  }
  ```
- Bộ phân tích (`parse_verifier_json`) kiểm tra tính hợp lệ của cấu trúc JSON. Nếu xảy ra lỗi phân tích cú pháp, hệ thống cho phép thử lại đúng một lần. Nếu tiếp tục thất bại, hệ thống báo lỗi rõ ràng chứ tuyệt đối không giả lập dữ liệu trả về.

## 3.5. Định danh sự kiện (Event Typing)

Công đoạn định danh sự kiện (`vad/score/typing.py`) được tách biệt hoàn toàn khỏi bước phát hiện bất thường:
- Thực hiện phân loại zero-shot 13 lớp hành vi phạm tội của UCF-Crime (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism).
- Lấy vector đặc trưng trung bình của toàn bộ các khung hình trong cửa sổ đã được xác nhận bất thường, sau đó tính khoảng cách cosine với 13 câu mô tả mẫu trong `configs/prompts.yaml` (ví dụ: "a surveillance video frame showing people fighting").
- Nhãn có độ tương đồng lớn nhất được gán làm loại sự kiện cho cửa sổ. Độ chính xác của bước này được đánh giá độc lập dưới chỉ số Top-1 Accuracy so với mức ngẫu nhiên $1/13 \approx 7.69\%$.

## 3.6. Đồ thị chính sách cảnh báo (Alert Policy Graph)

Đồ thị chính sách (`vad/graph/policy.py`) được xây dựng trên thư viện NetworkX từ tệp cấu hình trung tâm `configs/policy.yaml`. Cấu trúc đồ thị bao gồm các loại nút định kiểu:
- `Camera`: Định danh vị trí lắp đặt và góc quan sát.
- `Zone`: Khu vực không gian (`school_gate`, `market`, `highway`, `parking_lot`, `default`).
- `EventType`: 13 loại sự kiện bất thường và sự kiện không xác định (`Unknown`).
- `Severity`: Mức độ nghiêm trọng định thứ tự (`low < medium < high < critical`).
- `Responder`: Lực lượng ứng cứu (`security`, `police`, `ambulance`, `fire_dept`, `traffic_police`).
- `Protocol`: Quy trình xử lý tương ứng (`P-VIOLENCE`, `P-ACCIDENT`, `P-FIRE`, `P-THEFT`...).

Các cơ chế vận hành chính của đồ thị:
1. **Biến điệu mức độ nghiêm trọng theo ngữ cảnh (Contextual Severity Shift):**
   Mức độ nghiêm trọng cơ bản của sự kiện được tăng hoặc giảm dựa trên thuộc tính khu vực và khung giờ. Ví dụ sự kiện `Fighting` vốn có mức độ `medium`, nhưng khi diễn ra tại `school_gate` (độ dịch +1) hoặc vào ban đêm `night` (độ dịch +1) sẽ tự động leo thang lên mức `critical`.
2. **Quy tắc leo thang đa sự kiện (Multi-Event Escalation):**
   Nếu trong cùng một cửa sổ thời gian hoặc hai cửa sổ chồng lấn xuất hiện đồng thời hai sự kiện có quan hệ tương tác đặc biệt (ví dụ `RoadAccidents` và `Explosion`), đồ thị tự động kích hoạt luật leo thang: tăng một bậc nghiêm trọng và bổ sung thêm lực lượng cứu hỏa (`fire_dept`).
3. **Vệ sinh cảnh báo (Alert Hygiene & Cooldown):**
   Để tránh làm quá tải nhân viên điều hành, hệ thống áp dụng cơ chế đóng băng cảnh báo (`cooldown_s = 30.0` giây) cho cùng một loại sự kiện trên cùng một camera. Các cảnh báo xuất hiện cách nhau dưới khoảng trống ngắn (`merge_gap_s = 5.0` giây) được gom thành một sự cố duy nhất.
4. **Đường dẫn giải thích (Explainability Path):**
   Mỗi cảnh báo tạo ra trên giao diện đều mang theo chuỗi giải thích rõ ràng về các mắt xích đã duyệt qua trên đồ thị, ví dụ:
   `Fighting @ school_gate, night -> critical -> responders: [security, police] -> protocol: P-VIOLENCE`.

## 3.7. Các nguyên tắc thiết kế chống sai lệch (Anti-Patterns)

Để bảo đảm tính trung thực khoa học, hệ thống tuân thủ nghiêm ngặt 5 nguyên tắc chống phản mẫu (Anti-Patterns) được quy định trong tài liệu dự án:

1. **Tuyệt đối không đưa tên lớp vào prompt phát hiện:**
   Tên các hành vi như "fighting", "accident" chỉ được dùng ở giai đoạn Event Typing. Việc đưa tên lớp vào bước phát hiện nhị phân (bình thường vs bất thường) sẽ gây hiện tượng rò rỉ nhãn (label leakage), làm sai lệch bản chất zero-shot tổng quát.
2. **Không căn chỉnh ngưỡng trên tập kiểm thử:**
   Mọi ngưỡng kích hoạt $\tau$ đều được xác định trên tập chuẩn hóa bình thường (`split=calib_normal`), tuyệt đối không tinh chỉnh trên tập kiểm thử chính thức (UCF-Crime test split).
3. **Khoảng tin cậy Bootstrap lấy mẫu theo đơn vị video:**
   Vì các khung hình trong cùng một video có tính tự tương quan thời gian rất cao (autocorrelated), việc lấy mẫu bootstrap ngẫu nhiên theo khung hình sẽ cho khoảng tin cậy sai lệch và hẹp giả tạo. Khoảng tin cậy 95% CI trong báo cáo này bắt buộc lấy mẫu hoàn lại trên tập các video.
4. **Không định danh Zero-Shot cho tập ShanghaiTech:**
   Do các bất thường của ShanghaiTech (đi xe đạp, chạy bộ, ném đồ) mang tính phụ thuộc mạnh vào cảnh nền khuôn viên trường học thay vì ngữ nghĩa tội phạm phổ quát, nếu thực nghiệm trên tập này, hệ thống phải được ghi nhãn trung thực là "training-free kết hợp khung hình tham chiếu bình thường", không gọi là zero-shot.
5. **Giữ nguyên trạng thái Stub khi chưa hoàn thiện:**
   Các module chưa nối mô hình thật (như backend của `VLMVerifier`) phải kích hoạt ngoại lệ `NotImplementedError`, tuyệt đối không sinh dữ liệu giả ngẫu nhiên nhằm làm đẹp báo cáo.
