# Chương 2: Nghiên cứu liên quan

## 2.1. Mô hình nhúng thị giác ngôn ngữ (Dual-Encoder VLMs)

Các mô hình nhúng đa phương thức dạng hai nhánh (dual-encoder) ánh xạ hình ảnh và văn bản vào cùng một không gian vector đa chiều thông qua hàm mất mát tương phản.

- **CLIP (Contrastive Language-Image Pre-training)** [Radford et al., 2021]: CLIP sử dụng hàm mất mát InfoNCE softmax trên ma trận tương đồng theo lô (batch similarity matrix). Kiến trúc này chứng minh khả năng zero-shot mạnh mẽ trên nhiều tác vụ phân loại hình ảnh. Trong bài toán VAD trên video, biến thể CLIP ViT-B/32 thường được chọn làm đường cơ sở (baseline) tái hiện mức độ hiệu quả khi so khớp từng khung hình đơn lẻ.
- **SigLIP và SigLIP 2** [Zhai et al., 2023; Google, 2025 [CẦN KIỂM TRA]]: SigLIP thay thế hàm mất mát softmax bằng hàm mất mát sigmoid độc lập cho từng cặp ảnh văn bản. Cải tiến này giúp việc học biểu diễn ổn định hơn ở kích thước lô nhỏ và không đòi hỏi chuẩn hóa toàn bộ lô. Phiên bản SigLIP 2 nâng cao khả năng nắm bắt chi tiết không gian và hỗ trợ các độ phân giải linh hoạt. Trong đồ án này, mô hình `google/siglip2-base-patch16-384` được dùng làm bộ trích xuất đặc trưng chính cho Tầng 1 nhờ tốc độ suy luận nhanh và kích thước bộ nhớ phù hợp.

## 2.2. Phát hiện bất thường trong video (Video Anomaly Detection)

### Phương pháp giám sát yếu (Weakly-Supervised VAD)
Các phương pháp giám sát yếu chỉ sử dụng nhãn cấp video (video bình thường hoặc video có chứa bất thường) trong quá trình huấn luyện, không có mốc thời gian chính xác từng khung hình.
- Tiêu biểu là phương pháp **RTFM (Robust Temporal Feature Magnitude)** [Tian et al., ICCV 2021 [CẦN KIỂM TRA]]. RTFM tận dụng độ lớn vector đặc trưng để phân biệt phân đoạn bất thường và đạt ROC-AUC khoảng 84% đến 86% trên tập UCF-Crime.
- Mặc dù đạt độ chính xác cao, các phương pháp này yêu cầu tập huấn luyện lớn và không thể tổng quát hóa tốt khi gặp các dạng bất thường mới chưa từng xuất hiện.

### Phương pháp không cần huấn luyện và Zero-Shot (Zero-Shot VAD)
Xu hướng gần đây chuyển sang khai thác tri thức mở từ các mô hình nền tảng được huấn luyện trước trên dữ liệu web quy mô lớn.
- **LAVAD (Harnessing Large Language Models for Training-free Video Anomaly Detection, Zanella et al.)** [CVPR 2024]: Công trình công bố các mốc chuẩn ROC-AUC cấp khung hình (frame-level micro AUC) trên tập dữ liệu UCF-Crime:
  - ZS-CLIP (Zero-Shot CLIP ViT-B/32, so khớp từng khung hình với câu lệnh đơn): đạt 53.16% (gần tương đương phỏng đoán ngẫu nhiên).
  - ZS-ImageBind: đạt khoảng 53.65% đến 55.78%.
  - LLaVA-1.5: đạt 72.84%.
  - LAVAD (hệ thống kết hợp ngôn ngữ nâng cao): đạt 80.28%.
- Kết quả từ LAVAD khẳng định một luận điểm kỹ thuật quan trọng: bản thân mô hình nhúng khung hình khi áp dụng prompt thô không thể tự giải quyết bài toán VAD do thiếu thông tin chuỗi thời gian và bối cảnh chuyển động. Hệ thống cần các cơ chế kết hợp chuỗi thời gian và lọc kiểm chứng để nâng cao độ chính xác.

## 2.3. Mô hình thị giác ngôn ngữ sinh (Generative VLMs) trong kiểm chứng sự kiện

Các mô hình thị giác ngôn ngữ sinh lớn (như LLaVA, Qwen2-VL, SmolVLM2) có khả năng đọc chuỗi khung hình, hiểu ngữ cảnh câu hỏi phức tạp và sinh câu trả lời tự nhiên dưới dạng văn bản có cấu trúc (JSON).

Trong bài toán giám sát an ninh, việc đưa toàn bộ luồng video qua Generative VLM là bất khả thi vì chi phí suy luận lớn và độ trễ cao. Tuy nhiên, nếu chỉ sử dụng VLM ở Tầng 2 để thẩm định các đoạn video ngắn (candidate windows) đã được Tầng 1 lọc sẵn, hệ thống có thể:
1. Giảm thiểu các trường hợp dương tính giả (false alarms) do chuyển động bất thường vô hại gây ra (như người chạy tập thể dục hoặc đổi hướng đột ngột).
2. Tạo câu mô tả sự việc ngắn gọn (caption), hỗ trợ người trực ban hiểu ngay bản chất sự cố mà không cần xem lại toàn bộ băng ghi hình.
3. Trích xuất loại sự kiện (event type) trực tiếp hoặc hỗ trợ bước định danh sự kiện.

## 2.4. Điều phối cảnh báo và đồ thị chính sách (Alert Policy Graph)

Trong các trung tâm chỉ huy an ninh, việc phát hiện điểm số bất thường vượt ngưỡng mới chỉ là bước khởi đầu. Bài toán thực tế đòi hỏi chuyển đổi sự kiện thành hành động ứng phó phù hợp:
- Bảng luật tra cứu tĩnh (rule table) thường gặp hiện tượng bùng nổ số lượng luật khi kết hợp nhiều chiều thông tin (loại sự kiện, khu vực, thời điểm ban ngày hay ban đêm).
- Đồ thị chính sách (Policy Graph) biểu diễn các thực thể `Camera`, `Zone`, `EventType`, `Severity`, `Responder`, `Protocol` và mối quan hệ giữa chúng dưới dạng đồ thị có hướng. Cấu trúc này cho phép:
  - Tách biệt chính sách khỏi mã nguồn, dễ dàng cập nhật qua file cấu hình YAML.
  - Tự động điều chỉnh mức độ nghiêm trọng dựa trên khu vực nhạy cảm hoặc khung giờ đêm.
  - Xử lý leo thang khi nhiều sự cố xuất hiện đồng thời trong cùng cửa sổ thời gian (ví dụ tai nạn giao thông kèm cháy nổ).
  - Cung cấp đường dẫn giải thích minh bạch cho từng cảnh báo phát ra trên bảng điều khiển.
