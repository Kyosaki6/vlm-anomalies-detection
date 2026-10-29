# Chương 1: Giới thiệu

## 1.1. Đặt vấn đề

Phát hiện sự kiện bất thường trong video giám sát (Video Anomaly Detection - VAD) là bài toán then chốt trong an ninh đô thị, giao thông và quản lý trật tự công cộng. Các camera giám sát tạo ra hàng nghìn giờ video mỗi ngày. Nhân sự an ninh không thể quan sát toàn bộ màn hình liên tục. Hậu quả là nhiều sự cố nghiêm trọng chỉ được phát hiện sau khi thiệt hại đã xảy ra.

Các phương pháp VAD truyền thống chủ yếu dựa trên học có giám sát (supervised) hoặc bán giám sát yếu (weakly-supervised). Những kỹ thuật này đòi hỏi tập dữ liệu gán nhãn chi tiết ở cấp độ video hoặc khung hình. Việc thu thập và dán nhãn cho các hành vi phạm tội (fighting, robbery, arson...) trong dữ liệu camera rất tốn kém và khó bao quát hết mọi tình huống thực tế. Thêm vào đó, mô hình huấn luyện sẵn trên một bối cảnh thường giảm sút độ chính xác khi đưa sang vị trí lắp đặt mới do khác biệt về góc quay, ánh sáng và mật độ người qua lại.

Sự xuất hiện của các mô hình nền tảng thị giác ngôn ngữ (Vision-Language Models - VLM) và mô hình nhúng đa phương thức (CLIP, SigLIP 2) mở ra cơ hội giải quyết bài toán VAD theo hướng không cần huấn luyện bổ sung (training-free) hoặc zero-shot. Tuy nhiên, nếu chỉ so khớp trực tiếp từng khung hình độc lập với câu lệnh văn bản đơn giản, độ chính xác thường ở mức ngẫu nhiên do thiếu ngữ cảnh thời gian và độ nhạy cao với nhiễu nền. Hơn nữa, việc chỉ xuất ra một điểm số bất thường không đủ giúp lực lượng vận hành phản ứng kịp thời. Một hệ thống thực tế cần diễn giải được sự kiện gì đang xảy ra và hành động cụ thể cần thực hiện là gì.

## 1.2. Mục tiêu đề tài

Đề tài hướng tới xây dựng một hệ thống phát hiện bất thường và điều phối cảnh báo an ninh từ video camera theo cơ chế không cần huấn luyện (training-free / zero-shot). Các mục tiêu cụ thể gồm:

1. Xây dựng quy trình phát hiện hai tầng (Two-Tier Architecture):
   - Tầng 1 (Tier-1): Sử dụng mô hình nhúng SigLIP 2 kết hợp tập prompt chung (generic prompts), bộ nhớ khung hình bình thường (normal memory) và các phép xử lý chuỗi thời gian để lọc nhanh các cửa sổ nghi vấn với chi phí tính toán thấp.
   - Tầng 2 (Tier-2): Đưa các cửa sổ vượt ngưỡng vào mô hình VLM sinh ngôn ngữ (Generative VLM) để xác nhận lại, tạo câu mô tả sự việc và phân loại sự kiện.
2. Xây dựng đồ thị chính sách cảnh báo (Alert Policy Graph): Chuyển đổi loại sự kiện đã được kiểm chứng cùng ngữ cảnh (khu vực camera, khung giờ) thành cảnh báo cụ thể với mức độ nghiêm trọng, lực lượng xử lý tương ứng và quy trình xử lý tiêu chuẩn.
3. Trực quan hóa kết quả: Xây dựng giao diện bảng điều khiển (Dashboard) cho phép xem lại video kèm dòng thời gian điểm số bất thường, danh sách cảnh báo, câu mô tả VLM và đường dẫn giải thích quyết định.
4. Đánh giá nghiêm ngặt: Kiểm nghiệm hệ thống trên tập dữ liệu chuẩn UCF-Crime, so sánh với các nghiên cứu công bố (LAVAD, ZS-CLIP) bằng phương pháp thống kê tin cậy (bootstrap theo video).

## 1.3. Đóng góp của đề án

Đề án tập trung vào thiết kế hệ thống tổng thể nhằm thu hẹp khoảng cách giữa mô hình nhúng thị giác và yêu cầu vận hành thực tế:

- Kiến trúc hai tầng kết hợp hiệu năng và độ chính xác: Lọc thô bằng mô hình nhúng nhẹ và tinh lọc bằng VLM sinh ngôn ngữ giúp hệ thống kiểm soát được tải tính toán trên phần cứng giới hạn (GPU 6GB hoặc môi trường đám mây miễn phí).
- Xử lý chuỗi thời gian ổn định: Áp dụng chuẩn hóa theo video (median/MAD), trượt cửa sổ thời gian (top-k pooling) và làm mịn (EMA, median filter) giúp giảm báo động giả từ các biến động nhất thời về ánh sáng.
- Tách bạch giữa phát hiện và phân loại: Tuân thủ quy tắc prompt phát hiện không chứa tên lớp bất thường, tránh hiện tượng rò rỉ nhãn (label leakage). Giai đoạn phân loại sự kiện (Event Typing) được đo lường riêng biệt.
- Tích hợp đồ thị chính sách hỗ trợ ra quyết định: Thay vì chỉ đưa ra điểm số hoặc bảng tra cứng, Alert Policy Graph hỗ trợ điều chỉnh mức độ nghiêm trọng theo khu vực, thời gian, xử lý leo thang khi nhiều sự kiện đồng xuất hiện và cung cấp đường dẫn giải thích rõ ràng.

## 1.4. Phạm vi và các cắt giảm kỹ thuật

Do hạn chế về thời gian thực hiện (kế hoạch thực hiện rút ngắn còn 6 ngày) và phần cứng nhóm nghiên cứu (1 GPU RTX 3050 6GB, 1 GPU RTX 3050 4GB, 1 GPU GTX 1060 3GB và môi trường CPU / Colab T4), phạm vi đồ án được xác định rõ ràng với các giới hạn kỹ thuật sau:

1. Tập dữ liệu chuẩn:
   - Tập kiểm thử chính và bắt buộc là UCF-Crime test split gồm 290 video (140 video bất thường và 150 video bình thường).
   - Tập ShanghaiTech Campus được xếp vào diện mục tiêu mở rộng (stretch goal). Nếu không kịp thời gian tính toán nhúng và đánh giá, nhóm sẽ báo cáo trung thực giới hạn này mà không giả lập kết quả.
2. Quy mô kiểm chứng Tầng 2 (Tier-2 VLM):
   - Mô hình VLM sinh ngôn ngữ được chạy trên các cửa sổ ứng viên nghi vấn. Trong trường hợp giới hạn thời gian tính toán của GPU 6GB, Tier-2 được phép chạy trên một tập con đại diện có ghi nhận chi phí thời gian và bộ nhớ thực tế, tuyệt đối không cắt bỏ hoàn toàn vì đây là ràng buộc cốt lõi của đề tài đăng ký.
3. Dữ liệu thử nghiệm thực tế:
   - Thu thập khoảng 5 clip đường phố Việt Nam phục vụ trình diễn (demo), trong đó có clip bình thường và clip dàn dựng có sự đồng ý. Các clip này không dùng để báo cáo điểm chuẩn benchmark.
4. Bảng điều khiển:
   - Hoạt động ở chế độ xem lại (replay mode) từ các tệp kết quả JSON đã tính trước (precomputed results), bảo đảm khả năng trình chiếu mượt mà trên máy tính không có GPU.
5. Biến thể mô hình (Ablations):
   - Tập trung vào các biến thể A0 đến A5 và biến thể sau VLM (A6). Hủy bỏ biến thể A7 (backbone SigLIP nhỏ 256) để dành tài nguyên cho chuỗi thực nghiệm chính.
