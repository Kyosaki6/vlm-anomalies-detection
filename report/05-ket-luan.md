# Chương 5: Kết luận và Hướng phát triển

## 5.1. Kết luận

Đề tài đã xây dựng và hoàn thiện khung hệ thống phát hiện bất thường và điều phối cảnh báo an ninh từ video giám sát theo cơ chế không cần huấn luyện (training-free / zero-shot). Hệ thống giải quyết các thách thức cố hữu của việc ứng dụng mô hình ngôn ngữ thị giác vào giám sát thực tế thông qua các giải pháp kỹ thuật cụ thể:

1. **Kiến trúc hai tầng cân bằng hiệu năng và độ sâu ngữ nghĩa:** Tầng 1 sử dụng mô hình nhúng SigLIP 2 kết hợp prompt tổng quát và bộ nhớ cảnh tĩnh để chấm điểm nhanh từng khung hình, sau đó các phép lọc chuỗi thời gian (robust_normalize, top-k pooling, EMA, median filter) trích xuất các cửa sổ ứng viên nghi vấn. Tầng 2 chỉ kích hoạt mô hình Generative VLM trên các cửa sổ này để kiểm chứng, giảm tải tối đa cho phần cứng giới hạn.
2. **Loại bỏ rò rỉ nhãn và chuẩn hóa thực nghiệm:** Hệ thống tuân thủ nguyên tắc không sử dụng tên loại tội phạm trong prompt phát hiện, thực hiện hiệu chỉnh ngưỡng hoàn toàn trên tập video bình thường và tính khoảng tin cậy thống kê bằng phương pháp bootstrap trên cấp độ video.
3. **Chuyển đổi từ điểm số sang hành động điều phối:** Alert Policy Graph ánh xạ loại sự kiện cùng ngữ cảnh không gian và thời gian sang các lực lượng ứng cứu và quy trình xử lý cụ thể, đồng thời cung cấp đường dẫn giải thích minh bạch cho người vận hành.
4. **Trực quan hóa qua bảng điều khiển:** Giao diện Streamlit cho phép xem lại video kèm dòng thời gian điểm số, danh sách cảnh báo thời gian thực và mô tả chi tiết từ mô hình VLM.

## 5.2. Hạn chế của hệ thống

Bên cạnh những kết quả đạt được, đồ án tồn tại một số hạn chế kỹ thuật khách quan:

1. **Thiếu véc-tơ chuyển động chuyên biệt:** Việc trích xuất đặc trưng dựa trên từng khung hình 2D độc lập (kể cả khi đã làm mịn theo chuỗi thời gian) khiến hệ thống khó phân biệt các hành vi có bề ngoài tĩnh tương đồng nhưng khác biệt về vận tốc chuyển động (ví dụ chạy bộ bình thường so với bỏ chạy khỏi hiện trường).
2. **Chi phí tính toán của Generative VLM:** Mặc dù đã giới hạn VLM ở Tầng 2, tốc độ suy luận của mô hình ngôn ngữ thị giác sinh trên GPU cá nhân (RTX 3050 6GB) vẫn ở mức vài giây cho mỗi cửa sổ ứng viên, chưa thể đáp ứng tức thì cho hàng chục luồng camera cùng lúc mà không có cụm máy chủ chuyên dụng.
3. **Phạm vi kiểm thử theo giới hạn thời gian:** Do kế hoạch rút ngắn còn 6 ngày, tập dữ liệu ShanghaiTech được đưa vào diện mở rộng và chưa thể hoàn tất việc giải mã và tính nhúng toàn diện; hệ thống tập trung toàn lực vào tập chuẩn UCF-Crime test split (290 video).
4. **Tầng 2 giảm báo động giả nhưng làm mất sự cố thật:** trên UCF-Crime test, VLM đưa FA/hour từ 20.64 xuống 0.17 và precision cửa sổ từ 28.0% lên 49.6%, nhưng 22 cửa sổ giao nhãn mặt đất bị loại, incident recall giảm từ 51.9% xuống 39.7% và AUC micro giảm từ 73.00 xuống 67.90 (hiệu cặp -5.10, CI 95% [-11.42, -0.25]). Kiểm tay 20 cửa sổ cho precision cửa sổ được xác nhận 6/10 (CI rộng) và chưa kiểm tra được bỏ sót. Luật gộp OR trên chunk làm cửa sổ dài dễ bị xác nhận thừa. Nhãn cảnh báo do bộ định danh zero-shot quyết định vì tính tất định, dù trên mẫu 62 cửa sổ VLM gọi tên đúng nhiều hơn (83.9% so với 24.2%).
5. **Vận hành ở chế độ xem lại (Replay Mode):** Để bảo đảm tính ổn định khi trình diễn trên các thiết bị không có GPU, bảng điều khiển hiện đọc dữ liệu từ các tệp kết quả JSON tính sẵn thay vì kết nối trực tiếp với luồng camera RTSP thời gian thực.

## 5.3. Hướng phát triển tiếp theo

1. **Tích hợp luồng nạp video thời gian thực:** Phát triển module tiếp nhận luồng RTSP và hàng đợi xử lý khung hình bất đồng bộ (asynchronous queue), cho phép hệ thống triển khai trực tiếp tại các mắt camera giao thông.
2. **Khoanh vùng không gian bất thường (Spatial Grounding):** Bổ sung khả năng định vị tọa độ hộp bao (bounding box) cho đối tượng hoặc hành vi gây bất thường trong khung hình, giúp lực lượng an ninh nhận biết ngay vị trí sự cố trong khung cảnh đông đúc.
3. **Cơ chế tự thích ứng ngưỡng theo camera:** Xây dựng module tự động học phân phối bình thường của từng camera theo thời gian thực (online background learning), giúp tự động điều chỉnh ngưỡng cảnh báo phù hợp với sự thay đổi thời tiết, ngày và đêm mà không cần can thiệp thủ công.
4. **Tối ưu hóa suy luận mô hình:** Lượng tử hóa mô hình xuống mức INT4 hoặc chuyển đổi sang định dạng TensorRT / ONNX Runtime để tăng tốc độ xử lý của cả SigLIP 2 và Generative VLM trên các thiết bị tính toán biên (Edge AI).
