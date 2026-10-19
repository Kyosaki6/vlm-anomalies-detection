# EVAL top-3 — google/siglip2-base-patch16-384

## 1. Thông tin lần chạy

- Ngày chạy: **2026-10-19 14:17** · máy: `DESKTOP-AVDV3SH`
- Model: `google/siglip2-base-patch16-384`
- Thiết bị: `cpu`
- Gallery: `data` — 15 ảnh, mỗi lần đo còn 14 ảnh (đã bỏ ảnh query)

## 2. Cách đo

**Leave-one-out.** Với mỗi ảnh trong `data/`, tạo một thư mục gallery tạm chỉ chứa **14 ảnh còn lại** — loại chính ảnh query ra khỏi gallery — rồi chạy `search(..., topk=3)` để lấy 3 kết quả giống nhất.

**Tại sao không self-match (không tự-so-với-chính-nó):** nếu để ảnh query nằm trong gallery thì cosine similarity của nó với chính nó = 1,0 (100%), nó luôn đứng top-1 và recall ra 15/15 = 100% — hoàn toàn vô nghĩa về mặt đánh giá. Vòng smoke test đầu tiên đã cho ra đúng kết quả 100% giả đó. Chỉ loại ảnh query khỏi gallery thì mới đo được khả năng hệ thống thật sự khớp được một ảnh mới với ảnh cũ.

**Ba chỉ số.** `recall@1 (event-id)`: top-1 có đúng id sự kiện của query. `recall@3 (event-id)`: id sự kiện của query nằm bất kỳ đâu trong top-3. `recall@1 (category)`: ảnh top-1 cùng **nhóm** với query.

**Quy tắc nhóm (category):** nhóm lấy từ tiền tố tên file trước số thứ tự cuối cùng — `danh_nhau_3.jpg` → `danh_nhau`, `duong_pho_1.jpg` → `duong_pho`, `tai_nan_xe_5.jpg` → `tai_nan_xe`. Ba nhóm × 5 ảnh = 15 ảnh.

## 3. Kết quả từng ảnh (top-3)

| Query | Top-1 | Top-3 | Đúng id? | Đúng nhóm? | Score top-1 | Giây |
|---|---|---|---|---|---|---|
| danh_nhau_1.jpg | danh_nhau_4.jpg | danh_nhau_4.jpg (80%) · danh_nhau_3.jpg (78%) · tai_nan_xe_1.jpg (75%) | ✗ | ✓ | 80% | 16.7 |
| danh_nhau_2.jpg | danh_nhau_5.jpg | danh_nhau_5.jpg (68%) · danh_nhau_4.jpg (56%) · danh_nhau_1.jpg (53%) | ✗ | ✓ | 68% | 8.2 |
| danh_nhau_3.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (78%) · danh_nhau_4.jpg (77%) · tai_nan_xe_3.jpg (70%) | ✗ | ✓ | 78% | 8.0 |
| danh_nhau_4.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (80%) · danh_nhau_3.jpg (77%) · duong_pho_2.jpg (75%) | ✗ | ✓ | 80% | 8.8 |
| danh_nhau_5.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (72%) · danh_nhau_4.jpg (70%) · danh_nhau_2.jpg (68%) | ✗ | ✓ | 72% | 8.4 |
| duong_pho_1.jpg | duong_pho_5.jpg | duong_pho_5.jpg (77%) · duong_pho_2.jpg (77%) · duong_pho_4.jpg (68%) | ✗ | ✓ | 77% | 7.7 |
| duong_pho_2.jpg | duong_pho_5.jpg | duong_pho_5.jpg (81%) · duong_pho_1.jpg (77%) · danh_nhau_4.jpg (75%) | ✗ | ✓ | 81% | 7.6 |
| duong_pho_3.jpg | duong_pho_5.jpg | duong_pho_5.jpg (74%) · duong_pho_2.jpg (70%) · duong_pho_4.jpg (68%) | ✗ | ✓ | 74% | 7.6 |
| duong_pho_4.jpg | duong_pho_2.jpg | duong_pho_2.jpg (73%) · duong_pho_5.jpg (73%) · duong_pho_3.jpg (68%) | ✗ | ✓ | 73% | 7.6 |
| duong_pho_5.jpg | duong_pho_2.jpg | duong_pho_2.jpg (81%) · duong_pho_1.jpg (77%) · duong_pho_3.jpg (74%) | ✗ | ✓ | 81% | 8.0 |
| tai_nan_xe_1.jpg | tai_nan_xe_3.jpg | tai_nan_xe_3.jpg (86%) · tai_nan_xe_5.jpg (82%) · tai_nan_xe_4.jpg (81%) | ✗ | ✓ | 86% | 8.4 |
| tai_nan_xe_2.jpg | tai_nan_xe_4.jpg | tai_nan_xe_4.jpg (77%) · tai_nan_xe_5.jpg (77%) · tai_nan_xe_1.jpg (74%) | ✗ | ✓ | 77% | 8.5 |
| tai_nan_xe_3.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (86%) · tai_nan_xe_4.jpg (78%) · tai_nan_xe_5.jpg (78%) | ✗ | ✓ | 86% | 8.0 |
| tai_nan_xe_4.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (81%) · tai_nan_xe_3.jpg (78%) · tai_nan_xe_2.jpg (77%) | ✗ | ✓ | 81% | 8.6 |
| tai_nan_xe_5.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (82%) · tai_nan_xe_3.jpg (78%) · tai_nan_xe_2.jpg (77%) | ✗ | ✓ | 82% | 8.4 |

## 4. Tổng kết

| Chỉ số | Kết quả | Tỉ lệ |
|---|---|---|
| recall@1 (event-id) | **0/15** | 0.000 |
| recall@3 (event-id) | **0/15** | 0.000 |
| recall@1 (category) | **15/15** | 1.000 |

- Thời gian trung bình: **8.7s/ảnh** trên `cpu` (tổng 131s cho 15 ảnh).

## 5. Đọc kết quả

- `events.json`: mỗi id sự kiện dùng cho nhiều nhất **1 ảnh**. Ảnh query đã bị loại khỏi gallery nên không ảnh nào trong gallery mang được id của query — recall@1 và recall@3 theo event-id bằng 0 **theo cấu trúc dữ liệu**, không phải vì model yếu: hai chỉ số này đo khả năng phân biệt các sự kiện *khác nhau*, mà 15 ảnh này mỗi ảnh là một sự kiện khác nhau.
- Vì vậy con số **đáng dùng cho demo** là `recall@1 (category)` = **15/15**: hệ thống không tách được 15 sự kiện riêng biệt, nhưng vẫn giữ đúng nhóm tình huống ở ảnh đầu tiên — đúng mức để demo câu chuyện "gợi ý top-3, người giám sát chọn".
- Ảnh lọt sang nhóm khác ở top-1: không có ảnh nào.
- 15/15 ảnh không khớp id ở top-1 — xem bảng mục 3 để biết ảnh nào bị nhận nhầm sang ảnh nào.

## 6. Đánh giá bất thường (bình thường vs bất thường)

**Điểm bất thường** = `1 - max cosine similarity` tới gallery **chỉ gồm ảnh bình thường** (`duong_pho_*`). Càng xa ảnh bình thường thì điểm càng cao. Đây là phép đo khác với mục 3: mục 3 hỏi *ảnh giống sự kiện nào*, mục 6 hỏi *ảnh có bất thường không*.

- Nhãn: **5 bình thường** (`duong_pho`) · **10 bất thường** (`danh_nhau` + `tai_nan_xe`).
- **AUC = 0.960** (điểm càng cao càng tách tốt; 0,5 = đoán mò).
- Ngưỡng mặc định: **0.300** — chọn tròn giữa phân bố bình thường và bất thường, **không phải** ngưỡng tổng quát cho camera thật. Mục 6.2 cho thấy ngưỡng rất nhạy khi chỉ có 5 ảnh bình thường.

### 6.1. Kết quả tại ngưỡng mặc định

| TP | FP | TN | FN | Precision | Recall | F1 | FPR | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 9 | 0 | 5 | 1 | 1.000 | 0.900 | 0.947 | 0.000 | 0.933 |

### 6.2. Ngưỡng tốt nhất và kiểm chứng leave-one-out

| Cách chọn ngưỡng | Ngưỡng | F1 | FPR | Accuracy |
|---|---|---|---|---|
| Tốt nhất trên toàn bộ 15 ảnh (lạc quan) | 0.291 | 0.947 | 0.000 | 0.933 |
| Leave-one-out (trung thực hơn) | TB 0.286 | 0.857 | 0.400 | 0.800 |

**Vì sao cần LOO:** nếu chọn ngưỡng trên cùng 15 ảnh rồi báo kết quả trên chính 15 ảnh đó, con số bị lạc quan. LOO chọn ngưỡng trên 14 ảnh rồi mới dự đoán ảnh còn lại.

### 6.3. Từng ảnh

| Ảnh | Nhãn | Giống bình thường | Điểm bất thường | Dự đoán |
|---|---|---|---|---|
| danh_nhau_1.jpg | bất thường | 68.3% | 0.317 | bất thường |
| danh_nhau_2.jpg | bất thường | 49.0% | 0.510 | bất thường |
| danh_nhau_3.jpg | bất thường | 67.0% | 0.330 | bất thường |
| danh_nhau_4.jpg | bất thường | 75.0% | 0.250 | bình thường |
| danh_nhau_5.jpg | bất thường | 62.2% | 0.378 | bất thường |
| duong_pho_1.jpg | bình thường | 76.6% | 0.234 | bình thường |
| duong_pho_2.jpg | bình thường | 81.1% | 0.189 | bình thường |
| duong_pho_3.jpg | bình thường | 73.9% | 0.261 | bình thường |
| duong_pho_4.jpg | bình thường | 73.3% | 0.267 | bình thường |
| duong_pho_5.jpg | bình thường | 81.1% | 0.189 | bình thường |
| tai_nan_xe_1.jpg | bất thường | 68.5% | 0.315 | bất thường |
| tai_nan_xe_2.jpg | bất thường | 61.6% | 0.384 | bất thường |
| tai_nan_xe_3.jpg | bất thường | 67.9% | 0.321 | bất thường |
| tai_nan_xe_4.jpg | bất thường | 56.1% | 0.439 | bất thường |
| tai_nan_xe_5.jpg | bất thường | 62.0% | 0.380 | bất thường |

### 6.4. Baseline (để biết model hơn đoán mò bao nhiêu)

| Baseline | Accuracy | Ghi chú |
|---|---|---|
| Luôn đoán 'bất thường' | 10/15 = 0.667 | Recall = 1 nhưng FPR = 1 (báo động mọi ảnh bình thường) |
| Luôn đoán 'bình thường' | 5/15 = 0.333 | Bỏ sót toàn bộ bất thường |
| Ngẫu nhiên 3 nhóm | ~1/3 = 0.333 | Kỳ vọng đoán đúng nhóm |

## 7. Ma trận nhầm lẫn 3 nhóm (top-1)

| Thật \ Dự đoán | danh_nhau | duong_pho | tai_nan_xe |
|---|---|---|---|
| **danh_nhau** | 5 | 0 | 0 |
| **duong_pho** | 0 | 5 | 0 |
| **tai_nan_xe** | 0 | 0 | 5 |

## 8. Giới hạn của phép đo

- n = 15 ảnh; **±1 ảnh = ±6,7%**. Không có khoảng tin cậy.
- 5 ảnh 'bình thường' là ảnh đường phố tự thu thập, không phải CCTV; nhãn bình thường/bất thường do nhóm quy ước.
- Ngưỡng chọn trên chính bộ dữ liệu này; camera thật cần hiệu chỉnh lại.
- Không đo được event-level vì mỗi sự kiện chỉ có 1 ảnh (xem mục 5).
- Điểm bất thường dựa trên embedding toàn ảnh; không có local/patch feature nên không phát hiện được vật thể nhỏ.
