# EVAL top-3 — google/siglip2-base-patch16-384

## 1. Thông tin lần chạy

- Ngày chạy: **2026-10-06 11:14** · máy: `DESKTOP-AVDV3SH`
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
| danh_nhau_1.jpg | danh_nhau_4.jpg | danh_nhau_4.jpg (80%) · danh_nhau_3.jpg (78%) · tai_nan_xe_1.jpg (75%) | ✗ | ✓ | 80% | 25.4 |
| danh_nhau_2.jpg | danh_nhau_5.jpg | danh_nhau_5.jpg (68%) · danh_nhau_4.jpg (56%) · danh_nhau_1.jpg (53%) | ✗ | ✓ | 68% | 9.0 |
| danh_nhau_3.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (78%) · danh_nhau_4.jpg (77%) · tai_nan_xe_3.jpg (70%) | ✗ | ✓ | 78% | 9.1 |
| danh_nhau_4.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (80%) · danh_nhau_3.jpg (77%) · duong_pho_2.jpg (75%) | ✗ | ✓ | 80% | 9.6 |
| danh_nhau_5.jpg | danh_nhau_1.jpg | danh_nhau_1.jpg (72%) · danh_nhau_4.jpg (70%) · danh_nhau_2.jpg (68%) | ✗ | ✓ | 72% | 9.7 |
| duong_pho_1.jpg | duong_pho_5.jpg | duong_pho_5.jpg (77%) · duong_pho_2.jpg (77%) · duong_pho_4.jpg (68%) | ✗ | ✓ | 77% | 9.2 |
| duong_pho_2.jpg | duong_pho_5.jpg | duong_pho_5.jpg (81%) · duong_pho_1.jpg (77%) · danh_nhau_4.jpg (75%) | ✗ | ✓ | 81% | 11.8 |
| duong_pho_3.jpg | duong_pho_5.jpg | duong_pho_5.jpg (74%) · duong_pho_2.jpg (70%) · duong_pho_4.jpg (68%) | ✗ | ✓ | 74% | 9.5 |
| duong_pho_4.jpg | duong_pho_2.jpg | duong_pho_2.jpg (73%) · duong_pho_5.jpg (73%) · duong_pho_3.jpg (68%) | ✗ | ✓ | 73% | 9.3 |
| duong_pho_5.jpg | duong_pho_2.jpg | duong_pho_2.jpg (81%) · duong_pho_1.jpg (77%) · duong_pho_3.jpg (74%) | ✗ | ✓ | 81% | 9.7 |
| tai_nan_xe_1.jpg | tai_nan_xe_3.jpg | tai_nan_xe_3.jpg (86%) · tai_nan_xe_5.jpg (82%) · tai_nan_xe_4.jpg (81%) | ✗ | ✓ | 86% | 9.3 |
| tai_nan_xe_2.jpg | tai_nan_xe_4.jpg | tai_nan_xe_4.jpg (77%) · tai_nan_xe_5.jpg (77%) · tai_nan_xe_1.jpg (74%) | ✗ | ✓ | 77% | 10.0 |
| tai_nan_xe_3.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (86%) · tai_nan_xe_4.jpg (78%) · tai_nan_xe_5.jpg (78%) | ✗ | ✓ | 86% | 10.1 |
| tai_nan_xe_4.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (81%) · tai_nan_xe_3.jpg (78%) · tai_nan_xe_2.jpg (77%) | ✗ | ✓ | 81% | 18.8 |
| tai_nan_xe_5.jpg | tai_nan_xe_1.jpg | tai_nan_xe_1.jpg (82%) · tai_nan_xe_3.jpg (78%) · tai_nan_xe_2.jpg (77%) | ✗ | ✓ | 82% | 8.4 |

## 4. Tổng kết

| Chỉ số | Kết quả | Tỉ lệ |
|---|---|---|
| recall@1 (event-id) | **0/15** | 0.000 |
| recall@3 (event-id) | **0/15** | 0.000 |
| recall@1 (category) | **15/15** | 1.000 |

- Thời gian trung bình: **11.3s/ảnh** trên `cpu` (tổng 169s cho 15 ảnh).

## 5. Đọc kết quả

- `events.json`: mỗi id sự kiện dùng cho nhiều nhất **1 ảnh**. Ảnh query đã bị loại khỏi gallery nên không ảnh nào trong gallery mang được id của query — recall@1 và recall@3 theo event-id bằng 0 **theo cấu trúc dữ liệu**, không phải vì model yếu: hai chỉ số này đo khả năng phân biệt các sự kiện *khác nhau*, mà 15 ảnh này mỗi ảnh là một sự kiện khác nhau.
- Vì vậy con số **đáng dùng cho demo** là `recall@1 (category)` = **15/15**: hệ thống không tách được 15 sự kiện riêng biệt, nhưng vẫn giữ đúng nhóm tình huống ở ảnh đầu tiên — đúng mức để demo câu chuyện "gợi ý top-3, người giám sát chọn".
- Ảnh lọt sang nhóm khác ở top-1: không có ảnh nào.
- 15/15 ảnh không khớp id ở top-1 — xem bảng mục 3 để biết ảnh nào bị nhận nhầm sang ảnh nào.
