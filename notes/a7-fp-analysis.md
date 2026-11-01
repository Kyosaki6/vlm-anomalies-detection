# A7 — phân tích false positive / lost true positive của tier 2 (Phước)

**Ngày:** 2026-11-01 · **Task:** A7 (optional) · **Nguồn:** `results/verify/` + `results/windows/` + UCF GT
**Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit · **Prompt hash:** `3d71fdba` · **Fake:** `false`

## 0. Cách chọn case (viết trước khi xem frame)

- **Lost TP** = cửa sổ VLM `is_anomaly=false` nhưng chồng lấn ít nhất một đoạn GT, theo đúng rule của `notes/a6-protocol.md` §3: GT `[a, b]` frame → `g0=a/fps`, `g1=(b+1)/fps`; cửa sổ `[start, end)` chồng nếu `start < g1 and end > g0`.
- Có **22** lost TP. Chọn **5** có thời lượng chồng GT lớn nhất (không chọn theo cảm tính):
  `Shoplifting004` 90.0 s, `Arrest039[6]` 64.5 s, `Arrest030[1]` 55.5 s, `Fighting047[0]` 54.4 s, `Stealing062` 23.0 s.
- **Surviving FA** = cửa sổ VLM `confirmed` trong video normal. Toàn bộ repo chỉ có **1** case (`Normal_Videos_924[10]`), nên dùng cả 1 và ghi rõ **1/5** — không bịa thêm case.
- Đây là phân tích định tính cho report, không phải benchmark; 22 lost TP là con số rule-based, TH7c là nguồn chính thức cho `lost_true_positive_windows`.

## 1. 5 lost TP lớn nhất

| # | Cửa sổ | Thời gian | GT overlap | VLM type / `is_anomaly` | Caption VLM | Nguyên nhân |
|---|---|---|---:|---|---|---|
| 1 | `Shoplifting004_x264[0]` | 28–201 s | 90.0 s | `Normal` / false | "A man is seen handling money at a counter." | Hành vi tay–tiền ở quầy, không có động tác lấy đồ rõ; cửa sổ dài 173 s. VLM mô tả đúng cảnh nhưng không gọi được shoplifting. |
| 2 | `Arrest039_x264[6]` | 280–526 s | 64.5 s | `Arrest` / false | "A person is being arrested by law enforcement." | **Mâu thuẫn type–verdict**: caption/type nói Arrest, frame có người bị khống chế, nhưng `is_anomaly=false`. |
| 3 | `Arrest030_x264[1]` | 15–289 s | 55.5 s | `Arrest` / false | "A person enters a room and is arrested by another individual." | Cùng lỗi: type/caption Arrest, frame có người nằm/giữ, verdict false. Cửa sổ 274 s. |
| 4 | `Fighting047_x264[0]` | 4–149 s | 54.4 s | `Fighting` / false | "Two individuals are engaged in a physical altercation near a blue pole in a workshop setting." | Cùng lỗi: caption mô tả đánh nhau, frame có đám đông/xô xát, verdict false. |
| 5 | `Stealing062_x264[0]` | 0–52 s | 23.0 s | `Normal` / false | "A person is walking away from a vehicle." | Người quanh xe cảnh sát, hành vi trộm nhỏ/che khuất; VLM chỉ thấy "walking away". |

## 2. 1 surviving FA

| # | Cửa sổ | Thời gian | VLM type / `is_anomaly` | Caption VLM | Nguyên nhân |
|---|---|---|---:|---|---|
| 6 | `Normal_Videos_924_x264[10]` | 855–1156 s | `Robbery` / true | "A person is sitting on the floor while another person approaches them with a bag." | Cửa sổ 301 s trong phòng, người ngồi/đi lại bình thường; VLM over-confirm. Trùng với FP #4 của A6b. |

## 3. Nguyên nhân lặp

1. **Type–verdict contradiction (3/5 lost TP).** `Arrest039[6]`, `Arrest030[1]`, `Fighting047[0]`: VLM tự nói type là Arrest/Fighting và caption mô tả đúng hành vi, nhưng `is_anomaly=false`. Đây chính là nhóm **37 cửa sổ** đã đếm ở A5b (`is_anomaly=false`, `event_type != Normal`). Với verdict-only, 3 case này bị mất oan.
2. **Hành vi nhỏ / cảnh rộng (2/5 lost TP).** `Shoplifting004` (tay–tiền ở quầy) và `Stealing062` (người quanh xe, hành vi bị che): VLM mô tả cảnh bình thường, không nhận ra hành vi trộm. Đây là giới hạn của 4 frame/cửa sổ + cửa sổ dài, không chỉ prompt.
3. **Over-confirm trên cảnh thường dài (1 FA).** `Normal_Videos_924[10]` 301 s: cùng cơ chế với 4 FP của A6b — bằng chứng yếu, cửa sổ dài, VLM nghiêng về `true`.

## 4. Liên hệ A6c (chỉ là gợi ý, không đổi rule)

- A6c được phép sửa prompt **một lần**. Hai hướng đúng với dữ liệu này:
  1. **Buộc model tự nhất quán**: nếu `event_type != Normal` thì caption phải mô tả hành vi phạm tội cụ thể và `is_anomaly` phải `true`; nếu caption chỉ mô tả hành vi bình thường thì `is_anomaly=false`.
  2. **Siết bằng chứng**: nếu không thấy vật thể/hành vi phạm tội rõ trong frame thì trả `false`, kể cả cửa sổ dài.
- **Không được** post-process `event_type` thành verdict sau khi đã thấy số: đó là đổi định nghĩa A6 (`is_anomaly` là verdict), không phải prompt tuning. Nếu muốn thử, phải là row riêng, ghi deviation.
- Với 3/5 lost TP do contradiction, prompt nhất quán có khả năng cứu được các case này; 2/5 còn lại (hành vi nhỏ) khó sửa bằng prompt, cần frame dày hơn/cửa sổ ngắn hơn — ngoài scope A6c.

## 5. Phụ lục — 22 lost TP (rule-based, sorted by overlap)

| Cửa sổ | Overlap GT (s) | VLM type | Ghi chú |
|---|---:|---|---|
| `Shoplifting004_x264[0]` | 90.0 | Normal | case #1 |
| `Arrest039_x264[6]` | 64.5 | Arrest | case #2 |
| `Arrest030_x264[1]` | 55.5 | Arrest | case #3 |
| `Fighting047_x264[0]` | 54.4 | Fighting | case #4 |
| `Stealing062_x264[0]` | 23.0 | Normal | case #5 |
| `Arrest007_x264[2]` | 21.0 | Arrest | contradiction |
| `Shooting047_x264[1]` | 17.0 | Robbery | contradiction |
| `Shoplifting029_x264[0]` | 15.0 | Shoplifting | contradiction |
| `Burglary037_x264[1]` | 13.0 | Robbery | contradiction |
| `Arrest001_x264[0]` | 10.0 | Normal | caption tả cảnh thường |
| `Shoplifting028_x264[0]` | 9.0 | Shoplifting | contradiction |
| `Shoplifting005_x264[1]` | 7.0 | Shoplifting | contradiction |
| `Shoplifting031_x264[0]` | 7.0 | Normal | caption tả cảnh thường |
| `Stealing079_x264[0]` | 7.0 | Robbery | contradiction |
| `Shoplifting021_x264[2]` | 5.0 | Stealing | contradiction |
| `Vandalism036_x264[0]` | 5.0 | Vandalism | contradiction |
| `Shoplifting022_x264[2]` | 4.0 | Normal | caption tả cảnh thường |
| `Arrest039_x264[4]` | 3.5 | Arrest | contradiction |
| `Arrest039_x264[5]` | 3.0 | Arrest | contradiction |
| `Burglary035_x264[0]` | 2.0 | Burglary | contradiction |
| `Shoplifting022_x264[0]` | 1.0 | Normal | caption tả cảnh thường |
| `RoadAccidents021_x264[0]` | 0.03 | Normal | biên GT, overlap rất nhỏ |

## 6. Giới hạn

- 5 lost TP chọn theo overlap lớn nhất, không random; 1 FA là toàn bộ số FA normal-video trong repo.
- Phân tích nguyên nhân dựa trên 12 frame/cửa sổ + caption VLM, không xem toàn bộ video.
- 22 lost TP là rule-based trên `results/verify/` hiện tại; TH7c (`results/ablation_A6.json`) là nguồn chính thức cho report. Nếu TH7c ra số khác, điều tra rule trước khi sửa note.
- Đây là task optional; không dùng để đổi tau/rule A6.
