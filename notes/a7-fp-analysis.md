# A7 — phân tích false positive / lost true positive của tier 2 (Phước)

**Ngày:** 2026-11-01 · **Task:** A7 (optional) · **Nguồn:** `results/verify/` + `results/windows/` + UCF GT
**Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit · **Prompt hash:** `3d71fdba` · **Fake:** `false`
**Nguồn bằng chứng:** chỉ các JSON trong `results/verify/` (caption, `event_type`, verdict từng chunk, `n_chunks`, `frame_times_s`) + GT. **Chưa xem frame/video**; mọi nhận định về nội dung hình ảnh bên dưới là suy luận từ caption, không phải quan sát (xem §6).

## 0. Cách chọn case (viết trước khi xem frame)

- **Lost TP** = cửa sổ VLM `is_anomaly=false` nhưng chồng lấn ít nhất một đoạn GT, theo đúng rule của `notes/a6-protocol.md` §3: GT `[a, b]` frame → `g0=a/fps`, `g1=(b+1)/fps`; cửa sổ `[start, end)` chồng nếu `start < g1 and end > g0`.
- Có **22** lost TP (khớp từng cửa sổ với `lost_true_positive_windows` trong `results/ablation_A6.json`). Chọn **5** có thời lượng chồng GT lớn nhất (không chọn theo cảm tính):
  `Shoplifting004` 90.0 s, `Arrest039[6]` 64.5 s, `Arrest030[1]` 55.5 s, `Fighting047[0]` 54.4 s, `Stealing062` 23.0 s.
- **Lưu ý thiên lệch của luật chọn:** xếp theo giây tuyệt đối ưu tiên cửa sổ dài, nơi GT chỉ chiếm một phần (cột *GT/cửa sổ* ở §1: 20–52%). Có **6/22** lost TP nằm **100% trong GT** (`Shooting047[1]`, `Burglary037[1]`, `Stealing079[0]`, `Vandalism036[0]`, `Arrest039[5]`, `Burglary035[0]`) và không được chọn vì ngắn; xem §5. Đọc "lost TP" ở đây là "cửa sổ có chồng GT bị reject", **không** phải "VLM bỏ sót tội phạm đúng lúc nó xảy ra".
- **Surviving FA** = cửa sổ VLM `confirmed` trong video normal. Toàn bộ repo chỉ có **1** case (`Normal_Videos_924[10]`), nên dùng cả 1 và ghi rõ **1/5** — không bịa thêm case.
- Đây là phân tích định tính cho report, không phải benchmark; 22 lost TP là con số rule-based, TH7c là nguồn chính thức cho `lost_true_positive_windows`.

## 1. 5 lost TP lớn nhất

`GT/cửa sổ` = giây chồng GT chia độ dài cửa sổ. `Chunk (frame)` = số chunk (≤30 s, mỗi chunk 4 frame) và tổng frame VLM thấy.

| # | Cửa sổ | Thời gian | GT overlap | GT/cửa sổ | Chunk (frame) | VLM type / `is_anomaly` | Caption VLM | Nhận định (từ JSON) |
|---|---|---|---:|---:|---:|---|---|---|
| 1 | `Shoplifting004_x264[0]` | 28–201 s | 90.0 s | 52% | 6 (24) | `Normal` / false | "A man is seen handling money at a counter." | Cửa sổ dài 173 s, ~1 frame/7 s. Caption mô tả cảnh quầy bình thường, không nêu hành vi lấy đồ. |
| 2 | `Arrest039_x264[6]` | 280–526 s | 64.5 s | 26% | 9 (36) | `Arrest` / false | "A person is being arrested by law enforcement." | **Mâu thuẫn type–verdict**: caption/type nói Arrest nhưng `is_anomaly=false`. 74% cửa sổ nằm ngoài GT. |
| 3 | `Arrest030_x264[1]` | 15–289 s | 55.5 s | 20% | 10 (40) | `Arrest` / false | "A person enters a room and is arrested by another individual." | Cùng mâu thuẫn. 80% cửa sổ nằm ngoài GT nên verdict `false` một phần có thể bào chữa. |
| 4 | `Fighting047_x264[0]` | 4–149 s | 54.4 s | 37% | 5 (20) | `Fighting` / false | "Two individuals are engaged in a physical altercation near a blue pole in a workshop setting." | Cùng mâu thuẫn: caption mô tả đánh nhau nhưng verdict `false`. |
| 5 | `Stealing062_x264[0]` | 0–52 s | 23.0 s | 44% | 2 (8) | `Normal` / false | "A person is walking away from a vehicle." | Caption chỉ nêu "walking away", không nêu hành vi trộm. Không kết luận được gì thêm về nội dung frame khi chưa xem. |

## 2. 1 surviving FA

| # | Cửa sổ | Thời gian | Window type / `is_anomaly` | Caption chunk 0 | Nguyên nhân |
|---|---|---|---:|---|---|
| 6 | `Normal_Videos_924_x264[10]` | 855–1156 s | `Robbery` / true | "A person is sitting on the floor while another person approaches them with a bag." | **Artifact của luật gộp OR-trên-chunk (D-1), không phải VLM nghiêng về `true`.** Cửa sổ 301 s có **11 chunk**; chỉ **chunk 0** (855–882 s) là `Robbery`/true (conf 0.95); **10 chunk còn lại đều `Normal`/false**, caption mô tả người ngồi/đứng gần nhau. Một chunk fire là đủ để cả cửa sổ bị confirm. |

Cửa sổ càng dài thì càng nhiều cơ hội để một chunk fire sai (tăng theo `n_chunks`). Case này **khác cơ chế** với 3 FP còn lại của A6b (`Assault010`, `Burglary032`, `Explosion002`: cửa sổ ngắn 1–4 s, 1 chunk, bằng chứng yếu). `notes/a6-handcheck.md` §3 đã ghi đây là ngoại lệ mà sửa prompt không xử lý được.

## 3. Nguyên nhân lặp

1. **Type–verdict contradiction (3/5 lost TP trong mẫu; 15/22 lost TP tổng).** `Arrest039[6]`, `Arrest030[1]`, `Fighting047[0]`: VLM tự nói type là Arrest/Fighting, caption mô tả đúng hành vi, nhưng `is_anomaly=false`. Đây là phần giao giữa nhóm **37 cửa sổ** đã đếm ở `notes/a5b-log.md` (`is_anomaly=false`, `event_type != Normal`) và GT. Chỉ **15/37** chồng GT; **22/37** không chồng (19 ở video anomalous, 3 ở video normal: `Normal_Videos_884[0]`, `894[0]`, `894[2]`). Nhóm này là hỗn hợp, không phải "mất oan" hoàn toàn.
2. **Hành vi nhỏ / cảnh rộng (2/5 lost TP).** `Shoplifting004` và `Stealing062`: caption mô tả cảnh bình thường. Mật độ frame thưa (4 frame **mỗi chunk** ≤30 s, thực tế ~1 frame/7 s: 24 frame/173 s, 8 frame/52 s) nên một hành vi ngắn có thể rơi giữa hai frame được lấy mẫu. Đây là giới hạn của mật độ frame + cửa sổ dài, không chỉ prompt.
3. **Artifact OR-trên-chunk (1 FA).** `Normal_Videos_924[10]` (301 s, 11 chunk, 1 chunk fire). Cơ chế gộp, không phải over-confirm do bằng chứng yếu, và khác nhóm 3 FP ngắn của A6b. Đổi luật gộp là thứ A6 không cho phép.

## 4. Hướng cải thiện (limitation / future work; A6c đã SKIPPED)

A6c không chạy (gate D4-2, `notes/2026-11-01-day4-tasks.md`: untuned A6 là headline). Mục này chỉ để report ghi như hạn chế và hướng tương lai, **không phải việc cần làm**.

1. **Tự nhất quán type–verdict** (`event_type != Normal` ⇒ `is_anomaly=true`): chưa chắc có lợi. Trong 37 cửa sổ mâu thuẫn chỉ 15 chồng GT. Giả sử đảo cả 37 thì tính lại từ `results/verify/` ra **79 TP / 87 FP, precision 47.6%**, so với **64 / 65, 49.6%** hiện tại, và thêm 3 FA ở video normal. Vì vậy **không** viết "prompt nhất quán sẽ cứu được các case này".
2. **Siết bằng chứng**: có thể giảm FP cửa sổ ngắn của A6b, nhưng không sửa được artifact OR-trên-chunk của §2.
3. **Frame dày hơn / cửa sổ ngắn hơn / ngưỡng số chunk thay cho OR**: đúng với nhóm hành vi nhỏ và case #6, nhưng ngoài scope A6.
- **Không** post-process `event_type` thành verdict sau khi đã thấy số: đó là đổi định nghĩa A6, không phải prompt tuning. Nếu thử, phải là row riêng, ghi deviation.

## 5. Phụ lục — 22 lost TP (rule-based, sorted by overlap)

Tính lại từ `results/verify/` + GT; tập 22 cửa sổ khớp `results/ablation_A6.json`. 6 hàng **100%** là lost TP "sạch nhất" (cửa sổ nằm trọn trong GT) nhưng không vào mẫu 5 vì ngắn.

| Cửa sổ | Overlap GT (s) | GT/cửa sổ | Chunk (frame) | VLM type | Ghi chú |
|---|---:|---:|---:|---|---|
| `Shoplifting004_x264[0]` | 90.0 | 52% | 6 (24) | Normal | case #1 |
| `Arrest039_x264[6]` | 64.5 | 26% | 9 (36) | Arrest | case #2 |
| `Arrest030_x264[1]` | 55.5 | 20% | 10 (40) | Arrest | case #3 |
| `Fighting047_x264[0]` | 54.4 | 37% | 5 (20) | Fighting | case #4 |
| `Stealing062_x264[0]` | 23.0 | 44% | 2 (8) | Normal | case #5 |
| `Arrest007_x264[2]` | 21.0 | 75% | 1 (4) | Arrest | contradiction |
| `Shooting047_x264[1]` | 17.0 | **100%** | 1 (4) | Robbery | contradiction |
| `Shoplifting029_x264[0]` | 15.0 | 48% | 2 (8) | Shoplifting | contradiction |
| `Burglary037_x264[1]` | 13.0 | **100%** | 1 (4) | Robbery | contradiction |
| `Arrest001_x264[0]` | 10.0 | 13% | 3 (12) | Normal | caption tả cảnh thường |
| `Shoplifting028_x264[0]` | 9.0 | 35% | 1 (4) | Shoplifting | contradiction |
| `Shoplifting005_x264[1]` | 7.0 | 33% | 1 (4) | Shoplifting | contradiction |
| `Shoplifting031_x264[0]` | 7.0 | 47% | 1 (4) | Normal | caption tả cảnh thường |
| `Stealing079_x264[0]` | 7.0 | **100%** | 1 (4) | Robbery | contradiction |
| `Shoplifting021_x264[2]` | 5.0 | 34% | 1 (4) | Stealing | contradiction |
| `Vandalism036_x264[0]` | 5.0 | **100%** | 1 (4) | Vandalism | contradiction |
| `Shoplifting022_x264[2]` | 4.0 | 58% | 1 (4) | Normal | caption tả cảnh thường |
| `Arrest039_x264[4]` | 3.5 | 58% | 1 (4) | Arrest | contradiction |
| `Arrest039_x264[5]` | 3.0 | **100%** | 1 (4) | Arrest | contradiction |
| `Burglary035_x264[0]` | 2.0 | **100%** | 1 (4) | Burglary | contradiction |
| `Shoplifting022_x264[0]` | 1.0 | 7% | 1 (4) | Normal | caption tả cảnh thường |
| `RoadAccidents021_x264[0]` | 0.03 | 2% | 1 (4) | Normal | biên GT, overlap rất nhỏ |

## 6. Giới hạn

- 5 lost TP chọn theo overlap lớn nhất (giây tuyệt đối), không random và **không** theo tỉ lệ GT/cửa sổ; xem §0 và cột *GT/cửa sổ*. 1 FA là toàn bộ số FA ở video normal trong repo ("surviving FA" ở đây chỉ tính video normal). Ngoài ra còn 65 cửa sổ confirmed không chồng GT (precision cửa sổ 64/129 = 49.6%), trong đó 3 ở video anomalous đã được gán nhãn tay ở A6b (`Assault010`, `Burglary032`, `Explosion002`).
- Phân tích nguyên nhân dựa trên **caption + type + verdict từng chunk + độ dài cửa sổ** trong JSON; **chưa xem frame**. Mọi câu về nội dung hình ảnh là suy luận từ caption. Số frame mỗi cửa sổ = 4 × `n_chunks` (không cố định), mật độ ~1 frame/7 s.
- 22 lost TP là rule-based trên `results/verify/`; `results/ablation_A6.json` (TH7c) là nguồn chính thức và khớp. Nếu sau này ra số khác, điều tra rule trước khi sửa note.
- Đây là task optional (task sheet yêu cầu 10 case, giao 6); không dùng để đổi tau/rule A6.
