# A6b — hand check 20 cửa sổ tier 2 (Phước)

**Ngày:** 2026-11-01 · **Task:** A6b · **Mẫu:** `notes/a6-handcheck-sample.csv` (PR #36)
**Nhãn người:** `notes/a6-handcheck-labelled.csv`, commit `a6de7a1` — **commit trước khi unblind**
**Key:** ngoài repo, SHA-256 `4f239f615f1814a45b27e383c42852505deac9f88773596607d43d3eb2f49805`
**Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit · **Prompt hash:** `3d71fdba` · **Fake:** `false`

## 0. Cách chấm

- 20 cửa sổ = 10 VLM `confirmed` + 10 VLM `rejected`; 12 cửa sổ thuộc video anomalous, 8 thuộc video normal.
- Người chấm xem **12 frame/cửa sổ** (frame của VLM + frame đều trong khoảng), không xem verdict/caption/GT.
- Trang label: `anomaly` / `normal` / `unsure`; ghi chú ngắn nếu có.
- CSV nhãn được commit (`a6de7a1`) **trước** khi mở key; sau đó mới unblind để viết note này.
- Đây là **nhãn người**, không phải GT UCF: "anomaly" = người thấy có sự cố trong cửa sổ, kể cả khi ngoài đoạn GT.

## 1. Kết quả

Nhãn người: **6 anomaly · 13 normal · 1 unsure**.

| VLM \ Người | anomaly | normal | unsure |
|---|---:|---:|---:|
| **confirmed** | 6 (TP) | 4 (FP) | 0 |
| **rejected** | 0 (FN) | 9 (TN) | 1 |

- **Precision của confirmed windows = 6/10 = 60.0%** (95% Wilson: 31.3–83.2%).
- **Recall trên nhãn người quyết định = 6/6 = 100%**; rejected đúng 9/9 cửa sổ quyết định, 1 unsure.
- Accuracy trên 19 nhãn quyết định = 15/19 = **78.9%**. Nếu tính unsure là normal: 16/20 = 80%; nếu unsure là anomaly: 15/20 = 75%.
- Không có false negative trong 9 cửa sổ rejected quyết định; rủi ro chính của mẫu này là **over-confirm**, không phải bỏ sót.

## 2. So với GT overlap (proxy cũ)

Trên cùng 20 cửa sổ:
- 6 TP theo nhãn người: chỉ **2** chồng lấn GT (`Robbery048`, `Shooting008`); **4** không chồng GT (`Explosion021`, `Burglary079`, `Explosion036`, `Assault006`).
- 4 FP theo nhãn người: tất cả **không** chồng GT; 3 nằm trong video anomalous (`Assault010`, `Burglary032`, `Explosion002`), 1 trong video normal (`Normal_Videos_924`).

Vì vậy con số **2/9 ≈ 22% "confirmation precision"** tính bằng GT overlap ở PR #36 là **proxy**, không phải precision của VLM: nó phạt cả những cửa sổ có sự cố thật nhưng nằm ngoài đoạn GT. Với nhãn người, precision là **6/10 = 60%**. Hai số đo hai thứ khác nhau; report phải nói rõ định nghĩa.

## 3. 4 false positive — nguyên nhân

| # | Cửa sổ | Dài | VLM nói | Người | Nguyên nhân (đọc frame + caption VLM) |
|---|---|---|---:|---|---|
| 1 | `Assault010_x264[0]` | 3 s | confirmed, `Assault`, "A prisoner is being restrained by guards in a prison cell." | normal | Người nằm/ngồi trên ghế, một người đứng cạnh; dáng bất thường nhưng không có hành vi tấn công. VLM đọc tư thế thành "bị khống chế". |
| 2 | `Burglary032_x264[3]` | 1 s | confirmed, `Burglary`, "A person is seen entering a room through a door." | normal | Cửa sổ 1 s, hình tối/mờ, chỉ thấy người đi qua cửa; không có dấu hiệu đột nhập. VLM over-read chuyển động. |
| 3 | `Explosion002_x264[1]` | 4 s | confirmed, `Explosion`, "A person is seen walking towards a gas pump at night." | normal | Caption của VLM mô tả **hành vi bình thường** (người đi bộ tới trụ xăng) nhưng `is_anomaly=true` và type `Explosion`; không có lửa/nổ trong frame. Đây là mâu thuẫn caption–verdict. |
| 4 | `Normal_Videos_924_x264[10]` | 301 s | confirmed, `Robbery`, "A person is sitting on the floor while another person approaches them with a bag." | normal | Cửa sổ rất dài (301 s) trong phòng, người ngồi/đi lại bình thường; VLM over-confirm một cảnh thường. |

**Nguyên nhân lặp:**
- **3/4** là cửa sổ ngắn 1–4 s (`Burglary032`, `Assault010`, `Explosion002`) hoặc rất dài 301 s (`Normal_Videos_924[10]`): bằng chứng ít, VLM dễ over-read chuyển động/tư thế.
- **2/4** là caption mô tả hành vi bình thường nhưng verdict vẫn `true` (`Explosion002`, `Normal_Videos_924[10]`).
- Cả 4 đều là **bằng chứng yếu**: không có vật thể/hành vi phạm tội rõ ràng. Một prompt yêu cầu "nếu caption không mô tả được hành vi phạm tội cụ thể thì `is_anomaly=false`" có thể nhắm đúng nguyên nhân này.

## 4. 6 true positive — nhãn người

| Cửa sổ | Người | Note | GT overlap | VLM type/caption |
|---|---|---|---|---|
| `Explosion021_x264[0]` | anomaly | cháy nổ | no | `Explosion`, mô tả vụ nổ — đúng. |
| `Robbery048_x264[0]` | anomaly | trộm cướp | yes | `Robbery`, caption chỉ nói "entering a convenience store" — verdict đúng nhưng caption yếu. |
| `Burglary079_x264[2]` | anomaly | hành vi | no | `Burglary`, người mở cửa xe/tìm đồ — đúng. |
| `Shooting008_x264[0]` | anomaly | đấu súng | yes | `Shoplifting`, caption "attempting to steal a bag" — **type/caption sai**, verdict đúng. |
| `Explosion036_x264[2]` | anomaly | tụ tập đông người | no | `Explosion`, người nằm/đám đông — verdict đúng, type chưa chắc. |
| `Assault006_x264[0]` | anomaly | trộm | no | `Assault`, caption "running away from a security camera" — verdict đúng, type/caption lệch. |

Bài học: VLM `is_anomaly` có ích, nhưng **type/caption không đáng tin** — 2/6 TP có type/caption sai rõ (`Shooting008`, `Assault006`). Việc D-2 chuyển alert type sang typer là đúng; report phải tách dòng "VLM type accuracy" khỏi detection.

## 5. 1 unsure

`Normal_Videos_246_x264[2]` (76–167 s, 91 s): VLM `rejected`, người chấm `unsure` vì hình bị che khuất. Không đưa vào precision/recall; nếu tính là anomaly thì rejected recall 0/10, nếu normal thì 10/10. Sample quá nhỏ để kết luận.

## 6. Kết luận cho A6 / gate D4-2

- Trên 10 cửa sổ VLM confirmed: **6 đúng, 4 sai → precision 60%** (CI rất rộng 31–83%); trên 10 rejected: **9 đúng, 1 unsure, 0 bỏ sót**.
- VLM **không bỏ sót** trong mẫu này; lỗi là **over-confirm** trên cửa sổ dài/ngắn/ambiguous.
- **Gate D4-2:** có đủ input trước 13:00. 4/4 FP cùng một họ nguyên nhân ("bằng chứng yếu / caption bình thường nhưng verdict true"), nên **có thể cân nhắc A6c**. Tuy nhiên mẫu chỉ 10 confirmed, CI rộng; default của plan là **skip**. Nếu GO: sửa prompt **một lần**, thêm ràng buộc caption phải mô tả hành vi phạm tội cụ thể, rerun 307 cửa sổ ra `results/verify_tuned/`, tạo row `A6-tuned` riêng; không thay row A6 gốc.
- Không đổi tau, không đổi định nghĩa A6 sau khi đã thấy số.

## 7. Giới hạn

- 20 cửa sổ, 10 confirmed — CI rộng; không phải ước lượng prevalence.
- Người chấm xem 12 frame/cửa sổ, không xem toàn bộ video; sự cố ngắn giữa các frame có thể bị bỏ qua.
- `video_id` vẫn lộ lớp (`Normal_Videos_*`) — blind không tuyệt đối (đã ghi ở A6a).
- `results/verify/` trong repo chứa verdict/caption của đúng 20 cửa sổ; blind dựa trên kỷ luật, người chấm không mở các file đó.
- Nhãn người không phải GT UCF; 4 TP không chồng GT cho thấy khác biệt giữa "người thấy bất thường" và "đoạn annotation".
