# Kiểm toán mọi con số trong báo cáo (02/11/2026)

**Câu hỏi đặt ra:** "tôi không chắc về các con số". Note này trả lời bằng cách kiểm lại **từng con số
đầu mối** in trong `report/`, không tin vào file JSON tóm tắt có sẵn.

**Kết quả:** `scripts/audit_report_numbers.py` chạy **55/55 PASS**. Các con số còn lại (typing,
hand-check, số test, thông số trích dẫn) được kiểm bằng ba cách khác, ghi ở §3–§5.

```powershell
python scripts/audit_report_numbers.py `
  --annotations datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

Script đọc dữ liệu thô (`manifests/ucf_test.csv`, `results/run_A4c/*.json`,
`results/run_A4c/windows/`, `results/verify/`) rồi **tự tính lại**: đếm video/cửa sổ, FA/hour, tỉ lệ
xác nhận tầng 2, precision, incident recall, latency, AUC frame-level (tự viết hàm AUC có xử lý hoà),
và so với giá trị đang in trong báo cáo. Không cần GPU, không cần video.

## 1. Bảng đối chiếu (55 mục, trích các mục chính)

| Con số trong báo cáo | Nguồn gốc | Cách kiểm | Kết quả |
|---|---|---|---|
| 290 video · 140 bất thường · 150 bình thường | `manifests/ucf_test.csv` | đếm lại theo `path` | PASS |
| 1,111,808 frame gốc | manifest `n_frames` | cộng lại | PASS |
| 307 cửa sổ · 114 video có cửa sổ · 1.06 cửa sổ/video | `results/run_A4c/windows/` | đếm lại từ 290 file | PASS |
| 124 cửa sổ trên video bình thường | windows + nhãn video | đếm lại | PASS |
| 6.01 giờ video bình thường · FA/hour 20.64 | manifest `n_frames/fps` | tính lại | PASS |
| 4.01 giờ · FA/hour 9.73 (bỏ 2 video dài) | như trên, trừ `Normal_Videos_935/924` | tính lại | PASS |
| 307 file tầng 2 · 129 xác nhận · 178 loại · 42.0% | `results/verify/*/window_*.json` | đếm cờ `is_anomaly` từng file | PASS |
| 1 cửa sổ còn trên video bình thường · 0.17 FA/hour · giảm 99.2% | như trên | tính lại | PASS |
| Precision 28.0% (86/307) · 49.6% (64/129) | `ablation_A6.json` → `operating_*` | đối chiếu + **tính lại từ cửa sổ + file nhãn** | PASS |
| Incident recall 51.9% (81/156) · 39.7% (62/156) | như trên | tính lại từ GT (156 sự cố đếm lại được) | PASS |
| Latency 2.03 s (n=81) · 1.74 s (n=62) | như trên | tính lại `max(0, min(start) − gt_start)` | PASS |
| AUC 73.00 [64.45, 81.15] · 67.90 [58.39, 77.49] · Δ −5.10 [−11.42, −0.25] | `ablation_A4c.json`, `ablation_A6.json` | **tái lập 73.00 và 63.35 trực tiếp từ `anomaly_scores` + nhãn GT** | PASS |
| AUC macro 63.35 | `ablation_A4c.json` | tính lại (trung bình theo video) | PASS |
| A0…A6 = 48.77 / 65.87 / 72.39 / 58.99 / 58.41 / 57.08 / 67.90 | `results/ablation_A*.json` | đọc lại từng file | PASS |
| VLM typing 83.9% (52/62) · typer 24.2% (15/62) | `scripts/eval_vlm_typing.py` | **chạy lại script** trên `results/verify` | khớp từng số |
| Hand-check 6/4/0 và 0/9/1, precision mẫu 6/10 | `notes/a6-handcheck-labelled.csv` + `results/verify` | dựng lại bảng chéo từ nhãn người | khớp |
| Median 3.71 s · p90 14.41 s mỗi cửa sổ VLM | `ablation_A6.json` → `cost` | đọc lại | PASS |
| 868 test pass | `pytest -q` | chạy lại toàn bộ (cục bộ 25 s; CI 4/4 job: ubuntu/windows × py3.10/3.11) | khớp |
| RTFM 84.30% frame-level AUC (UCF-Crime, I3D RGB) | bài báo RTFM, arXiv:2101.10030 | đọc **Bảng 2** của bài báo | khớp |

## 2. Hai điều dễ nói sai khi bảo vệ

1. **Không được lấy 73.00% "trung bình cộng" của AUC từng video.** Hai chỉ số khác nhau về định nghĩa:
   - **Micro 73.00%**: gộp toàn bộ cặp (frame bất thường, frame bình thường) trên 290 video, kể cả cặp
     **giữa hai video khác nhau**.
   - **Macro 63.35%**: trung bình AUC của từng video. Báo cáo in cả hai; nếu chỉ nói "73%" mà giáo viên
     hỏi "trung bình theo video là bao nhiêu" thì câu trả lời là 63.35%, không phải 73%.
   Đã kiểm: 63.35% tái lập được bằng trung bình AUC từng video; 73.00% tái lập được bằng gộp frame.
2. **Peak VRAM có hai con số từ hai máy khác nhau.** Báo cáo ghi `2.91 GB (nvidia-smi)` = pilot 10 cửa
   sổ trên **RTX 3050**, và `2.52 GB (torch allocated)`; `ablation_A6.json` lưu `peak_vram_gb = 2.93` =
   benchmark trên **T4**. Bảng trong báo cáo đã ghi rõ "pilot 10 cửa sổ, không phải lượt 307 cửa sổ", nên
   khi trả lời phải nói kèm máy đo, đừng đọc "2.91" như thể là số của lượt chạy A6.

## 3. Phần chưa tái lập được (và lý do)

- **Khoảng tin cậy (CI)**: giá trị trong báo cáo khớp `ablation_*.json`, nhưng chưa chạy lại bootstrap
  1000 lần (seed 42). Muốn tái lập phải chạy `scripts/build_results.py` với đúng seed; số sẽ khớp nếu
  seed và dữ liệu y hệt.
- **Số liệu typing/hand-check** đã chạy lại và khớp, nhưng phụ thuộc file nhãn UCF **không có trong repo**
  (nằm ở máy có dataset). Trên máy không có dataset, script tự báo thiếu thay vì bịa số.

## 4. Ghi chú vận hành

- `scripts/audit_report_numbers.py` giữ bảng `CLAIMS` = các giá trị **đang in trong báo cáo**. Sửa số
  trong `report/` thì phải sửa `CLAIMS`, nếu không script sẽ báo FAIL — đó là mục đích của nó.
- Test đi kèm: `tests/vad_suite/test_audit_report_numbers.py` (8 test, chỉ kiểm phần toán học AUC và bộ
  đếm PASS/FAIL; không đọc dữ liệu thật nên CI không cần dataset).
