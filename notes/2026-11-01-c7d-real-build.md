# C7d — dựng lại `results/run_A4c/` với tier-2 thật (2026-11-01)

Lần chạy: `main` @ `45bca1b` + nhánh `feat/nam-c7d-real-build`, máy Nam (CPU, không GPU).

```powershell
python scripts/build_results.py --manifest manifests/ucf_test.csv `
    --calib-manifest manifests/ucf_calib_normal.csv --ablation A4c `
    --threshold 1.1686 --threshold-source calib_normal `
    --verify-dir results/verify --out-dir results/run_A4c
```

## Kết quả

| Chỉ số | Giá trị |
|---|---|
| File kết quả | 290 (`results/run_A4c/*.json`) + `windows/` 290 file |
| Cửa sổ tier-1 | 307 trong 114 video; 176 video còn lại không có cửa sổ |
| Tier-2 (Qwen2.5-VL-3B) | 129 xác nhận / 178 loại (42.0 %) |
| Cảnh báo | 101 trong 72 video |
| `metadata.note` khác rỗng | 0 (không file nào bị dashboard gắn nhãn "giả") |
| Dashboard đọc | 290 id, 0 placeholder, 101 cảnh báo, 114 video có tier-2 |

## Bảy mục kiểm của kế hoạch D4

1. **`windows/` so với bản đóng băng `results/windows/`:** *không thể* byte-equal — hai thư mục do hai
   writer khác nhau ghi (bản đóng băng không có `metadata.tier/tier2`, có `n_frames/fps`), và mỗi file mang
   `metadata.timestamp` của lần chạy. Kiểm theo nghĩa có nghĩa: `candidate_windows` của 114 video khớp
   mốc thời gian ở cả 114; 111/114 khớp cả `score` từng bit, 3 video (`Explosion007`, `RoadAccidents002`,
   `RoadAccidents010`) lệch **2 ULP (~2e-16)** — nhỏ hơn dung sai `SCORE_TOLERANCE = 1e-4` rất nhiều và
   không đổi mốc giây. Ngược lại, so với chính file C7c đã commit (`git HEAD`) thì **290/290 giống hệt**.
2. **`metadata.tier2` theo từng video:** 114 file `"json"`, 176 file `"none"` (khớp `metadata.tier`
   `"1+2"` / `"1"`). Trước đây trường này là của cả lần chạy nên 290 file đều ghi `"json"` — sửa trong
   `scripts/build_results.py` + test `test_tier2_metadata_speaks_per_video_not_per_run`.
3. **`metadata.note`:** rỗng ở cả 290 file (ngưỡng thật `calib_normal`).
4. **Cảnh báo chỉ từ cửa sổ `is_anomaly = True`:** dựng lại đúng pipeline cảnh báo chỉ từ 129 cửa sổ đã xác
   nhận cho ra đúng 101 cảnh báo như file đã ghi; 0 cảnh báo tham chiếu cửa sổ bị loại. 37 cửa sổ
   `is_anomaly = false` nhưng nhãn VLM khác `Normal` (theo `a6-protocol.md` §1) không sinh cảnh báo nào.
5. **Nhãn cảnh báo:** 101/101 lấy từ `ZeroShotEventTyper`; nhãn VLM giữ trong
   `metadata.event_typing.windows[].vlm_event_type`. 0 lệch.
6. **AUC tier-1 tính lại từ file mới:** 0.7299981359 (CI 0.6445330969–0.8115136756); bản đóng băng
   0.7299980646 (CI 0.6445330315–0.8115136017). Lệch 7.1e-8 do điểm bị làm tròn 6 chữ số khi ghi JSON —
   cùng lý do đã ghi ở `notes/2026-10-31-tier1-auc-recheck.md`. Ở 2 chữ số: **73.00 [64.45, 81.15]**.
7. **129 cửa sổ xác nhận → 101 cảnh báo:** chênh 28 do hai bước gộp của `vad/graph/alerts.py`:
   `merge_alerts` (cùng loại, cách nhau < 5 s) gộp 129 → 125, rồi `apply_cooldown` (30 s, cùng video +
   cùng loại) bỏ 24 → 101. Không cảnh báo nào bị mất vì lý do khác.

## Điều kiện kiểm này KHÔNG chứng minh

- Không chứng minh VLM đúng: 129 "xác nhận" là nhãn của mô hình, không phải người kiểm (việc đó là A6b).
- Không phải số A6: cắt cảnh báo/gộp/cooldown không đổi điểm frame, nên AUC ở mục 6 vẫn là AUC tier-1.
  Con số A6 (trước/sau tier-2) do `scripts/eval_tier2.py` sinh ra ở `results/ablation_A6.json`.
