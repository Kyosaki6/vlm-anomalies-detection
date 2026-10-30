# C7 — `scripts/build_results.py`: từ cache ra file kết quả dashboard đọc được (2026-10-30)

Trạng thái: **code + test xong; đã chạy end-to-end trên cache GIẢ trong thư mục tạm, CHƯA chạy trên
video/cache UCF thật** (máy Nam không có `cache/embeddings/` — cache nằm ở máy Thịnh/Colab).
`python -m pytest -q` → **525 passed, 1 skipped** (trước file này: 495 passed; 30 test mới). Không GPU,
không tải model, không cần file video.

**Sửa sau review của Thịnh (30/10):** hai lỗ đã đóng, mỗi lỗ có test viết TRƯỚC (chạy trên code cũ thấy
`DID NOT RAISE`), rồi mới sửa — xem bảng chốt chặn bên dưới. Bài học: "số lượng khớp" không phải "cặp
khớp"; muốn chắc thì phải so cả mốc thời gian.

## Vì sao có script này

`run_benchmark.py` trả lời *"ablation này được bao nhiêu AUC"* — nó không sinh ra thứ dashboard đọc.
`build_results.py` là mảnh nối cuối của đường ống:

```
cache/embeddings/*.npy  ->  score_cached_video (tier-1)  ->  cắt cửa sổ bằng NGƯỠNG
                        ->  (tuỳ chọn) kết quả tier-2     ->  build_alert_feed  ->  results/<video_id>.json
```

Không có bước này thì dashboard chỉ có `results/sample_run.json` (dữ liệu giả do `make_sample_result.py` sinh).

## Chạy

```bash
# hôm nay: 5 video, ngưỡng giả định, tier-2 bằng FakeVerifier (chỉ để dựng hình dạng file)
python scripts/build_results.py --manifest manifests/ucf_test.csv --ablation A1 \
    --threshold 0.5 --verifier fake --limit 5

# khi đã có tier-2 thật (A3 ghi JSON từng cửa sổ):
python scripts/build_results.py --manifest manifests/ucf_test.csv --ablation A3 \
    --threshold 2.5 --verify-dir results/tier2 --limit 20

# kiểm nhanh, không nạp model, không ghi file:
python scripts/build_results.py --manifest manifests/ucf_test.csv --dry-run
```

Dashboard đọc đúng thư mục này (`VAD_RESULTS_DIR` nếu muốn trỏ nơi khác). Mỗi video sinh **2 file**:
`results/<id>.json` và `results/windows/<id>.json` (file sau là đầu vào cho lần chạy tier-2 sau).

Mẫu log của lần chạy thử (cache giả, 3 video):

```
  Abuse028_x264                   40 mẫu    1 cửa sổ   1 cảnh báo  tier=1+2  [PLACEHOLDER]
  Normal_Videos586_x264           30 mẫu    0 cửa sổ   0 cảnh báo  tier=1    [PLACEHOLDER]
  RoadAccidents001_x264           50 mẫu    1 cửa sổ   1 cảnh báo  tier=1+2  [PLACEHOLDER]
```

## Ba thứ script này KHÔNG được phép làm (và cách nó bị chặn)

| Cám dỗ | Cách chặn |
|---|---|
| Tự chọn/đoán ngưỡng | `--threshold` là tham số bắt buộc-thủ-công; phải khai `--threshold-source`. `placeholder` → ghi `metadata.note`, dashboard hiện banner. `calib_normal` → **bắt buộc** kèm `--calib-manifest` làm chứng cứ. Không có lựa chọn nào lấy ngưỡng từ tập test. |
| Bịa tier-2 | Không có `--verifier fake` / `--verify-dir` thì `verify_results` và `alerts` rỗng, metadata ghi `tier="1"`. Dùng `--verifier fake` thì mọi file mang câu "TIER-2 GIẢ: … FakeVerifier sinh, KHÔNG phải VLM". |
| Bỏ qua video im lặng | Thiếu cache → một lỗi liệt kê **mọi** video thiếu. Cache lệch quá ±2 mẫu so với `n_frames/fps` → lỗi. Số kết quả tier-2 ≠ số cửa sổ → lỗi "lệch cặp". `video_id` kiểu đường dẫn (`../x`) → bị `vad/results.py` từ chối, không ghi ra ngoài `results/`. |
| Ghép tier-2 sai cửa sổ | `check_pairs_match`: từng cặp phải trùng `start_s`/`end_s` (dung sai 1e-3 s). Cùng số cửa sổ nhưng khác mốc giây → lỗi. |
| Nhận file tier-2 rỗng | `tier2_source="json"` mà video có cửa sổ nhưng 0 kết quả → lỗi, để metadata không ghi `tier2=json` khi thực ra không có tier-2. |

Chi tiết thiết kế đáng nhớ:

- **`metadata.note` rỗng = file này là kết quả thật.** `app.data.is_placeholder()` đọc đúng nó. Vì thế
  ngưỡng đã calibrate **không** ghi gì vào `note`; xuất xứ nằm ở `metadata.threshold_source` +
  `metadata.calib_manifest`. Nếu ghi xuất xứ vào `note`, dashboard sẽ dán nhãn "giả" lên kết quả thật.
- `verify_windows()` truyền `frames=[]`: chỉ đúng cho `FakeVerifier`/replay. VLM thật cần khung hình giải
  mã từ video — việc đó thuộc `scripts/run_verify.py` (A3), nó ghi JSON rồi ta đọc lại bằng `--verify-dir`.
  Hàm để trống chữ `frames=[]` có chủ ý, kèm docstring, để không ai tưởng đường này chạy được với VLM thật.
- Ghép cửa sổ ↔ kết quả tier-2 theo **thứ tự và số lượng**: cùng ngưỡng thì cùng số cửa sổ. Đổi ngưỡng mà
  không chạy lại tier-2 → lỗi ngay, không gán nhầm nhãn cho cửa sổ khác.
- Dùng lại helper của `run_benchmark.py` (`read_benchmark_manifest`, `select_rows`,
  `load_or_embed_prompts`, `expected_samples`, `git_commit`) thay vì chép lại: cache prompt
  `cache/prompts/*.npz` dùng chung, nên benchmark và build_results không thể tính prompt khác nhau.
- `infer_ablation(config)` suy ra `A0..A5` từ config để file kết quả ghi được tên preset.

## Bàn giao cho Thịnh (dashboard)

1. Chạy lệnh đầu tiên ở trên khi máy có `cache/embeddings/` (hoặc trỏ `--cache-dir` vào cache Colab tải về).
2. `streamlit run app/streamlit_app.py` (hoặc `VAD_RESULTS_DIR=<đường dẫn>`) → chọn một video trong danh
   sách, xác nhận timeline + cửa sổ + cảnh báo hiện đúng, và banner "note" hiện với file placeholder.
3. **Chưa commit file kết quả sinh ra vào repo.** Hiện `results/` chỉ nên chứa `sample_run.json` (giả, tự
   khai) và `ablation_A*.json` (số đo thật). Nếu muốn commit 1-2 video làm mẫu thì nên là video đã chạy
   thật và giữ nguyên `note` nếu ngưỡng chưa calibrate — quyết định của Thịnh.

## Còn thiếu (không nằm trong C7)

- Chạy trên cache thật → chỉ khi có dataset/cache.
- Ngưỡng thật từ `calib_normal` (TH5). Hiện **mọi** đường chạy thật đều phải đi qua `placeholder`.
- `scripts/run_verify.py` (A3, Huy) để có tier-2 thật thay cho `FakeVerifier`.
- `logit_scale=100` vẫn là mặc định trong khi SigLIP 2 thật trả **112.85** (biết từ 29/10, chưa sửa).
  Sigmoid là hàm tăng nên **thứ hạng** khung hình không đổi, nhưng `vad/score/text.py` cắt biên ở ±60
  trước sigmoid: scale lớn hơn thì nhiều khung chạm trần 1.0 → hoà nhau → AUC dịch một chút. Việc của TH3/TH5.
