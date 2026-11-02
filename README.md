# Zero-Shot Video Anomaly Detection & Surveillance Alerting

Phát hiện bất thường trong video giám sát **không cần huấn luyện** (training-free) và biến phát hiện đó
thành **cảnh báo có quy trình xử lý**: SigLIP 2 nhúng từng khung hình → chấm điểm bằng prompt tổng quát →
lọc theo thời gian → **Generative VLM kiểm chứng** từng cửa sổ nghi vấn → Alert Policy Graph → dashboard
Streamlit. Đồ án cơ sở KHTN, 4 thành viên, nộp 04/11/2026.

> **Đọc đúng thuật ngữ:** SigLIP 2 là **dual-encoder**, không phải VLM sinh ngôn ngữ. Nó chỉ cho một
> vector cho mỗi khung hình. Câu mô tả sự việc ("người đang đánh nhau") do **Qwen2.5-VL-3B** sinh ra ở
> tầng 2, và chỉ cho các cửa sổ tầng 1 đã nghi vấn. Tên repo giữ nguyên theo đề bài.

## TL;DR kết quả (UCF-Crime test, 290 video, τ = 1.1686)

| Chỉ số | Tầng 1 (A4c) | Sau kiểm chứng VLM (A6) |
|---|---:|---:|
| ROC-AUC micro (frame-level) [CI 95% theo video] | **73.00** [64.45, 81.15] | 67.90 [58.39, 77.49] |
| Báo động giả / giờ (150 video bình thường) | 20.64 | **0.17** |
| Precision cửa sổ (giao nhãn mặt đất) | 28.0% (86/307) | **49.6%** (64/129) |
| Incident recall | **51.9%** (81/156) | 39.7% (62/156) |

Đọc bảng này đúng cách: tầng 2 **đổi báo động giả lấy recall**. Nó dập 99% báo động giả và nâng precision
cửa sổ lên gần gấp đôi, nhưng làm mất 22 cửa sổ có thật, nên AUC giảm 5.10 điểm (hiệu cặp CI 95%
[-11.42, -0.25] — giảm có ý nghĩa thống kê sát biên). Đây là kết quả chính, không phải lỗi cần che.

Dữ liệu đã tính sẵn trong repo: 290 kết quả tầng 1 (`results/run_A4c/`), 307 cửa sổ đã được VLM thật
kiểm chứng (`results/verify/`, 129 xác nhận), **101 cảnh báo trong 72 video**, 5 hình cho báo cáo
(`results/figures/`). Tất cả là `fake=false`, có thể kiểm tra lại từng file.

## Hệ thống chạy thế nào

```
Video ──> lấy mẫu 1 fps ──> SigLIP 2 (cache .npy)
                              │
        ┌─────────────────────┴─────────────────────┐
        │ TẦNG 1 (rẻ, chạy mọi khung hình)          │
        │ score = prompt tổng quát + normal memory  │
        │ robust_normalize (median/MAD calib_normal)│
        │ top-k pooling + EMA + median filter        │
        │ candidate window (nối các mẫu > τ, gap 5s) │
        └─────────────────────┬─────────────────────┘
                              │ chỉ những cửa sổ này (307/290 video)
        ┌─────────────────────┴─────────────────────┐
        │ TẦNG 2 (đắt, chỉ chạy trên cửa sổ)        │
        │ Qwen2.5-VL-3B-Instruct 4-bit, 4 khung/chunk│
        │ is_anomaly = OR trên các chunk (quy tắc D-1)│
        └─────────────────────┬─────────────────────┘
                              │ 129 xác nhận / 178 bị loại
        Typer zero-shot (13 lớp) ──> Alert Policy Graph (YAML → NetworkX)
                              │
                    Dashboard Streamlit (replay)
```

Ba nguyên tắc thiết kế được giữ bằng test, không chỉ bằng lời:

1. **Không rò rỉ nhãn:** prompt phát hiện chỉ dùng câu tổng quát ("an unusual event is happening"), không
   bao giờ chứa tên lớp ("fighting", "explosion"). Tên lớp chỉ xuất hiện ở bước gọi tên cuối cùng.
2. **Không hiệu chỉnh trên tập test:** ngưỡng τ = 1.1686 lấy từ phân vị 95% của 60 video `calib_normal`.
3. **Cửa sổ bị VLM loại thì không bao giờ thành cảnh báo:** chặn hai lần (build + policy), có test khóa.

## Cài đặt và chạy

```bash
pip install -r requirements-dev.txt        # test, CPU, không tải model
pip install -r requirements-app.txt        # thêm streamlit để mở dashboard

python -m pytest -q                       # 859 passed, 1 skipped (~110 s), không cần GPU
python -m streamlit run app/streamlit_app.py   # dashboard đọc results/run_A4c/ (dữ liệu thật)
```

Dashboard mở tab **Replay** (chọn video, xem timeline điểm theo thời gian, cửa sổ tô màu theo verdict,
danh sách cảnh báo kèm lực lượng ứng cứu) và tab **Overview** (bảng ablation, điểm vận hành A6, chi phí).

Kiểm tra nhanh máy có chạy được mô hình không (tải ~1.5 GB lần đầu):

```bash
python scripts/sanity_embedder.py
```

> ⚠️ Đừng chạy `streamlit run app/streamlit_app.py -- --demo`. Cờ đó hiển thị **dữ liệu tổng hợp**
> (điểm ngẫu nhiên, verifier giả) để thử giao diện, và app tự in cảnh báo "DEMO DATA". Khi trình diễn,
> chạy lệnh trơn như trên, hoặc trỏ `VAD_RESULTS_DIR=results/run_demo` cho 7 clip demo.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `vad/` | Thư viện lõi: `types.py`/`interfaces.py` (hợp đồng), `embed/` (SigLIP 2 + CLIP, cache), `score/`, `temporal/`, `verify/` (parser JSON + VLMVerifier), `graph/` (policy), `eval/` (AUC, bootstrap theo video, FA/hour), `datasets/` (GT UCF-Crime, sharding) |
| `scripts/` | `precompute_embeddings.py` (cache, chia shard), `run_benchmark.py`, `calibrate_threshold.py`, `run_verify.py` (tầng 2), `build_results.py` (RunResult + cảnh báo), `eval_tier2.py`, `eval_vlm_typing.py`, `make_figures.py` |
| `app/` | `data.py` (đọc RunResult, không import streamlit), `streamlit_app.py` |
| `configs/` | `prompts.yaml` (prompt phát hiện/gọi tên), `policy.yaml` (zone, khung giờ, leo thang) |
| `manifests/` | `ucf_test.csv` (290 video), `ucf_calib_normal.csv` (60), `own_demo*.csv` (7 clip demo) |
| `results/` | Kết quả đã tính sẵn: `ablation_A*.json`, `run_A4c/` (290 + windows), `verify/` (307), `figures/`, `vlm_typing_eval.json` |
| `report/` | Báo cáo tiếng Việt 6 chương + phụ lục + tài liệu tham khảo |
| `notes/` | Nhật ký thí nghiệm, giao thức đăng ký trước, phân tích lỗi |
| `tests/` | `tests/vad_suite/` — test của hệ thống mới; `tests/` gốc là test của pipeline cũ |

Trạng thái từng module và ai phụ trách: `vad/README.md`. Kế hoạch và quyết định: `notes/2026-10-28-plan-zero-shot-video-vad.md`.

## Kết quả chi tiết

Bảng ablation đầy đủ nằm ở `report/04-thuc-nghiem.md` §4.2 (nguồn: `results/ablation_A*.json`). Tóm tắt
đường đi: A0 CLIP B/32 **48.77** → A1 SigLIP 2 **65.87** → A2 thêm prompt ensemble **72.39** → A3/A4/A5
chuẩn hóa theo video làm micro AUC **giảm** (58.99 / 58.41 / 57.08) → A3c/A4c dùng thống kê cố định từ
`calib_normal` trở lại **73.00**, A4c là cấu hình đóng băng → A6 thêm VLM **67.90**.

Hai kết quả âm được báo cáo nguyên trạng: normal memory bank (A5c) giảm 8.1 điểm so với A4c, và chuẩn hóa
theo từng video (A3) làm mất thông tin so sánh giữa các video.

Cổng kiểm chứng A0 cũng **không đạt như kỳ vọng**: tái lập ZS-CLIP của LAVAD cho 48.77 (CI [38.30, 59.40]),
thấp hơn mốc 53.16 và nằm ngoài dải chấp nhận 50.2–56.2 đã đặt trước khi chạy. Mốc của bài báo vẫn nằm
trong CI. Chi tiết các phép kiểm tra sai sót (đảo nhãn, dịch nhãn, stride, logit scale): `notes/a0-gate.md`.

Gọi tên sự kiện (C7e, 62 cửa sổ vừa được VLM xác nhận vừa giao nhãn mặt đất): VLM **83.9%** (52/62) so với
typer zero-shot **24.2%** (15/62) và mức đoán mò 7.7%; typer không chứng minh được là hơn baseline luôn
đoán "Shooting" (25.8%). Mẫu này lệch lớp (73% là 4 lớp), nên **không** được trích thành "VLM gọi tên đúng
84% mọi sự cố". Nhãn cảnh báo vẫn do typer quyết định vì tính tất định (quyết định D-2), không phải vì nó
chính xác hơn.

## Hạn chế phải nói khi bảo vệ

- **Tầng 2 không miễn phí:** đổi 99% báo động giả lấy 22 sự cố thật bị loại (recall 51.9% → 39.7%).
- **Không có véc-tơ chuyển động:** đặc trưng từng khung hình 2D, nên hành vi "trông tĩnh" (đi bộ vs bỏ chạy)
  khó phân biệt.
- **Chỉ UCF-Crime:** ShanghaiTech không hoàn tất (thiếu thời gian); báo cáo ghi rõ đây là phạm vi mở rộng.
- **VLM không tái lập 100%:** chạy lại 10 cửa sổ cho 9/10 verdict giống nhau; nguyên nhân chưa xác định.
- **37/307 cửa sổ VLM tự mâu thuẫn** (`is_anomaly=false` nhưng `event_type` khác Normal); theo giao thức
  đăng ký trước, verdict chỉ lấy `is_anomaly` nên chúng tính là bị loại.
- **Replay mode:** dashboard đọc kết quả JSON tính sẵn, không nối luồng camera RTSP thời gian thực.

## Clip demo

7 clip ngắn tải từ iStock (bản preview có watermark, **chưa mua licence**) để minh họa giao diện. Kết quả
chạy thật nằm ở `results/run_demo/` khi đã merge nhánh E4b. Các clip này **không tham gia bất kỳ số liệu
benchmark nào** và `metadata.note` của chúng ghi rõ "demo, not benchmark".

## Tái lập

```bash
# 0) dữ liệu: tải UCF-Crime test + sinh manifest (xem scripts/download_ucf_crime.py)
# 1) nhúng khung hình (chia shard i/N, chạy được --dry-run trước)
python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 1/4

# 2) kiểm chuẩn tầng 1
python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A4c \
    --calib-manifest manifests/ucf_calib_normal.csv --out-dir results

# 3) hiệu chỉnh ngưỡng trên calib_normal (KHÔNG dùng tập test)
python scripts/calibrate_threshold.py --ablation A4c --calib-manifest manifests/ucf_calib_normal.csv

# 4) tầng 2 trên các cửa sổ ứng viên
python scripts/run_verify.py --manifest manifests/ucf_test.csv --windows-dir results/windows \
    --out-dir results/verify --k 4 --verifier-model Qwen/Qwen2.5-VL-3B-Instruct

# 5) dựng RunResult + cảnh báo cho dashboard
python scripts/build_results.py --manifest manifests/ucf_test.csv \
    --calib-manifest manifests/ucf_calib_normal.csv --ablation A4c --threshold 1.1686 \
    --threshold-source calib_normal --verify-dir results/verify --out-dir results/run_A4c

# 6) đánh giá, hình, gọi tên
python scripts/eval_tier2.py --manifest manifests/ucf_test.csv --verify-dir results/verify
python scripts/eval_vlm_typing.py
python scripts/make_figures.py
```

Các lệnh trên đã chạy thật để sinh `results/`; số liệu trong báo cáo lấy trực tiếp từ các file JSON đó,
không chép tay.

## Báo cáo và tài liệu

- `report/README.md` — cấu trúc báo cáo, trạng thái từng chương, lệnh xuất PDF/DOCX bằng pandoc.
- `notes/a6-protocol.md` — giao thức đánh giá tầng 2, commit **trước** khi có verdict.
- `notes/a6-handcheck.md` — kiểm tay 20 cửa sổ (precision 6/10, mẫu nhỏ, không kiểm được bỏ sót).
- `notes/a7-fp-analysis.md` — phân tích 5 lost true positive và 1 báo động giả sống sót.
- `notes/own-clips.md` — bằng chứng chạy 7 clip demo.

## Pipeline cũ (15 ảnh) — không phải bằng chứng cho hệ thống này

`main.py`, `search.py`, `graph.py`, `decision.py`, `data/`, `events.json` là pipeline ảnh tĩnh của giai
đoạn trước (15 ảnh, truy hồi sự kiện + Knowledge Graph). Nó còn trong repo làm fixture chạy thử và để
`decision.py` tái sử dụng phần toán đánh giá, nhưng **không** dùng để chứng minh bất kỳ kết quả video nào.
Kết quả video nằm ở `results/` và `report/04-thuc-nghiem.md`.

## License

MIT — © 2026 Kyosaki6, xem [LICENSE](LICENSE).
