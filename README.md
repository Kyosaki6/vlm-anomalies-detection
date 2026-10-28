# VLM - Anomalies Detection

Phát hiện sự kiện bất thường từ ảnh camera bằng **vision-language embedding model
(SigLIP 2)** kết hợp **Knowledge Graph (NetworkX)**.

> **Nói đúng thuật ngữ:** SigLIP 2 là **dual-encoder** (một nhánh ảnh, một nhánh văn bản,
> cùng nằm trong một không gian vector), không phải VLM sinh ngôn ngữ. Hệ thống không
> "hiểu" hay mô tả ảnh bằng câu. Nhánh **ảnh** dùng để tìm sự kiện gần nhất và chấm điểm
> bất thường so với ảnh bình thường; nhánh **văn bản** cho phép chấm **zero-shot** bằng
> prompt (xem mục "Nhánh văn bản"). Tên repo giữ nguyên theo đề bài.

Khi có một ảnh mới từ camera, hệ thống tìm **sự kiện cũ giống nhất** trong kho dữ liệu rồi tra ra **cách xử lý** đã được ghi nhận cho sự kiện đó.

```
Ảnh mới ──> search.py (SigLIP 2 embedding + cosine) ──> (tên file khớp nhất, độ khớp)
                                                          │
                                              events.json (file → id) │
                                                              ▼
                                          graph.py (NetworkX) ──> "Cách xử lý: Báo bảo vệ cổng A, gọi cứu thương"
```

## Cài đặt

```bash
pip install -r requirements.txt        # chạy hệ thống
pip install -r requirements-dev.txt    # thêm pytest để chạy test
```

Cần Python **3.10+** (code dùng kiểu `X | None`). `requirements*.txt` chỉ đặt cận dưới phiên bản;
`transformers>=4.49` vì SigLIP 2 chỉ có từ bản đó.

> Lần đầu chạy sẽ **tải model ~1.5 GB** (1,432 MiB — `google/siglip2-base-patch16-384`) từ HuggingFace về máy.
> Máy yếu: đổi trong `search.py` thành `google/siglip2-small-patch16-256`.

## Chạy

> Các lệnh dưới đây đã hoạt động trên máy thật (`search.py` + `graph.py` đã có trên `main`).

```bash
python search.py data/danh_nhau_1.jpg   # so sánh ảnh với 15 ảnh lưu sẵn
python graph.py 1                        # tra cách xử lý của sự kiện id = 1
python graph.py                          # in toàn bộ bảng Sự kiện → Cách xử lý
python main.py data/danh_nhau_1.jpg      # pipeline đầy đủ: ảnh → sự kiện → cách xử lý
python -m pytest -q                      # 149 test (graph + pipeline + toán đánh giá), không cần tải model
```

`main.py` in thêm **kết luận bất thường / bình thường**:

- Điểm bất thường = `1 - max cosine` tới 5 ảnh **bình thường** (`duong_pho_*`).
- Mặc định ngưỡng `0.30`; đổi bằng `python main.py <ảnh> --threshold 0.25`.
- Ngưỡng này là **minh hoạ cho bộ 15 ảnh**, không phải ngưỡng tổng quát.
- Kết luận và cách xử lý **luôn nhất quán**: ảnh BÌNH THƯỜNG thì cách xử lý là
  `Không cần xử lí, chỉ quan sát và ghi nhận`; sự kiện gần nhất chỉ in để tham khảo.

### Dùng như thư viện: `PipelineResult`

`main.run()` trả về `PipelineResult` (dataclass bất biến) thay vì gọi `sys.exit()`, nên gọi được từ code khác:

```python
import main

res = main.run("data/danh_nhau_1.jpg", raise_on_error=True, verbose=False)
res.is_abnormal, res.anomaly_score, res.threshold   # True, 0.317, 0.30
res.event_id, res.event_name, res.action            # sự kiện + cách xử lý (None nếu bình thường)
res.reference_event_id, res.reference_event_name    # sự kiện gần nhất, chỉ để tham khảo khi bình thường
```

`raise_on_error=True` ném `FileNotFoundError` / `ValueError`; mặc định (`False`) in lỗi và trả `None`, đúng hành vi CLI cũ.

### Nhánh văn bản (zero-shot)

Nhánh văn bản của SigLIP 2 chấm ảnh bằng prompt, **không cần gallery**:

```python
import search

search.text_anomaly_score("anh.jpg")    # (cosine tới prompt bình thường, cosine tới prompt bất thường)
search.fused_anomaly_score("anh.jpg")   # dict: image_score, text_score, fused_score, ...
```

Prompt mặc định nằm trong `search.NORMAL_PROMPT` / `search.ABNORMAL_PROMPTS`. `text_score` là xác suất
"bất thường" (softmax 2 lớp trên cosine); `fused_score = 0.5·ảnh + 0.5·văn bản`.

> ⚠️ Hai prompt bất thường gọi đúng tên hai nhóm trong bộ dữ liệu (đánh nhau, tai nạn) nên AUC văn bản = 1,0 **gần với rò rỉ nhãn**.
> Nó chứng minh nhánh văn bản chạy được, **không** chứng minh tổng quát hoá sang loại bất thường khác.
> Vì vậy `main.py` vẫn chỉ dùng điểm ảnh.

## Contract (nhóm đã chốt)

| File | Hàm | Trả về |
|------|-----|--------|
| `search.py` | `search(image_path, gallery_dir=None, model_name=None, device=None, topk=1)` *(đã có trên `main`)* | `topk=1`: `(file, score)` — ảnh khớp nhất + độ khớp 0..100 · `topk>1`: `[(file, score), ...]` |
| `graph.py` | `get_action(event_id)` | `str` — cách xử lý của sự kiện (hoặc `None` nếu không có) |
| `graph.py` | `get_event(event_id)` | `str \| None` — **tên** sự kiện (`ten_su_kien`), không phải dict (hoặc `None` nếu không có) |
| `search.py` | `anomaly_score(image_path, normal_prefix="duong_pho")` | `(max_similarity, anomaly_score)` — độ giống ảnh bình thường gần nhất + `1 - similarity` |
| `decision.py` | `binary_metrics` / `roc_auc` / `threshold_sweep` / `loo_threshold_eval` | chỉ số đánh giá nhị phân (TP/FP/TN/FN, P/R/F1/FPR, AUC) — có test |
| `search.py` | `text_anomaly_score(image_path_or_emb, ...)` / `fused_anomaly_score(image_path, alpha=0.5, ...)` | `(normal_sim, abnormal_sim)` / `dict` điểm ảnh + văn bản + kết hợp |
| `graph.py` | `get_event_by_id` / `get_events_by_category` / `get_action_details` | dict đầy đủ metadata (category, severity, responder) hoặc `None` |
| `decision.py` | `bootstrap_ci` / `action_recall` | khoảng tin cậy bootstrap · tỉ lệ đúng cách xử lý |
| `main.py` | `python main.py <ảnh>` · `main.run(...)` | ảnh → kết luận bất thường/bình thường → sự kiện gần nhất → cách xử lý · `PipelineResult` |

## Cách đo & kết quả (23/10/2026, CPU)

Hai phép đo **khác nhau**, đừng lẫn:

1. **Truy hồi sự kiện** (`eval.py`, leave-one-out 15 ảnh):
   - recall@1 event-id = **0/15** — **hệ quả cấu trúc** (mỗi id đúng 1 ảnh, ảnh query bị loại khỏi gallery).
   - recall@1 category = **15/15** — luôn đúng nhóm (đánh nhau / tai nạn / đường phố).
   - **recall theo cách xử lý (action-level) = 6/15 = 0,400** (95% CI [0,198; 0,667]) — hai sự kiện khác id nhưng cùng cách xử lý
     vẫn tính là đúng về mặt vận hành. Con số này mới phản ánh được việc "đưa ra đúng hành động", thay cho 0/15 vô nghĩa.
2. **Bất thường vs bình thường** (điểm = `1 - cosine` tới 5 ảnh `duong_pho`):
   - **AUC = 0,960** (10 bất thường vs 5 bình thường).
   - Ngưỡng 0,30: TP=9 · FP=0 · TN=5 · FN=1 → precision 1,00 · recall 0,90 · F1 0,947 · FPR 0,00.
   - Chọn ngưỡng kiểu leave-one-out: F1 0,857 · accuracy 0,80 → **ngưỡng rất nhạy vì chỉ có 5 ảnh bình thường**.
   - Khoảng tin cậy bootstrap 95% (1000 lần, seed 42): AUC ảnh = 0,960 [0,833; 1,000]; F1 = 0,947 [0,800; 1,000].
3. **So sánh nguồn điểm** (cùng bộ 15 ảnh, ngưỡng ảnh 0,30 · văn bản/kết hợp 0,50):

   | Nguồn điểm | AUC | F1 |
   |---|---|---|
   | ảnh (so ảnh bình thường) | 0,960 | 0,947 |
   | văn bản (zero-shot) | 1,000* | 0,947 |
   | kết hợp 0,5/0,5 | 1,000* | 0,947 |

   \* Xem cảnh báo rò rỉ prompt ở mục "Nhánh văn bản". Với n = 15, F1 giống nhau không có nghĩa ba nguồn tương đương.

`eval.py` mã hoá mỗi ảnh gallery **một lần** rồi cắt tensor trong RAM (không copy file): trên CPU ~61s để mã hoá
15 ảnh, sau đó chấm leave-one-out ~0,002s/ảnh (cách cũ `--legacy-tempdir`: ~43s/ảnh, kết quả giống hệt).

Chi tiết + bảng từng ảnh: `notes/2026-10-19-eval-binary.md`, `notes/2026-10-23-eval-wave3.md`. Dữ liệu: `notes/2026-10-19-data-card.md`. Tổng kết các thay đổi: `notes/2026-10-22-project-improvements.md`.

## Giới hạn

- 15 ảnh tự thu thập từ web, **không phải CCTV**; 2 ảnh có watermark; 1 ảnh/sự kiện.
- n = 15 → **±1 ảnh = ±6,7%**; khoảng tin cậy bootstrap rất rộng (xem trên); số liệu chỉ mang tính minh hoạ.
- Không đo được event-level (0/15 là hệ quả cấu trúc, không phải model yếu); action-level 6/15 bị chặn bởi việc nhiều sự kiện có cách xử lý khác nhau dù cùng nhóm.
- Ngưỡng bất thường chọn trên chính bộ dữ liệu này → **không dùng được cho camera thật** khi chưa hiệu chỉnh.
- Embedding toàn ảnh, không có patch/local feature → khó phát hiện vật thể nhỏ.
- Không có video/thời gian; chưa phải temporal anomaly detection.
- Không so với SOTA (MNAD/DMAD/Jigsaw-VAD đo AUC trên video, bài toán khác).

## Dữ liệu

| Thành phần | Trạng thái |
|------------|------------|
| `data/` — 15 ảnh: 5 đánh nhau, 5 tai nạn xe, 5 đường phố bình thường | ✅ (Huy) |
| **`events.json` — 15 sự kiện thật** (id/file/ten_su_kien/cach_xu_ly) | ✅ (Huy) |

> ⚠️ **Lưu ý Windows:** sửa `events.json` hãy lưu kiểu **UTF-8 (không BOM)** — Notepad/Excel/PowerShell hay chèn thêm 3 byte BOM ẩn vào đầu file, làm crash mọi reader nghiêm ngặt (Python `json`, jq, Node…).

Schema `events.json` (mỗi phần tử là 1 sự kiện; bốn trường đầu là gốc, bốn trường sau thêm ở Wave 2):

```json
{"id": "1", "file": "danh_nhau_1.jpg", "ten_su_kien": "...", "cach_xu_ly": "Báo công an phường và lập biên bản vụ việc",
 "category": "danh_nhau", "is_normal": false, "severity": "cao", "responder": ["cong_an"]}
```

`severity` ∈ `thap` · `trung_binh` · `cao`; `responder` là danh sách (có thể rỗng). `graph.py` gắn các trường này lên node sự kiện.

## Kiểm thử & CI

- `pytest -q` chạy **149 test không cần tải model** (mock `search`/`anomaly_score`); mất ~15s chủ yếu để import `torch`.
- `.github/workflows/ci.yml` chạy `tests/test_graph.py`, `test_decision.py`, `test_main.py` trên Ubuntu + Windows × Python 3.10 / 3.11.
  CI vẫn cài `torch` (bản CPU trên Linux) vì `main.py` import `search.py`; chỉ **không tải trọng số SigLIP 2**.

## Phân công

- **Phước** — `search.py` (SigLIP 2 qua HuggingFace)
- **Thịnh** — `graph.py` (NetworkX: sự kiện → cách xử lý)
- **Huy** — `data/` + `events.json` ✅
- **Nam** — `main.py`, repo, tích hợp, báo cáo

> Giai đoạn video (`vad/`): deadline nộp **trước 04/11/2026** (hạn chót cuối ngày 03/11/2026 23:59), nhóm 4 người; Thịnh làm dashboard Streamlit.
> Xem `notes/2026-10-29-task-assignment.md`.

## License

MIT — © 2026 Kyosaki6, xem [LICENSE](LICENSE).