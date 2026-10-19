# VLM - Anomalies Detection

Phát hiện sự kiện bất thường từ ảnh camera bằng **vision-language embedding model
(SigLIP 2)** kết hợp **Knowledge Graph (NetworkX)**.

> **Nói đúng thuật ngữ:** SigLIP 2 là **embedding model** (ảnh → vector), không
> phải VLM sinh ngôn ngữ. Hệ thống không "hiểu" hay mô tả ảnh bằng câu; nó so
> vector để tìm sự kiện gần nhất. Tên repo giữ nguyên theo đề bài.

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
pip install -r requirements.txt
```

> Lần đầu chạy sẽ **tải model ~1.5 GB** (1,432 MiB — `google/siglip2-base-patch16-384`) từ HuggingFace về máy.
> Máy yếu: đổi trong `search.py` thành `google/siglip2-small-patch16-256`.

## Chạy

> Các lệnh dưới đây đã hoạt động trên máy thật (`search.py` + `graph.py` đã có trên `main`).

```bash
python search.py data/danh_nhau_1.jpg   # so sánh ảnh với 15 ảnh lưu sẵn
python graph.py 1                        # tra cách xử lý của sự kiện id = 1
python graph.py                          # in toàn bộ bảng Sự kiện → Cách xử lý
python main.py data/danh_nhau_1.jpg      # pipeline đầy đủ: ảnh → sự kiện → cách xử lý
python -m pytest -q                      # 62 test (graph + toán đánh giá)
```

`main.py` in thêm **kết luận bất thường / bình thường**:

- Điểm bất thường = `1 - max cosine` tới 5 ảnh **bình thường** (`duong_pho_*`).
- Mặc định ngưỡng `0.30`; đổi bằng `python main.py <ảnh> --threshold 0.25`.
- Ngưỡng này là **minh hoạ cho bộ 15 ảnh**, không phải ngưỡng tổng quát.

## Contract (nhóm đã chốt)

| File | Hàm | Trả về |
|------|-----|--------|
| `search.py` | `search(image_path, gallery_dir=None, model_name=None, device=None, topk=1)` *(đã có trên `main`)* | `topk=1`: `(file, score)` — ảnh khớp nhất + độ khớp 0..100 · `topk>1`: `[(file, score), ...]` |
| `graph.py` | `get_action(event_id)` | `str` — cách xử lý của sự kiện (hoặc `None` nếu không có) |
| `graph.py` | `get_event(event_id)` | `str \| None` — **tên** sự kiện (`ten_su_kien`), không phải dict (hoặc `None` nếu không có) |
| `search.py` | `anomaly_score(image_path, normal_prefix="duong_pho")` | `(max_similarity, anomaly_score)` — độ giống ảnh bình thường gần nhất + `1 - similarity` |
| `decision.py` | `binary_metrics` / `roc_auc` / `threshold_sweep` / `loo_threshold_eval` | chỉ số đánh giá nhị phân (TP/FP/TN/FN, P/R/F1/FPR, AUC) — có test |
| `main.py` | `python main.py <ảnh>` | ảnh → kết luận bất thường/bình thường → sự kiện gần nhất → cách xử lý |

## Cách đo & kết quả (19/10/2026, CPU)

Hai phép đo **khác nhau**, đừng lẫn:

1. **Truy hồi sự kiện** (`eval.py`, leave-one-out 15 ảnh):
   - recall@1 event-id = **0/15** — **hệ quả cấu trúc** (mỗi id đúng 1 ảnh, ảnh query bị loại khỏi gallery).
   - recall@1 category = **15/15** — luôn đúng nhóm (đánh nhau / tai nạn / đường phố).
2. **Bất thường vs bình thường** (điểm = `1 - cosine` tới 5 ảnh `duong_pho`):
   - **AUC = 0,960** (10 bất thường vs 5 bình thường).
   - Ngưỡng 0,30: TP=9 · FP=0 · TN=5 · FN=1 → precision 1,00 · recall 0,90 · F1 0,947 · FPR 0,00.
   - Chọn ngưỡng kiểu leave-one-out: F1 0,857 · accuracy 0,80 → **ngưỡng rất nhạy vì chỉ có 5 ảnh bình thường**.

Chi tiết + bảng từng ảnh: `notes/2026-10-19-eval-binary.md`. Dữ liệu: `notes/2026-10-19-data-card.md`.

## Giới hạn

- 15 ảnh tự thu thập từ web, **không phải CCTV**; 2 ảnh có watermark; 1 ảnh/sự kiện.
- n = 15 → **±1 ảnh = ±6,7%**, không có khoảng tin cậy; số liệu chỉ mang tính minh hoạ.
- Không đo được event-level (0/15 là hệ quả cấu trúc, không phải model yếu).
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

Schema `events.json` (mỗi dòng là 1 sự kiện):

```json
{"id": "1", "file": "danh_nhau_1.jpg", "ten_su_kien": "Đánh nhau ở bãi xe", "cach_xu_ly": "Báo bảo vệ cổng A và gọi cứu thương"}
```

## Phân công

- **Phước** — `search.py` (SigLIP 2 qua HuggingFace)
- **Thịnh** — `graph.py` (NetworkX: sự kiện → cách xử lý)
- **Huy** — `data/` + `events.json` ✅
- **Nam** — `main.py`, repo, tích hợp, báo cáo
- **Thành** — research: 3 bài báo AI phát hiện bất thường qua video

## License

MIT — © 2026 Kyosaki6, xem [LICENSE](LICENSE).