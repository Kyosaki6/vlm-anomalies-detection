# VLM - Anomalies Detection

Phát hiện sự kiện bất thường từ camera bằng **Vision-Language Model (SigLIP 2)** kết hợp **Knowledge Graph (NetworkX)**.

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

> Lần đầu chạy sẽ **tải model ~400 MB** (`google/siglip2-base-patch16-384`) từ HuggingFace về máy.
> Máy yếu: đổi trong `search.py` thành `google/siglip2-small-patch16-256`.

## Chạy

> Các lệnh dưới đây đã hoạt động trên máy thật (`search.py` + `graph.py` đã có trên `main`).

```bash
python search.py data/danh_nhau_1.jpg   # so sánh ảnh với 15 ảnh lưu sẵn
python graph.py 1                        # tra cách xử lý của sự kiện id = 1
python graph.py                          # in toàn bộ bảng Sự kiện → Cách xử lý
python main.py data/danh_nhau_1.jpg      # pipeline đầy đủ: ảnh → sự kiện → cách xử lý
```

## Contract (nhóm đã chốt)

| File | Hàm | Trả về |
|------|-----|--------|
| `search.py` | `search(image_path)` *(cần thêm — hiện là CLI)* | `(file, score)` — tên ảnh khớp nhất + độ khớp 0..100 |
| `graph.py` | `get_action(event_id)` | `str` — cách xử lý của sự kiện (hoặc `None` nếu không có) |
| `graph.py` | `get_event(event_id)` | `dict` — thông tin sự kiện (hoặc `None`) |
| `main.py` | `python main.py <ảnh>` | ảnh → file khớp → `events.json` → sự kiện + cách xử lý |

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