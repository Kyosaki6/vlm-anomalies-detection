# Tổng kết cải tiến dự án (Wave 1–4)

Nhánh `feat/project-improvements`, PR #6. Kế hoạch: `.omo/plans/project-improvements.md` (14 todo, 4 wave).
Số liệu đo ngày 23/10/2026 trên CPU (máy `fedora`), model `google/siglip2-base-patch16-384`; báo cáo đầy đủ: `notes/2026-10-23-eval-wave3.md`.

## 1. Đã sửa gì

| Wave | Việc | Kết quả kiểm chứng |
|---|---|---|
| 1 | `graph.py` đọc `events.json` bằng `utf-8-sig`; đường dẫn mặc định của `search.py`/`main.py` tính theo `__file__` | `python graph.py 1` chạy từ `/tmp` vẫn đúng |
| 1 | `search.py` ném `ImportError` thay vì `SystemExit(2)`; `eval.py` dùng `platform.node()` thay `COMPUTERNAME` | giả lập thiếu torch → `ImportError` |
| 1 | `main.run()` trả `PipelineResult` (dataclass bất biến); bình thường thì cách xử lý luôn là `Không cần xử lí…`; `raise_on_error` thay `sys.exit` | hết mâu thuẫn "BÌNH THƯỜNG" + "Báo 113" |
| 1 | `tests/test_main.py` 11 test mock model | chạy không cần tải SigLIP |
| 2 | `events.json` thêm `category`, `is_normal`, `severity`, `responder` (giữ nguyên 4 trường gốc) | 4 trường gốc giống hệt `main` |
| 2 | `graph.py` gắn thuộc tính lên node + 3 hàm truy vấn; `--list` hiện `[category \| severity]` | `test_graph_has_edge_event_to_action` vẫn xanh |
| 2 | `tests/test_graph.py` thêm 58 case | 108 test trong file |
| 3 | `search.py` thêm nhánh văn bản: `embed_texts`, `text_anomaly_score`, `fused_anomaly_score` | `search()`/`anomaly_score()` giữ nguyên chữ ký |
| 3 | `eval.py` mã hoá gallery một lần, cắt tensor (không copy file); `--legacy-tempdir` giữ cách cũ | top-3 và cờ hit giống hệt cách cũ; điểm bất thường lệch tối đa 1,2e-7 |
| 3 | `decision.py` thêm `bootstrap_ci`, `action_recall`; báo cáo `eval.py` có CI + so sánh nguồn điểm | `tests/test_decision.py` 12 → 30 test |
| 4 | `requirements.txt` đặt cận dưới, thêm `requirements-dev.txt`; CI GitHub Actions; README | xem dưới |

## 2. Số liệu mới

- **Tốc độ**: mã hoá 15 ảnh một lần ≈ 61s; chấm leave-one-out sau đó ≈ 0,002s/ảnh. Cách cũ ≈ 43s/ảnh (645s tổng). Đầu-cuối ≈ 10×.
  (Kế hoạch ước tính cách cũ ~8,7s/ảnh; máy này chậm hơn.)
- **Recall theo cách xử lý**: 6/15 = **0,400**, 95% CI [0,198; 0,667]. Recall@1 event-id vẫn 0/15 (hệ quả cấu trúc), recall@1 category 15/15.
- **Phân biệt bình thường/bất thường** (bootstrap 1000 lần, seed 42):

  | Nguồn điểm | AUC [95% CI] | F1 [95% CI] |
  |---|---|---|
  | ảnh | 0,960 [0,833; 1,000] | 0,947 [0,800; 1,000] |
  | văn bản (zero-shot) | 1,000 [1,000; 1,000] | 0,947 [0,800; 1,000] |
  | kết hợp 0,5/0,5 | 1,000 [1,000; 1,000] | 0,947 [0,800; 1,000] |

## 3. Cảnh báo — đọc trước khi trích dẫn

- **AUC văn bản = 1,0 gần với rò rỉ nhãn.** Hai prompt bất thường ("fighting", "car accident") gọi đúng tên hai nhóm trong bộ dữ liệu.
  Kết quả chỉ chứng minh nhánh văn bản hoạt động, không chứng minh nó tổng quát hoá sang loại bất thường khác.
  Vì thế `main.py` **chưa** dùng điểm văn bản/kết hợp (kế hoạch ghi "chỉ dùng nếu AUC kết hợp ≥ ảnh" — điều kiện thoả về số, nhưng không đáng tin).
- **n = 15**: khoảng tin cậy rất rộng; F1 bằng nhau giữa ba nguồn điểm không có nghĩa chúng tương đương.
- **Ngưỡng 0,30 vẫn là ngưỡng minh hoạ** chọn trên chính bộ dữ liệu này.
- **Action-level 6/15** bị chặn bởi dữ liệu: một số cặp sự kiện cùng nhóm có cách xử lý khác nhau.
- `severity` và `responder` trong `events.json` do nhóm gán thủ công khi làm Wave 2, không phải nhãn từ nguồn bên ngoài.

## 4. Phần chưa làm

- Chưa nối điểm văn bản vào `main.py` (lý do ở mục 3).
- Chưa có test đơn vị cho các hàm văn bản của `search.py` và các hàm mới của `eval.py` (`action_level`, `score_comparison`, `add_text_scores`); chỉ `decision.py` được phủ test.
- Workflow CI chưa chạy trên GitHub (viết và kiểm tra cú pháp YAML ở máy; cần push để chạy thật).
- Cận dưới phiên bản trong `requirements.txt` chưa được thử trên Python 3.10 thật (máy chỉ có 3.14 và 3.11; chỉ kiểm tra cú pháp 3.10).
