"""
main.py — PIPELINE ĐẦY ĐỦ: Ảnh vào → sự kiện cũ giống nhất → CÁCH XỬ LÝ.

Người phụ trách: Nam (ENG-04). Tích hợp code của:
    - Phước  : search.py     — tìm ảnh giống nhất, trả về (tên file, độ khớp %)
    - Huy    : events.json   — map tên file → event id
    - Thịnh  : graph.py      — tra tên sự kiện + cách xử lý từ NetworkX

Cách chạy:
    python main.py                       # mặc định thử với data/danh_nhau_1.jpg
    python main.py data/tai_nan_xe_1.jpg
    python main.py <ảnh bất kỳ.jpg>
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import graph
from decision import DEFAULT_ANOMALY_THRESHOLD, decide_abnormal
from search import search, anomaly_score

# Windows console: cho phép in tiếng Việt đúng chuẩn
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE = os.path.join(ROOT, "events.json")
NORMAL_ACTION = "Không cần xử lí, chỉ quan sát và ghi nhận"


@dataclass(frozen=True)
class PipelineResult:
    image_path: str
    is_abnormal: bool
    anomaly_score: float
    threshold: float
    normal_similarity: float
    best_file: str
    score: float | int
    event_id: str | None
    event_name: str | None
    action: str
    reference_event_id: str | None
    reference_event_name: str | None


def load_file_to_id(path: str = EVENTS_FILE, raise_on_error: bool = False) -> dict | None:
    """Đọc events.json → map {tên file: event id}.

    Dùng utf-8-sig để chịu được cả file bị thêm BOM (Windows Notepad/Excel hay thêm).
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        if raise_on_error:
            raise FileNotFoundError(f"Không tìm thấy {path}")
        print(f"[main.py] Lỗi: không tìm thấy {path} — Huy đã đẩy events.json lên chưa?")
        return None
    with open(path, encoding="utf-8-sig") as f:
        events = json.load(f)
    return {ev["file"]: str(ev["id"]) for ev in events}


def run(
    image_path: str,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    raise_on_error: bool = False,
    verbose: bool = True,
) -> PipelineResult | None:
    # 0) kiểm tra ảnh đầu vào
    if not os.path.exists(image_path) and not os.path.isabs(image_path):
        candidate = os.path.join(ROOT, image_path)
        if os.path.exists(candidate):
            image_path = candidate

    if not os.path.exists(image_path):
        if raise_on_error:
            raise FileNotFoundError(f"Không thấy ảnh '{image_path}'.")
        print(f"[main.py] Lỗi: không thấy ảnh '{image_path}'.")
        return None

    # 1) tìm ảnh giống nhất trong bộ sưu tập — hàm search() của Phước (search.py)
    try:
        best_file, score = search(image_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        if raise_on_error:
            raise
        print(f"[main.py] Lỗi khi tìm ảnh: {e}")
        return None

    # 2) điểm bất thường: 1 - độ giống ảnh BÌNH THƯỜNG gần nhất.
    #    Không dùng "độ khớp %" làm xác suất: gallery có cả ảnh bất thường,
    #    nên vụ tai nạn vẫn "khớp" cao với vụ tai nạn khác (xem decision.py).
    try:
        normal_sim, a_score = anomaly_score(image_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        if raise_on_error:
            raise
        print(f"[main.py] Lỗi khi tính điểm bất thường: {e}")
        return None
    is_abnormal = decide_abnormal(a_score, threshold)

    # 3) map tên file → event id (dữ liệu của Huy)
    file_to_id = load_file_to_id(raise_on_error=raise_on_error)
    if file_to_id is None:
        return None
    raw_event_id = file_to_id.get(best_file)
    if raw_event_id is None:
        if raise_on_error:
            raise ValueError(f"Ảnh khớp '{best_file}' không có trong events.json.")
        print(f"[main.py] Cảnh báo: ảnh khớp '{best_file}' không có trong events.json.")
        print("           Kiểm tra khớp tên file giữa data/ và events.json (nhờ Huy).")
        return None

    # 4) tra tên sự kiện + cách xử lý từ đồ thị (Thịnh)
    raw_event_name = graph.get_event(raw_event_id)
    raw_action = graph.get_action(raw_event_id)
    if raw_action is None or raw_event_name is None:
        if raise_on_error:
            raise ValueError(f"Sự kiện id={raw_event_id} không có trong graph.")
        print(f"[main.py] Cảnh báo: sự kiện id={raw_event_id} không có trong graph. Kiểm tra events.json/graph.py.")
        return None

    if is_abnormal:
        action = raw_action
        event_id = raw_event_id
        event_name = raw_event_name
        reference_event_id = None
        reference_event_name = None
    else:
        action = NORMAL_ACTION
        event_id = None
        event_name = None
        reference_event_id = raw_event_id
        reference_event_name = raw_event_name

    # 5) in kết quả
    if verbose:
        title = ("PHÁT HIỆN SỰ KIỆN BẤT THƯỜNG" if is_abnormal
                 else "KHÔNG PHÁT HIỆN BẤT THƯỜNG")
        verdict = "BẤT THƯỜNG" if is_abnormal else "BÌNH THƯỜNG"
        print("=" * 60)
        print(f"  {title} — VLM ANOMALIES DETECTION")
        print("=" * 60)
        print(f"  Ảnh đầu vào       : {image_path}")
        print(f"  Kết luận          : {verdict} (điểm bất thường {a_score:.3f} "
              f"{'>=' if is_abnormal else '<'} ngưỡng {threshold:.3f})")
        print(f"  Giống bình thường : {normal_sim * 100:.0f}% (cosine)")
        print(f"  Sự kiện gần nhất  : #{raw_event_id} — {raw_event_name}")
        print(f"  Độ khớp           : {score}%")
        print("-" * 60)
        print(f"  CÁCH XỬ LÝ        : {action}")
        print("=" * 60)

    return PipelineResult(
        image_path=str(image_path),
        is_abnormal=bool(is_abnormal),
        anomaly_score=float(a_score),
        threshold=float(threshold),
        normal_similarity=float(normal_sim),
        best_file=str(best_file),
        score=score,
        event_id=event_id,
        event_name=event_name,
        action=action,
        reference_event_id=reference_event_id,
        reference_event_name=reference_event_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM Anomalies Detection — ảnh vào, in ra sự kiện + cách xử lý."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=os.path.join(ROOT, "data", "danh_nhau_1.jpg"),
        help="đường dẫn ảnh cần kiểm tra (mặc định: data/danh_nhau_1.jpg)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_ANOMALY_THRESHOLD,
        help=(f"ngưỡng điểm bất thường (mặc định {DEFAULT_ANOMALY_THRESHOLD}). "
              "Điểm >= ngưỡng => bất thường. Ngưỡng minh hoạ, chọn trên 15 ảnh."),
    )
    args = parser.parse_args()
    res = run(args.image, threshold=args.threshold)
    if res is None:
        sys.exit(1)