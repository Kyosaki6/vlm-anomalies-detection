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

Lưu ý: nếu search.py chưa có hàm search() (đang chờ Phước thêm), script in thông
báo rõ ràng thay vì crash — phần còn lại của pipeline vẫn chạy thử được.
"""
import argparse
import json
import os
import sys

import graph

# Windows console: cho phép in tiếng Việt đúng chuẩn
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE = os.path.join(ROOT, "events.json")


def load_file_to_id(path: str = EVENTS_FILE) -> dict:
    """Đọc events.json → map {tên file: event id}.

    Dùng utf-8-sig để chịu được cả file bị thêm BOM (Windows Notepad/Excel hay thêm).
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[main.py] Lỗi: không tìm thấy {path} — Huy đã đẩy events.json lên chưa?")
        sys.exit(1)
    with open(path, encoding="utf-8-sig") as f:
        events = json.load(f)
    return {ev["file"]: str(ev["id"]) for ev in events}


def run(image_path: str) -> None:
    # 0) kiểm tra ảnh đầu vào
    if not os.path.exists(image_path):
        print(f"[main.py] Lỗi: không thấy ảnh '{image_path}'.")
        sys.exit(1)

    # 1) tìm ảnh giống nhất trong bộ sưu tập — hàm search() do Phước thêm vào search.py
    try:
        from search import search
    except (ImportError, AttributeError):
        print("=" * 60)
        print("[main.py] search.py chưa có hàm search() — nhờ Phước thêm đoạn này:")
        print()
        print("    def search(image_path):")
        print("        best_file, score = <logic hiện tại của search.py>")
        print("        return best_file, score")
        print()
        print("Khi có hàm đó, chạy lại: python main.py <ảnh>")
        print("=" * 60)
        sys.exit(1)

    best_file, score = search(image_path)

    # 2) map tên file → event id (dữ liệu của Huy)
    file_to_id = load_file_to_id()
    event_id = file_to_id.get(best_file)
    if event_id is None:
        print(f"[main.py] Cảnh báo: ảnh khớp '{best_file}' không có trong events.json.")
        print("           Kiểm tra khớp tên file giữa data/ và events.json (nhờ Huy).")
        sys.exit(1)

    # 3) tra tên sự kiện + cách xử lý từ đồ thị (Thịnh)
    event_name = graph.get_event(event_id)
    action = graph.get_action(event_id)
    if action is None or event_name is None:
        print(f"[main.py] Cảnh báo: sự kiện id={event_id} không có trong graph. Kiểm tra events.json/graph.py.")
        sys.exit(1)

    # 4) in kết quả
    print("=" * 60)
    print("  PHÁT HIỆN SỰ KIỆN BẤT THƯỜNG — VLM ANOMALIES DETECTION")
    print("=" * 60)
    print(f"  Ảnh đầu vào   : {image_path}")
    print(f"  Giống sự kiện : #{event_id} — {event_name}")
    print(f"  Độ khớp       : {score}%")
    print("-" * 60)
    print(f"  CÁCH XỬ LÝ    : {action}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM Anomalies Detection — ảnh vào, in ra sự kiện + cách xử lý."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/danh_nhau_1.jpg",
        help="đường dẫn ảnh cần kiểm tra (mặc định: data/danh_nhau_1.jpg)",
    )
    args = parser.parse_args()
    run(args.image)