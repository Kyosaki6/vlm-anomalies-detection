# -*- coding: utf-8 -*-
"""
graph.py — Đồ thị tri thức [Sự kiện] -> [Cách xử lý] bằng NetworkX.

Dữ liệu đọc từ events.json của bạn Huy (15 sự kiện).

Cách dùng:
    python graph.py 1
    python graph.py "1"
    python graph.py --id 2
    python graph.py --list   # liệt kê tất cả id + tên sự kiện

Hàm chính:
    get_action("1") -> "Báo bảo vệ cổng A và gọi cứu thương"
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

EVENTS_FILE = Path(__file__).resolve().parent / "events.json"


def load_events(path: Path = EVENTS_FILE) -> list[dict]:
    """Đọc file events.json (UTF-8-SIG) và trả về danh sách dict sự kiện."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def build_graph(events: list[dict]) -> tuple[nx.DiGraph, dict[str, str], dict[str, str]]:
    """
    Dựng đồ thị có hướng: [Sự kiện] -> [Cách xử lý].

    Trả về (graph, id_to_event, id_to_action).
    Mỗi cạnh lưu thêm thuộc tính id và file để truy vết ngược.
    """
    graph = nx.DiGraph()
    id_to_event: dict[str, str] = {}
    id_to_action: dict[str, str] = {}

    for item in events:
        event_id = str(item["id"])
        event = item["ten_su_kien"]
        action = item["cach_xu_ly"]

        id_to_event[event_id] = event
        id_to_action[event_id] = action

        graph.add_node(
            event,
            kind="su_kien",
            id=event_id,
            category=item.get("category", ""),
            is_normal=item.get("is_normal", False),
            severity=item.get("severity", ""),
            responder=item.get("responder", []),
            file=item.get("file", ""),
        )
        graph.add_node(action, kind="cach_xu_ly")
        # Nối: [Sự kiện] -> [Cách xử lý]
        graph.add_edge(event, action, id=event_id, file=item.get("file", ""))

    return graph, id_to_event, id_to_action


# Build sẵn 1 lần khi import để get_action() dùng ngay.
_EVENTS = load_events()
GRAPH, ID_TO_EVENT, ID_TO_ACTION = build_graph(_EVENTS)


def get_action(event_id: str | int) -> str | None:
    """
    Trả về cách xử lý cho id sự kiện.

    Ví dụ:
        get_action("1") -> "Báo bảo vệ cổng A và gọi cứu thương"

    Đi qua đồ thị NetworkX: tìm node [Sự kiện] theo id,
    rồi lấy successor duy nhất là [Cách xử lý].
    Trả về None nếu id không tồn tại.
    """
    eid = str(event_id).strip()
    event = ID_TO_EVENT.get(eid)
    if event is None:
        return None
    # Lấy qua cạnh đồ thị [Sự kiện] -> [Cách xử lý]
    successors = list(GRAPH.successors(event))
    if not successors:
        return ID_TO_ACTION.get(eid)
    return successors[0]


def get_event(event_id: str | int) -> str | None:
    """Trả về tên sự kiện cho id (tiện tra cứu ngược)."""
    return ID_TO_EVENT.get(str(event_id).strip())


def get_event_by_id(event_id: str | int) -> dict | None:
    """Trả về dict thông tin chi tiết sự kiện theo ID từ đồ thị.

    Bao gồm: id, file, ten_su_kien, cach_xu_ly, category, is_normal, severity, responder.
    Trả về None nếu id không tồn tại.
    """
    eid = str(event_id).strip()
    event_name = ID_TO_EVENT.get(eid)
    if event_name is None or event_name not in GRAPH.nodes:
        return None
    node = GRAPH.nodes[event_name]
    action = get_action(eid)
    return {
        "id": eid,
        "file": node.get("file", ""),
        "ten_su_kien": event_name,
        "cach_xu_ly": action,
        "category": node.get("category", ""),
        "is_normal": node.get("is_normal", False),
        "severity": node.get("severity", ""),
        "responder": node.get("responder", []),
    }


def get_events_by_category(category: str) -> list[dict]:
    """Trả về danh sách dict sự kiện thuộc category (ví dụ: danh_nhau, duong_pho, tai_nan_xe)."""
    cat = str(category).strip()
    results = []
    for eid in ID_TO_EVENT:
        ev = get_event_by_id(eid)
        if ev and ev.get("category") == cat:
            results.append(ev)
    return results


def get_action_details(event_id: str | int) -> dict | None:
    """Trả về dict chi tiết hành động xử lý cho event_id.

    Bao gồm: event_id, action, severity, responder, category.
    Trả về None nếu event_id không tồn tại.
    """
    eid = str(event_id).strip()
    ev = get_event_by_id(eid)
    if ev is None:
        return None
    return {
        "event_id": eid,
        "action": ev.get("cach_xu_ly"),
        "severity": ev.get("severity", ""),
        "responder": ev.get("responder", []),
        "category": ev.get("category", ""),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tra cứu cách xử lý sự kiện từ đồ thị NetworkX [Sự kiện] -> [Cách xử lý]."
    )
    p.add_argument("event_id", nargs="?", default=None, help="ID sự kiện, ví dụ: 1")
    p.add_argument("--id", dest="id_opt", default=None, help="ID sự kiện (dạng flag).")
    p.add_argument("--list", action="store_true", help="Liệt kê tất cả sự kiện kèm category và severity.")
    p.add_argument("--events-file", default=str(EVENTS_FILE), help="Đường dẫn events.json.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        events = load_events(Path(args.events_file))
        for item in events:
            cat = item.get("category", "-")
            sev = item.get("severity", "-")
            print(f'{item["id"]}: [{cat} | {sev}] {item["ten_su_kien"]} -> {item["cach_xu_ly"]}')
        return 0

    event_id = args.id_opt if args.id_opt is not None else args.event_id
    if event_id is None:
        # Mặc định demo đúng yêu cầu đề bài
        event_id = "1"

    # Nếu trỏ sang file events.json khác thì build lại graph tạm
    if Path(args.events_file).resolve() != EVENTS_FILE:
        events = load_events(Path(args.events_file))
        _, id_to_event, id_to_action = build_graph(events)
        eid = str(event_id).strip()
        action = id_to_action.get(eid)
    else:
        action = get_action(event_id)

    if action is None:
        print(f"Không tìm thấy sự kiện có id={event_id!r}.", file=sys.stderr)
        return 1

    print(action)
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main())
