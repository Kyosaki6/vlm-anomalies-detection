# -*- coding: utf-8 -*-
"""Alert Policy Graph (track T4, Nam): YAML -> NetworkX, trả Alert có đường đi giải thích được.

Đồ thị có kiểu:
    node kind: event | responder | protocol | zone | time_band | severity
    edge kind: has_severity (event->severity) | requires (event->responder) | follows (event->protocol)
               escalates_with (event<->event, có thuộc tính add_responders / severity_shift)
Mức cuối = base severity + zone shift + time-band shift (+ escalation), kẹp trong severity_order.

Bản này CHẠY ĐƯỢC (không phải stub) để Nam mở rộng; test ở tests/vad/test_policy.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from vad.types import Alert, EventType, VerifyResult, Window

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"


def load_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        policy = yaml.safe_load(f)
    for key in ("severity_order", "events", "zones", "time_bands", "escalations", "hygiene"):
        if key not in policy:
            raise ValueError(f"policy.yaml thiếu khoá bắt buộc: {key!r}")
    return policy


def build_policy_graph(policy: dict[str, Any]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for sev in policy["severity_order"]:
        g.add_node(("severity", sev), kind="severity")
    for name, spec in policy["events"].items():
        ev = ("event", name)
        g.add_node(ev, kind="event")
        g.add_edge(ev, ("severity", spec["severity"]), kind="has_severity")
        for r in spec.get("responders", []):
            g.add_node(("responder", r), kind="responder")
            g.add_edge(ev, ("responder", r), kind="requires")
        g.add_node(("protocol", spec["protocol"]), kind="protocol")
        g.add_edge(ev, ("protocol", spec["protocol"]), kind="follows")
    for zone, spec in policy["zones"].items():
        g.add_node(("zone", zone), kind="zone", severity_shift=spec.get("severity_shift", 0))
    for band, spec in policy["time_bands"].items():
        g.add_node(("time_band", band), kind="time_band", **spec)
    for esc in policy["escalations"]:
        a, b = esc["when"]
        for src, dst in ((a, b), (b, a)):
            g.add_edge(("event", src), ("event", dst), kind="escalates_with",
                       add_responders=tuple(esc.get("add_responders", [])),
                       severity_shift=esc.get("severity_shift", 0))
    return g


def time_band_of(hour: int, policy: dict[str, Any]) -> str | None:
    """Giờ 0-23 -> tên khung giờ; xử lý khung vắt qua nửa đêm (22 -> 6)."""
    if not 0 <= hour <= 23:
        raise ValueError(f"hour phải trong 0..23, nhận {hour}")
    for name, spec in policy["time_bands"].items():
        s, e = spec["start"], spec["end"]
        inside = (s <= hour < e) if s < e else (hour >= s or hour < e)
        if inside:
            return name
    return None


class GraphPolicy:
    """Triển khai `vad.interfaces.AlertPolicy`."""

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or load_policy()
        self.graph = build_policy_graph(self.policy)
        self._order = list(self.policy["severity_order"])

    def _clamp(self, idx: int) -> str:
        return self._order[max(0, min(idx, len(self._order) - 1))]

    def to_alert(self, window: Window, result: VerifyResult, camera: str | None = None,
                 zone: str | None = None, hour: int | None = None,
                 co_events: tuple[EventType, ...] = ()) -> Alert | None:
        hyg = self.policy["hygiene"]
        if not result.is_anomaly or result.event_type == EventType.NORMAL:
            return None
        if result.confidence < hyg["min_confidence"]:
            return None
        name = result.event_type.value
        node = ("event", name)
        if node not in self.graph:
            node, name = ("event", EventType.UNKNOWN.value), EventType.UNKNOWN.value

        spec = self.policy["events"][name]
        idx = self._order.index(spec["severity"])
        path = [f"event:{name}", f"severity:{spec['severity']}"]
        responders = list(spec.get("responders", []))

        zone_name = zone if zone in self.policy["zones"] else "default"
        shift = self.graph.nodes[("zone", zone_name)]["severity_shift"]
        if shift:
            idx += shift
            path.append(f"zone:{zone_name}({shift:+d})")
        if hour is not None:
            band = time_band_of(hour, self.policy)
            if band is not None:
                b_shift = self.graph.nodes[("time_band", band)]["severity_shift"]
                if b_shift:
                    idx += b_shift
                    path.append(f"time_band:{band}({b_shift:+d})")

        co_names = {e.value for e in co_events}
        for _, dst, data in self.graph.out_edges(node, data=True):
            if data.get("kind") == "escalates_with" and dst[1] in co_names:
                idx += data["severity_shift"]
                for r in data["add_responders"]:
                    if r not in responders:
                        responders.append(r)
                path.append(f"escalates_with:{dst[1]}")

        path.append(f"protocol:{spec['protocol']}")
        return Alert(
            video_id=window.video_id, start_s=window.start_s, end_s=window.end_s,
            event_type=result.event_type, severity=self._clamp(idx),
            responders=tuple(responders), protocol=spec["protocol"], path=tuple(path),
            caption=result.caption, confidence=result.confidence,
        )

    def responders_in(self, zone: str) -> set[str]:
        """Truy vấn: bên xử lý nào ĐƯỢC BỐ TRÍ trong zone này (khai báo `available_responders`).

        Zone lạ -> 'default'. Zone không khai báo -> tập rỗng nghĩa là "chưa có dữ liệu bố trí",
        KHÔNG phải "không ai ứng cứu" (coverage_gaps trả rỗng trong trường hợp đó).
        """
        zone_name = zone if zone in self.policy["zones"] else "default"
        return set(self.policy["zones"][zone_name].get("available_responders", []) or [])

    def events_requiring(self, responder: str) -> set[str]:
        """Truy vấn ngược: bên xử lý này tham gia vào những sự kiện nào (duyệt cạnh `requires`)."""
        node = ("responder", responder)
        if node not in self.graph:
            return set()
        return {src[1] for src, _, data in self.graph.in_edges(node, data=True)
                if data.get("kind") == "requires"}

    def coverage_gaps(self, zone: str) -> dict[str, tuple[str, ...]]:
        """Sự kiện nào KHÔNG đủ bên xử lý trong zone: {tên sự kiện: (bên còn thiếu...)}.

        Đây là lý do tồn tại của đồ thị thay vì bảng if-else: cùng một cấu trúc trả lời được
        "zone này thiếu ai cho sự kiện nào" mà không phải viết tay một truy vấn cho mỗi câu hỏi.
        Zone chưa khai báo `available_responders` -> {} (không có dữ liệu, không đoán).
        """
        available = self.responders_in(zone)
        if not available:
            return {}
        gaps: dict[str, tuple[str, ...]] = {}
        for name, spec in self.policy["events"].items():
            missing = tuple(r for r in spec.get("responders", []) if r not in available)
            if missing:
                gaps[name] = missing
        return gaps


def merge_alerts(alerts: list[Alert], merge_gap_s: float) -> list[Alert]:
    """Gộp cảnh báo cùng video + loại + sự kiện cách nhau < merge_gap_s thành một sự cố."""
    out: list[Alert] = []
    for a in sorted(alerts, key=lambda x: (x.video_id, x.event_type.value, x.start_s)):
        if out:
            p = out[-1]
            if (p.video_id, p.event_type) == (a.video_id, a.event_type) and a.start_s - p.end_s < merge_gap_s:
                out[-1] = Alert(
                    video_id=p.video_id, start_s=p.start_s, end_s=max(p.end_s, a.end_s),
                    event_type=p.event_type, severity=max((p.severity, a.severity),
                                                          key=lambda s: _SEV_RANK.get(s, 0)),
                    responders=tuple(dict.fromkeys(p.responders + a.responders)),
                    protocol=p.protocol, path=p.path, caption=p.caption or a.caption,
                    confidence=max(p.confidence, a.confidence))
                continue
        out.append(a)
    return sorted(out, key=lambda x: (x.video_id, x.start_s))


_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
