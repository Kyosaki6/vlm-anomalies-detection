# -*- coding: utf-8 -*-
"""Ghép kết quả tier-2 thành "bản tin cảnh báo" (track T4, Nam): gộp + cooldown + leo thang theo thời gian.

`vad/graph/policy.py` trả lời "một sự kiện -> một Alert". File này trả lời "cả một video -> danh sách
sự cố", tức là phần dashboard thực sự ăn. Ba việc:

  1. Leo thang theo NGỮ CẢNH THỜI GIAN: hai sự kiện chồng lấn thời gian trong cùng cửa sổ (vd tai nạn +
     nổ) là một sự cố phức hợp, không phải hai cảnh báo rời.
  2. `merge_alerts`: hai cảnh báo cùng loại cách nhau < merge_gap_s -> một sự cố (dùng lại từ policy.py).
  3. `apply_cooldown`: chặn báo lặp cùng video + cùng loại trong cooldown_s.

Chỉ numpy/thuần Python: chạy CPU, test bằng FakeVerifier.
"""
from __future__ import annotations

from collections.abc import Sequence

from vad.graph.policy import GraphPolicy, merge_alerts
from vad.types import Alert, EventType, VerifyResult, Window


def _overlap(a: Window, b: Window) -> bool:
    """Hai cửa sổ chồng lấn thời gian (biên chạm nhau KHÔNG tính là chồng)."""
    return a.start_s < b.end_s and b.start_s < a.end_s


def apply_cooldown(alerts: Sequence[Alert], cooldown_s: float) -> list[Alert]:
    """Bỏ cảnh báo lặp: cùng CAMERA + cùng loại, bắt đầu < cooldown_s sau cảnh báo đã giữ.

    Camera lấy từ `Alert.camera`; cảnh báo chưa biết camera (chuỗi rỗng) thì lùi về `video_id`
    — một video coi như một camera. Mốc tính từ `end_s` của cảnh báo đã giữ ("nguội" sau khi sự cố
    kết thúc). Cảnh báo bị bỏ KHÔNG làm mới cooldown — nếu không, một sự cố kéo dài sẽ tự chặn mình mãi mãi.
    """
    if cooldown_s < 0:
        raise ValueError(f"cooldown_s phải >= 0, nhận {cooldown_s}")

    kept: list[Alert] = []
    last_kept: dict[tuple[str, str], Alert] = {}
    for a in sorted(alerts, key=lambda x: (x.video_id, x.start_s)):
        key = (a.camera or a.video_id, a.event_type.value)
        prev = last_kept.get(key)
        if prev is not None and a.start_s - prev.end_s < cooldown_s:
            continue
        kept.append(a)
        last_kept[key] = a
    return kept


def build_alert_feed(detections: Sequence[tuple[Window, VerifyResult]], policy: GraphPolicy,
                     zone: str | None = None, hour: int | None = None,
                     camera: str | None = None) -> list[Alert]:
    """[(Window, VerifyResult)] -> danh sách Alert đã leo thang ngữ cảnh, gộp, cooldown.

    `zone`/`hour`/`camera` áp cho cả video (một camera một zone một khung giờ). Video nhiều camera
    hoặc nhiều zone thì gọi hàm này từng đoạn — không đoán từng cửa sổ.
    """
    anomalous = [(i, w, r) for i, (w, r) in enumerate(detections)
                 if r.is_anomaly and r.event_type is not EventType.NORMAL]

    alerts: list[Alert] = []
    for i, w, r in anomalous:
        # Sự kiện khác chồng lấn thời gian -> ứng viên leo thang
        co = tuple({r2.event_type for j, w2, r2 in anomalous if j != i and _overlap(w, w2)})
        alert = policy.to_alert(w, r, camera=camera, zone=zone, hour=hour, co_events=co)
        if alert is not None:
            alerts.append(alert)

    hyg = policy.policy["hygiene"]
    merged = merge_alerts(alerts, float(hyg["merge_gap_s"]))
    return apply_cooldown(merged, float(hyg["cooldown_s"]))
