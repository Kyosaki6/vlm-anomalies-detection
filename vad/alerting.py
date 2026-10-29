# -*- coding: utf-8 -*-
"""Điểm vào C5 của track T4 — tên đường dẫn theo phân công trong task sheet.

Toàn bộ logic nằm ở `vad/graph/alerts.py` (cạnh `policy.py`, vì nó dùng policy graph để tạo cảnh báo).
File này CHỈ re-export, không chứa logic: hai bản cài đặt song song là cách chắc chắn nhất để sáu tháng
sau có người sửa một bên và dashboard đọc bên kia.

Dùng đường nào cũng được, cùng một hàm:

    from vad.alerting import build_alert_feed      # theo task sheet (C5)
    from vad.graph.alerts import build_alert_feed  # theo vị trí thật trong code

`tests/vad_suite/test_alerting.py` khoá hai đường này là MỘT, để chúng không trôi khỏi nhau.
"""
from __future__ import annotations

from vad.graph.alerts import apply_cooldown, build_alert_feed
from vad.graph.policy import merge_alerts

__all__ = ["apply_cooldown", "build_alert_feed", "merge_alerts"]
