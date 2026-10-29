# -*- coding: utf-8 -*-
"""Test C5: `vad/alerting.py` là ĐIỂM VÀO, không phải bản cài đặt thứ hai.

Task sheet gọi task này là `vad/alerting.py`; code thật nằm ở `vad/graph/alerts.py` (cạnh policy.py).
File này chỉ kiểm tra ba điều:
  1. hai đường import là cùng một object (không ai copy-paste thành bản thứ hai),
  2. API công khai đủ cho dashboard dùng,
  3. bảng kịch bản của C5 (>= 10 ca) chạy qua ĐƯỜNG ĐƯỢC PHÂN CÔNG, không chỉ qua đường cũ.
"""
import pytest

import vad.alerting as alerting
import vad.graph.alerts as canonical
from vad.fakes import FakeVerifier
from vad.graph.policy import GraphPolicy
from vad.types import Alert, EventType, VerifyResult, Window


# --- 1. một bản cài đặt, hai đường dẫn -------------------------------------------

def test_reexports_point_at_the_same_function_objects():
    assert alerting.build_alert_feed is canonical.build_alert_feed
    assert alerting.apply_cooldown is canonical.apply_cooldown


def test_public_api_is_exported():
    assert set(alerting.__all__) == {"apply_cooldown", "build_alert_feed", "merge_alerts"}


def test_module_contains_no_logic_of_its_own():
    """Chỉ được import/re-export: không được định nghĩa hàm mới trong file alias."""
    own = [n for n, v in vars(alerting).items()
           if callable(v) and getattr(v, "__module__", None) == "vad.alerting"]
    assert own == [], f"vad/alerting.py phải là re-export thuần, nhưng có: {own}"


# --- 2/3. bảng kịch bản C5 qua đường `vad.alerting` -------------------------------

@pytest.fixture()
def policy():
    return GraphPolicy()


def vr(event=EventType.FIGHTING, conf=0.9, anomaly=True, caption="x"):
    return VerifyResult(anomaly, event, caption, conf)


def win(start, end, score=3.0, vid="v"):
    return Window(vid, start, end, score)


# 10 ca kịch bản (lấy đúng con số acceptance của C5).
CASES = [
    # (tên, detections, kwargs cho feed, số cảnh báo mong đợi)
    ("một sự kiện duy nhất", [(win(0, 10), vr(EventType.FIGHTING))], {}, 1),
    ("sự kiện bình thường bị bỏ", [(win(0, 10), vr(EventType.NORMAL))], {}, 0),
    ("VLM nói không bất thường", [(win(0, 10), vr(EventType.FIGHTING, anomaly=False))], {}, 0),
    ("confidence thấp bị bỏ", [(win(0, 10), vr(EventType.FIGHTING, conf=0.1))], {}, 0),
    ("hai cửa sổ rời nhau", [(win(0, 10), vr()), (win(100, 110), vr())], {}, 2),
    ("hai cửa sổ sát nhau bị gộp",
     [(win(0, 10), vr()), (win(12, 20), vr())], {}, 1),
    ("chồng lấn khác loại -> leo thang",
     [(win(0, 10), vr(EventType.ROAD_ACCIDENT)), (win(5, 15), vr(EventType.EXPLOSION))], {}, 2),
    ("cooldown chặn báo lặp", [(win(0, 5), vr()), (win(10, 15), vr())], {}, 1),
    ("zone + khung giờ đổi mức độ nặng",
     [(win(0, 10), vr())], {"zone": "school_gate", "hour": 23}, 1),
    ("camera được ghi vào cảnh báo",
     [(win(0, 10), vr())], {"camera": "cong_truong"}, 1),
]


@pytest.mark.parametrize("name,detections,kwargs,expected", CASES, ids=[c[0] for c in CASES])
def test_c5_scenario_table(name, detections, kwargs, expected, policy):
    feed = alerting.build_alert_feed(detections, policy, **kwargs)
    assert len(feed) == expected, f"{name}: được {len(feed)} cảnh báo, mong đợi {expected}"
    assert all(isinstance(a, Alert) for a in feed)


def test_escalation_reaches_critical_through_the_alias(policy):
    feed = alerting.build_alert_feed(
        [(win(0, 10), vr(EventType.ROAD_ACCIDENT)), (win(5, 15), vr(EventType.EXPLOSION))],
        policy, hour=23)
    assert any(a.severity == "critical" for a in feed)
    assert any("escalates_with" in " ".join(a.path) for a in feed)


def test_camera_survives_the_alias(policy):
    feed = alerting.build_alert_feed([(win(0, 10), vr())], policy, camera="cong_truong")
    assert feed[0].camera == "cong_truong"


def test_works_with_fake_verifier_through_the_alias(policy):
    fake = FakeVerifier()
    det = [(win(0, 10), fake.verify([], {})), (win(100, 110), fake.verify([], {}))]
    assert len(alerting.build_alert_feed(det, policy)) == 2
