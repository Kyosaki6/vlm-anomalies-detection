# -*- coding: utf-8 -*-
"""Test gộp/cooldown/leo thang cảnh báo (track T4). Thuần Python + numpy, không GPU/model."""
import pytest

from vad.fakes import FakeVerifier
from vad.graph.alerts import apply_cooldown, build_alert_feed
from vad.graph.policy import GraphPolicy
from vad.types import Alert, EventType, VerifyResult, Window


@pytest.fixture(scope="module")
def policy():
    return GraphPolicy()


def vr(event, conf=0.9, anomaly=True, caption="c"):
    return VerifyResult(anomaly, event, caption, conf)


def win(start, end, vid="v", score=3.0):
    return Window(vid, start, end, score)


def _alert(start, end, ev=EventType.FIGHTING, vid="v", sev="medium"):
    return Alert(vid, start, end, ev, sev, ("security",), "P", ("p",))


# --- apply_cooldown ---------------------------------------------------------------

def test_cooldown_drops_repeat_within_window():
    out = apply_cooldown([_alert(0, 4), _alert(10, 14)], cooldown_s=30.0)
    assert len(out) == 1 and out[0].start_s == 0


def test_cooldown_keeps_alert_after_window():
    out = apply_cooldown([_alert(0, 4), _alert(40, 44)], cooldown_s=30.0)
    assert [a.start_s for a in out] == [0, 40]


def test_cooldown_is_per_event_type_and_video():
    out = apply_cooldown(
        [_alert(0, 4), _alert(10, 14, ev=EventType.ARSON), _alert(10, 14, vid="other")],
        cooldown_s=30.0,
    )
    assert len(out) == 3


def test_dropped_alert_does_not_refresh_the_cooldown():
    """Sự cố kéo dài: 0-4, 10-14 (bỏ), 40-44. Cảnh báo 40 phải sống vì mốc là 4, không phải 14."""
    out = apply_cooldown([_alert(0, 4), _alert(10, 14), _alert(40, 44)], cooldown_s=30.0)
    assert [a.start_s for a in out] == [0, 40]


def test_cooldown_rejects_negative_budget():
    with pytest.raises(ValueError, match="cooldown_s"):
        apply_cooldown([_alert(0, 4)], cooldown_s=-1.0)


def test_zero_cooldown_keeps_everything():
    assert len(apply_cooldown([_alert(0, 4), _alert(4, 8)], cooldown_s=0.0)) == 2


# --- build_alert_feed -------------------------------------------------------------

def test_feed_merges_adjacent_same_event(policy):
    feed = build_alert_feed([(win(0, 5), vr(EventType.FIGHTING)), (win(7, 10), vr(EventType.FIGHTING))],
                            policy)
    assert len(feed) == 1 and (feed[0].start_s, feed[0].end_s) == (0, 10)


def test_feed_escalates_only_when_windows_overlap(policy):
    overlapped = build_alert_feed(
        [(win(10, 20), vr(EventType.ROAD_ACCIDENT)), (win(15, 25), vr(EventType.EXPLOSION))], policy)
    accident = [a for a in overlapped if a.event_type is EventType.ROAD_ACCIDENT][0]
    assert "fire_dept" in accident.responders
    assert accident.severity == "critical"
    assert "escalates_with:Explosion" in accident.path

    apart = build_alert_feed(
        [(win(0, 10), vr(EventType.ROAD_ACCIDENT)), (win(100, 110), vr(EventType.EXPLOSION))], policy)
    accident_alone = [a for a in apart if a.event_type is EventType.ROAD_ACCIDENT][0]
    assert "fire_dept" not in accident_alone.responders
    assert accident_alone.severity == "high"


def test_feed_ignores_normal_and_low_confidence(policy):
    feed = build_alert_feed(
        [(win(0, 5), vr(EventType.NORMAL, anomaly=False)),
         (win(10, 15), vr(EventType.FIGHTING, conf=0.2)),
         (win(20, 25), vr(EventType.FIGHTING, anomaly=False))], policy)
    assert feed == []


def test_feed_applies_cooldown_end_to_end(policy):
    """Hai cảnh báo cùng loại cách 6s: quá xa để gộp (merge_gap 5s) nhưng trong cooldown 30s -> còn 1."""
    feed = build_alert_feed([(win(0, 4), vr(EventType.FIGHTING)), (win(10, 14), vr(EventType.FIGHTING))],
                            policy)
    assert len(feed) == 1


def test_feed_respects_zone_and_hour_context(policy):
    det = [(win(10, 20), vr(EventType.FIGHTING))]
    day = build_alert_feed(det, policy, zone="market", hour=12)[0]
    night = build_alert_feed(det, policy, zone="school_gate", hour=23)[0]
    assert (day.severity, night.severity) == ("medium", "critical")


def test_feed_sorts_by_video_then_time(policy):
    feed = build_alert_feed(
        [(win(50, 60, vid="b"), vr(EventType.ARSON)),
         (win(5, 10, vid="a"), vr(EventType.ARSON)),
         (win(200, 210, vid="b"), vr(EventType.ARSON))], policy)
    assert [(a.video_id, a.start_s) for a in feed] == [("a", 5), ("b", 50), ("b", 200)]


def test_feed_works_with_fake_verifier(policy):
    """Đường ống dashboard dùng: FakeVerifier -> Window -> feed, không cần VLM thật."""
    fake = FakeVerifier()
    det = [(win(0, 10), fake.verify([], {})), (win(100, 110), fake.verify([], {}))]
    feed = build_alert_feed(det, policy, zone="school_gate", hour=23)
    assert len(feed) == 2
    assert all(a.event_type is EventType.FIGHTING and a.caption for a in feed)


# --- camera đi xuyên suốt: to_alert -> gộp -> cooldown ----------------------------

def test_feed_passes_camera_through_to_every_alert(policy):
    feed = build_alert_feed([(win(0, 10), vr(EventType.FIGHTING))], policy, camera="gate_a")
    assert feed[0].camera == "gate_a" and feed[0].path[0] == "camera:gate_a"


def test_cooldown_is_per_camera_not_per_video(policy):
    """Hai camera khác nhau cùng chỗ ghi hình: không được nuốt cảnh báo của nhau."""
    det = [
        (Window("site", 0, 5, 3.0), vr(EventType.FIGHTING)),
        (Window("site", 10, 15, 3.0), vr(EventType.FIGHTING)),
    ]
    same = build_alert_feed(det, policy, camera="cam1")          # cùng camera -> còn 1
    assert len(same) == 1
    # khác camera: gọi riêng từng camera rồi gộp tay, cooldown phải giữ cả hai
    a = policy.to_alert(det[0][0], det[0][1], camera="cam1")
    b = policy.to_alert(det[1][0], det[1][1], camera="cam2")
    assert len(apply_cooldown([a, b], cooldown_s=30.0)) == 2


def test_cooldown_falls_back_to_video_when_camera_unknown(policy):
    det = [
        (Window("site", 0, 5, 3.0), vr(EventType.FIGHTING)),
        (Window("site", 10, 15, 3.0), vr(EventType.FIGHTING)),
    ]
    assert len(build_alert_feed(det, policy)) == 1                # camera rỗng -> khoá theo video
