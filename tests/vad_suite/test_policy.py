import pytest

from vad.graph.policy import GraphPolicy, load_policy, merge_alerts, time_band_of
from vad.types import Alert, EventType, VerifyResult, Window

W = Window("vid", 10.0, 20.0, 3.0)


def vr(event, conf=0.9, anomaly=True):
    return VerifyResult(anomaly, event, "caption", conf)


@pytest.fixture(scope="module")
def policy():
    return GraphPolicy()


def test_normal_or_low_confidence_gives_no_alert(policy):
    assert policy.to_alert(W, vr(EventType.NORMAL, anomaly=False)) is None
    assert policy.to_alert(W, vr(EventType.FIGHTING, conf=0.3)) is None
    assert policy.to_alert(W, vr(EventType.NORMAL)) is None


def test_base_alert_and_path(policy):
    a = policy.to_alert(W, vr(EventType.FIGHTING))
    assert a.severity == "medium" and a.protocol == "P-VIOLENCE"
    assert a.responders == ("security",)
    assert a.path[0] == "event:Fighting" and a.path[-1] == "protocol:P-VIOLENCE"


def test_context_changes_severity_same_event(policy):
    base = policy.to_alert(W, vr(EventType.FIGHTING), zone="market", hour=12)
    school_night = policy.to_alert(W, vr(EventType.FIGHTING), zone="school_gate", hour=23)
    assert base.severity == "medium"
    assert school_night.severity == "critical"          # medium +1 (zone) +1 (night)
    assert any(p.startswith("zone:school_gate") for p in school_night.path)
    assert any(p.startswith("time_band:night") for p in school_night.path)


def test_severity_is_clamped_both_ends(policy):
    top = policy.to_alert(W, vr(EventType.SHOOTING), zone="school_gate", hour=23)
    assert top.severity == "critical"                    # không vượt trần
    low = policy.to_alert(W, vr(EventType.ARREST), zone="parking_lot", hour=12)
    assert low.severity == "low"                         # không thủng sàn


def test_escalation_adds_responder_and_raises_severity(policy):
    plain = policy.to_alert(W, vr(EventType.ROAD_ACCIDENT))
    esc = policy.to_alert(W, vr(EventType.ROAD_ACCIDENT), co_events=(EventType.EXPLOSION,))
    assert "fire_dept" not in plain.responders
    assert "fire_dept" in esc.responders
    assert esc.severity == "critical" and plain.severity == "high"
    assert "escalates_with:Explosion" in esc.path


def test_unknown_zone_and_event_fall_back(policy):
    a = policy.to_alert(W, vr(EventType.FIGHTING), zone="moon_base")
    assert a.severity == "medium"                        # zone lạ -> default, không crash
    u = policy.to_alert(W, vr(EventType.UNKNOWN))
    assert u.protocol == "P-REVIEW"


@pytest.mark.parametrize("hour,band", [(23, "night"), (0, "night"), (5, "night"), (6, "day"), (21, "day"), (22, "night")])
def test_time_band_wraps_midnight(hour, band):
    assert time_band_of(hour, load_policy()) == band


@pytest.mark.parametrize("bad", [-1, 24])
def test_time_band_rejects_bad_hour(bad):
    with pytest.raises(ValueError):
        time_band_of(bad, load_policy())


def test_graph_has_typed_edges(policy):
    kinds = {d["kind"] for _, _, d in policy.graph.edges(data=True)}
    assert {"has_severity", "requires", "follows", "escalates_with"} <= kinds


def test_policy_yaml_missing_key_is_rejected(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("severity_order: [low]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu khoá"):
        load_policy(f)


def _alert(start, end, sev="medium", ev=EventType.FIGHTING, vid="v"):
    return Alert(vid, start, end, ev, sev, ("security",), "P", ("p",))


def test_merge_alerts_joins_close_same_type_and_takes_max_severity():
    out = merge_alerts([_alert(0, 5, "medium"), _alert(7, 10, "high")], merge_gap_s=5.0)
    assert len(out) == 1 and (out[0].start_s, out[0].end_s) == (0, 10) and out[0].severity == "high"


def test_merge_alerts_keeps_far_or_different():
    assert len(merge_alerts([_alert(0, 5), _alert(50, 55)], merge_gap_s=5.0)) == 2
    assert len(merge_alerts([_alert(0, 5), _alert(6, 9, ev=EventType.ARSON)], merge_gap_s=5.0)) == 2
    assert len(merge_alerts([_alert(0, 5, vid="a"), _alert(6, 9, vid="b")], merge_gap_s=5.0)) == 2
