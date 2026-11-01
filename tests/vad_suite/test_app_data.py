# -*- coding: utf-8 -*-
"""Test app/data.py (E1): nạp results/*.json, banner dữ liệu giả, lọc cảnh báo. Không cần streamlit/GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import data  # noqa: E402
from vad.results import RunResult  # noqa: E402
from vad.types import Alert, EventType  # noqa: E402

SAMPLE_ROOT = ROOT / "results"


def _alert(sev: str, start: float) -> Alert:
    return Alert(video_id="v", start_s=start, end_s=start + 5, event_type=EventType.FIGHTING, severity=sev,
                 responders=(), protocol="P", path=())


def test_sample_run_loads_and_is_flagged_placeholder():
    run = data.load_run("sample_run", SAMPLE_ROOT)
    assert run.video_id == "sample_run" and len(run.anomaly_scores) == 120
    assert data.is_placeholder(run) and not data.is_demo(run)
    assert "DỮ LIỆU GIẢ" in data.run_note(run)      # nguyên văn, không được giấu
    assert data.duration_s(run) == 120.0


def _run_json(video_id: str, **over) -> str:
    return json.dumps({"schema_version": 1, "video_id": video_id, "sample_fps": 1.0, **over})

def test_list_video_ids_skips_windows_dir_and_missing_root(tmp_path):
    assert data.list_video_ids(tmp_path / "nope") == []
    (tmp_path / "b.json").write_text(_run_json("b"))
    (tmp_path / "a.json").write_text(_run_json("a"))
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "a.json").write_text(_run_json("a"))
    assert data.list_video_ids(tmp_path) == ["a", "b"]

def test_list_video_ids_mixed_dir_keeps_only_run_files(tmp_path):
    """Thư mục lẫn run + ablation/threshold/typing/non-dict/JSON hỏng/schema lạ: chỉ giữ run schema_version=1."""
    (tmp_path / "run1.json").write_text(_run_json("run1"))
    (tmp_path / "run2.json").write_bytes(b"\xef\xbb\xbf" + _run_json("run2").encode())   # BOM
    (tmp_path / "ablation_A0.json").write_text(json.dumps({"ablation": "A0", "config": {}}))
    (tmp_path / "threshold_A4c.json").write_text(json.dumps({"threshold": 1.1686}))
    (tmp_path / "typing_eval.json").write_text(json.dumps({"task": "C3-typing"}))
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "empty.json").write_text("")
    (tmp_path / "future.json").write_text(_run_json("future", schema_version=99))
    ids = data.list_video_ids(tmp_path)
    assert ids == ["run1", "run2"]
    for vid in ids:                      # mọi id liệt kê đều nạp được, không ném
        data.load_run(vid, tmp_path)

def test_list_video_ids_real_results_dir_has_no_ablation_files():
    ids = data.list_video_ids(SAMPLE_ROOT)
    assert "sample_run" in ids
    assert not any(i.startswith(("ablation_", "threshold_", "typing_", "a0_stride")) for i in ids)
    for vid in ids:
        data.load_run(vid, SAMPLE_ROOT)

def test_default_results_dir_prefers_run_a4c(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DEFAULT_RESULTS_DIR", tmp_path)
    assert data.default_results_dir() == tmp_path
    (tmp_path / "run_A4c").mkdir()
    assert data.default_results_dir() == tmp_path / "run_A4c"


def test_load_run_rejects_hostile_and_missing_ids(tmp_path):
    with pytest.raises(ValueError):
        data.load_run("../../etc/passwd", tmp_path)
    with pytest.raises(FileNotFoundError):
        data.load_run("missing", tmp_path)


def test_load_run_rejects_wrong_schema_version(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"schema_version": 99, "video_id": "old", "sample_fps": 1.0}))
    with pytest.raises(ValueError, match="schema_version"):
        data.load_run("old", tmp_path)


def test_demo_run_declares_itself_and_has_no_fabricated_verification():
    run = data.demo_run()
    assert data.is_demo(run) and data.is_placeholder(run)
    assert run.metadata["note"] == data.DEMO_NOTE
    assert run.verify_results == () and run.alerts == ()
    assert len(run.candidate_windows) >= 1
    assert all(v is None for _, v in data.detections(run))
    assert data.demo_run(seed=1).anomaly_scores == data.demo_run(seed=1).anomaly_scores


def test_detections_pairs_windows_with_verify_results():
    run = data.load_run("sample_run", SAMPLE_ROOT)
    pairs = data.detections(run)
    assert len(pairs) == 3 and all(v is not None for _, v in pairs)
    assert pairs[1][1].event_type is EventType.ROAD_ACCIDENT


def test_alerts_by_severity_filters_and_sorts_newest_first():
    alerts = [_alert("low", 1), _alert("critical", 10), _alert("high", 5), _alert("weird", 7)]
    out = data.alerts_by_severity(alerts, "high")
    assert [a.severity for a in out] == ["critical", "high"]
    assert [a.severity for a in data.alerts_by_severity(alerts)] == ["critical", "high", "low"]   # lạ bị bỏ
    with pytest.raises(ValueError):
        data.alerts_by_severity(alerts, "urgent")


def test_threshold_crossings_counts_runs_not_samples():
    run = RunResult(video_id="v", sample_fps=1.0, anomaly_scores=[0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9])
    assert data.threshold_crossings(run, 0.5) == 3      # chạy đầu mảng cũng tính
    assert data.threshold_crossings(run, 2.0) == 0
    with pytest.raises(ValueError):
        data.threshold_crossings(run, float("nan"))
    empty = RunResult(video_id="e", sample_fps=1.0, anomaly_scores=[])
    assert data.threshold_crossings(empty, 0.5) == 0


def test_timeline_x_axis_uses_sample_fps():
    run = RunResult(video_id="v", sample_fps=2.0, anomaly_scores=[0.1, 0.2, 0.3])
    t, s = data.timeline(run)
    assert t.tolist() == [0.0, 0.5, 1.0] and s.tolist() == pytest.approx([0.1, 0.2, 0.3])


# --- E2b ---------------------------------------------------------------------------------------------

from vad.types import VerifyResult, Window  # noqa: E402


def _vr(is_anomaly: bool, etype: EventType = EventType.NORMAL, caption: str = "c") -> VerifyResult:
    return VerifyResult(is_anomaly=is_anomaly, event_type=etype, caption=caption, confidence=0.9)


def _run_with_windows(verdicts, **meta) -> RunResult:
    wins = tuple(Window(video_id="v", start_s=10.0 * i, end_s=10.0 * i + 5, score=1.0 + i) for i in range(len(verdicts)))
    vrs = tuple(v for v in verdicts if v is not None) if all(v is not None for v in verdicts) else ()
    return RunResult(video_id="v", sample_fps=1.0, anomaly_scores=[0.1] * 60, candidate_windows=wins,
                     verify_results=vrs, metadata=meta)


def test_default_threshold_reads_metadata_with_source_and_never_invents():
    run = _run_with_windows([], threshold=1.1686, threshold_source="calib_normal")
    assert data.default_threshold(run) == (1.1686, "calib_normal")
    assert data.default_threshold(_run_with_windows([])) == (None, "")
    for bad in (None, "abc", float("nan"), float("inf")):
        assert data.default_threshold(_run_with_windows([], threshold=bad)) == (None, "")
    assert data.default_threshold(_run_with_windows([], threshold=0.5))[1] == "unknown"


def test_video_summaries_and_filter_windows_only(tmp_path):
    (tmp_path / "w.json").write_text(_run_json(
        "w", candidate_windows=[{"video_id": "w", "start_s": 0, "end_s": 5, "score": 1.0}],
        alerts=[], metadata={"tier2": "json"}))
    (tmp_path / "n.json").write_text(_run_json("n", metadata={"tier2": "none"}))
    (tmp_path / "ablation_A0.json").write_text(json.dumps({"ablation": "A0"}))
    (tmp_path / "bad.json").write_text("{nope")
    s = data.video_summaries(tmp_path)
    assert [x["video_id"] for x in s] == ["n", "w"]
    assert data.filter_video_ids(s) == ["w"]
    assert data.filter_video_ids(s, with_windows_only=False) == ["n", "w"]
    assert data.video_summaries(tmp_path / "missing") == []


def test_video_summaries_real_run_a4c_has_114_with_windows():
    root = SAMPLE_ROOT / "run_A4c"
    if not root.is_dir():
        pytest.skip("results/run_A4c not present")
    s = data.video_summaries(root)
    assert len(s) == 290
    assert len(data.filter_video_ids(s)) == 114


def test_find_video_path_handles_missing_manifest_missing_file_and_hostile_rows(tmp_path):
    (tmp_path / "vids").mkdir()
    (tmp_path / "vids" / "a.mp4").write_bytes(b"x")
    (tmp_path / "m.csv").write_text("video_id,path,dataset,split\na,vids/a.mp4,own,demo\nb,vids/gone.mp4,own,demo\n"
                                    "c,,own,demo\n", encoding="utf-8")
    assert data.find_video_path("a", ["m.csv"], tmp_path) == tmp_path / "vids" / "a.mp4"
    assert data.find_video_path("b", ["m.csv"], tmp_path) is None       # file not on this machine
    assert data.find_video_path("c", ["m.csv"], tmp_path) is None       # empty path
    assert data.find_video_path("zzz", ["m.csv"], tmp_path) is None     # id not in manifest
    assert data.find_video_path("a", ["nope.csv"], tmp_path) is None    # manifest missing
    (tmp_path / "bin.csv").write_bytes(b"\xff\xfe\x00bad")
    assert data.find_video_path("a", ["bin.csv"], tmp_path) is None     # undecodable manifest


def test_verdict_follows_is_anomaly_only_not_event_type():
    assert data.verdict_label(None) == "not verified"
    assert data.verdict_label(_vr(True, EventType.FIGHTING)) == "confirmed"
    assert data.verdict_label(_vr(False, EventType.FIGHTING)) == "rejected"    # the 37-window case
    assert data.verdict_label(_vr(True, EventType.NORMAL)) == "confirmed"


def test_window_bands_two_colours_and_unverified():
    run = _run_with_windows([_vr(True), _vr(False), _vr(False)])
    bands = data.window_bands(run)
    assert [b["verdict"] for b in bands] == ["confirmed", "rejected", "rejected"]
    assert bands[0]["start (s)"] == 0.0 and bands[1]["end (s)"] == 15.0
    assert data.vlm_rejected_count(run) == 2
    unverified = _run_with_windows([None, None])
    assert [b["verdict"] for b in data.window_bands(unverified)] == ["not verified"] * 2
    assert data.vlm_rejected_count(unverified) == 0
    assert data.window_bands(_run_with_windows([])) == []


def test_detection_rows_have_vlm_type_info_column():
    run = _run_with_windows([_vr(False, EventType.EXPLOSION, "boom"), _vr(True, EventType.ROBBERY)])
    rows = data.detection_rows(run)
    assert set(rows[0]) >= {"VLM verdict", "VLM type (info)", "caption", "tier-1 score"}
    assert "type" not in rows[0]
    assert rows[0]["VLM verdict"] == "rejected" and rows[0]["VLM type (info)"] == EventType.EXPLOSION.value
    assert data.detection_rows(_run_with_windows([None]))[0]["VLM type (info)"] == ""


def test_alert_card_uses_typer_type_and_keeps_vlm_type_separate():
    alert = Alert(video_id="v", start_s=10.0, end_s=15.0, event_type=EventType.ROAD_ACCIDENT, severity="high",
                  responders=("ambulance",), protocol="P-ACCIDENT", path=("camera:v", "event:RoadAccidents"),
                  caption="a van is robbed", confidence=0.95, camera="v")
    run = RunResult(video_id="v", sample_fps=1.0, anomaly_scores=[0.1] * 30, alerts=(alert,),
                    metadata={"event_typing": {"windows": [{"start_s": 10.0, "end_s": 15.0,
                                                            "vlm_event_type": "Robbery"}]}})
    card = data.alert_card(run, alert)
    assert card["type"] == EventType.ROAD_ACCIDENT.value and card["vlm_type"] == "Robbery"
    assert card["responders"] == ["ambulance"] and card["protocol"] == "P-ACCIDENT"
    assert card["path"] == ["camera:v", "event:RoadAccidents"] and card["caption"] == "a van is robbed"


def test_vlm_type_for_alert_falls_back_to_verify_results_then_empty():
    alert = Alert(video_id="v", start_s=0.0, end_s=5.0, event_type=EventType.FIGHTING, severity="low",
                  responders=(), protocol="P", path=())
    run = _run_with_windows([_vr(True, EventType.ROBBERY)])
    assert data.vlm_type_for_alert(run, alert) == EventType.ROBBERY.value
    other = Alert(video_id="v", start_s=99.0, end_s=100.0, event_type=EventType.FIGHTING, severity="low",
                  responders=(), protocol="P", path=())
    assert data.vlm_type_for_alert(run, other) == ""
    broken = RunResult(video_id="v", sample_fps=1.0, anomaly_scores=[0.1],
                       metadata={"event_typing": {"windows": [{"oops": 1}, "x"]}})
    assert data.vlm_type_for_alert(broken, alert) == ""


def test_real_run_a4c_alert_card_roundtrip():
    root = SAMPLE_ROOT / "run_A4c"
    if not (root / "Arrest024_x264.json").is_file():
        pytest.skip("results/run_A4c not present")
    run = data.load_run("Arrest024_x264", root)
    tau, src = data.default_threshold(run)
    assert tau == pytest.approx(1.1686) and src == "calib_normal"
    card = data.alert_card(run, run.alerts[0])
    assert card["type"] != card["vlm_type"] and card["vlm_type"] == "Robbery"
    assert data.vlm_rejected_count(run) == 1


# --- E2c ---------------------------------------------------------------------------------------------

def _abl(name: str, auc=0.5, lo=0.4, hi=0.6, **over) -> str:
    return json.dumps({"ablation": name, "auc_micro": auc, "ci_lo": lo, "ci_hi": hi,
                       "config": {"prompt_set": "ensemble", "use_memory": False}, **over})


def test_load_ablations_orders_rows_and_lists_skipped_files(tmp_path):
    (tmp_path / "ablation_A6.json").write_text(_abl("A6", 0.679, 0.584, 0.775))
    (tmp_path / "ablation_A0.json").write_text(_abl("A0", 0.4877, 0.383, 0.594))
    (tmp_path / "ablation_A3c.json").write_text(_abl("A3c"))
    (tmp_path / "ablation_A3.json").write_text(_abl("A3"))
    (tmp_path / "ablation_ZZ.json").write_text(_abl("ZZ"))
    (tmp_path / "ablation_bad.json").write_text("{nope")
    (tmp_path / "ablation_nan.json").write_text('{"ablation": "N", "auc_micro": NaN, "ci_lo": 0, "ci_hi": 1}')
    (tmp_path / "ablation_missing.json").write_text(json.dumps({"ablation": "M"}))
    rows, skipped = data.load_ablations(tmp_path)
    assert [r["ablation"] for r in rows] == ["A0", "A3", "A3c", "A6", "ZZ"]
    assert rows[0]["AUC (%)"] == 48.77 and rows[3]["tier"] == "1+2" and rows[0]["tier"] == "1"
    assert skipped == ["ablation_bad.json", "ablation_missing.json", "ablation_nan.json"]
    assert data.load_ablations(tmp_path / "nope") == ([], [])


def test_load_ablations_real_results_include_a0_to_a6():
    rows, skipped = data.load_ablations(SAMPLE_ROOT)
    names = [r["ablation"] for r in rows]
    assert skipped == []
    assert names[0] == "A0" and names[-1] == "A6" and "A4c" in names
    a6 = next(r for r in rows if r["ablation"] == "A6")
    assert a6["AUC (%)"] == pytest.approx(67.90, abs=0.01)
    assert {type(r["memory"]) for r in rows} == {str}
    assert next(r for r in rows if r["ablation"] == "A5c")["memory"] == "yes"
    assert next(r for r in rows if r["ablation"] == "A4c")["memory"] == "no"


def test_load_a6_missing_or_malformed_is_none_never_zeros(tmp_path):
    assert data.load_a6(tmp_path) is None
    (tmp_path / "ablation_A6.json").write_text("{nope")
    assert data.load_a6(tmp_path) is None
    (tmp_path / "ablation_A6.json").write_text(json.dumps({"ablation": "A6"}))
    assert data.load_a6(tmp_path) is None


def test_operating_and_cost_rows_from_real_a6():
    a6 = data.load_a6(SAMPLE_ROOT)
    assert a6 is not None
    rows = {r["metric"]: r for r in data.operating_rows(a6)}
    assert rows["window precision (%)"]["tier 1 (A4c)"] == pytest.approx(28.01, abs=0.01)
    assert rows["window precision (%)"]["tier 1+2 (A6)"] == pytest.approx(49.61, abs=0.01)
    assert rows["FA/hour (all normal videos)"]["tier 1 (A4c)"] == pytest.approx(20.64, abs=0.01)
    assert rows["FA/hour (excl. videos 935/924)"]["tier 1+2 (A6)"] == 0.0
    cost = {r["metric"]: r["value"] for r in data.cost_rows(a6)}
    assert cost["peak VRAM (GB)"] == 2.93 and cost["s/window median"] == pytest.approx(3.71, abs=0.01)


def test_cost_and_operating_rows_tolerate_missing_keys():
    assert data.cost_rows({}) == []
    assert data.cost_rows({"cost": {"peak_vram_gb": None, "s_per_window_p90": 1.234}}) == [
        {"metric": "s/window p90", "value": 1.23}]
    rows = data.operating_rows({"operating_before": {}, "operating_after": {}})
    assert all(r["tier 1 (A4c)"] is None and r["tier 1+2 (A6)"] is None for r in rows)


def test_pre_registered_claim_text_matches_protocol_sentence():
    assert "FA/hour and window precision" in data.PRE_REGISTERED_CLAIM
