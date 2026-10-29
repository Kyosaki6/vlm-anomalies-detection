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


def test_list_video_ids_skips_windows_dir_and_missing_root(tmp_path):
    assert data.list_video_ids(tmp_path / "nope") == []
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "a.json").write_text("{}")
    assert data.list_video_ids(tmp_path) == ["a", "b"]


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
