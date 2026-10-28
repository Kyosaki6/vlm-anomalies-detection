# -*- coding: utf-8 -*-
"""Test hợp đồng file kết quả (Task C1). Thuần Python, không GPU, không model."""
import json

import numpy as np
import pytest

from vad.results import (
    DEFAULT_RESULTS_DIR,
    SCHEMA_VERSION,
    RunResult,
    make_metadata,
    result_path,
    windows_path,
)
from vad.types import Alert, EventType, VerifyResult, Window


def a_window(start=10.0, end=20.0, vid="v1", score=3.5):
    return Window(vid, start, end, score)


def a_verify(event=EventType.FIGHTING, conf=0.9, caption="two men fighting"):
    return VerifyResult(True, event, caption, conf)


def an_alert(start=10.0, end=20.0, vid="v1", camera="gate_a"):
    return Alert(vid, start, end, EventType.FIGHTING, "high", ("security", "police"),
                 "P-VIOLENCE", ("camera:gate_a", "event:Fighting", "protocol:P-VIOLENCE"),
                 caption="two men fighting", confidence=0.9, camera=camera)


def a_result(**overrides):
    base = dict(
        video_id="v1", sample_fps=1.0, anomaly_scores=(0.1, 0.2, 0.9),
        candidate_windows=(a_window(),), verify_results=(a_verify(),),
        alerts=(an_alert(),), metadata=make_metadata(model="siglip2", commit="abc123"),
    )
    base.update(overrides)
    return RunResult(**base)


# --- round trip -------------------------------------------------------------------

def test_round_trip_through_json(tmp_path):
    original = a_result()
    path = original.to_json(tmp_path / "v1.json")
    assert path.is_file()
    loaded = RunResult.from_json(path)
    assert loaded == original


def test_default_path_layout(tmp_path):
    r = a_result()
    assert r.to_json(root=tmp_path) == tmp_path / "v1.json"
    assert r.save_windows(root=tmp_path) == tmp_path / "windows" / "v1.json"
    assert result_path("v9", tmp_path) == tmp_path / "v9.json"
    assert windows_path("v9", tmp_path) == tmp_path / "windows" / "v9.json"


def test_array_of_scores_accepts_numpy():
    r = a_result(anomaly_scores=np.array([0.1, 0.2, 0.9]))
    assert r.anomaly_scores == (0.1, 0.2, 0.9)


def test_scores_are_rounded_not_noisy(tmp_path):
    r = a_result(anomaly_scores=(1 / 3,))
    data = json.loads(r.to_json(tmp_path / "v1.json").read_text(encoding="utf-8"))
    assert data["anomaly_scores"] == [0.333333]
    assert data["n_frames"] == 1


def test_unicode_caption_survives(tmp_path):
    r = RunResult("v1", 1.0, (0.1,), (a_window(),),
                  (VerifyResult(True, EventType.FIGHTING, "hai người đánh nhau ở cổng trường", 0.9),),
                  (an_alert(),), {})
    loaded = RunResult.from_json(r.to_json(tmp_path / "v1.json"))
    assert loaded.verify_results[0].caption == "hai người đánh nhau ở cổng trường"


def test_windows_file_round_trip(tmp_path):
    r = a_result()
    r.save_windows(root=tmp_path)
    loaded = RunResult.load_windows("v1", tmp_path)
    assert loaded.candidate_windows == r.candidate_windows
    assert loaded.verify_results == r.verify_results
    assert loaded.anomaly_scores == ()          # file windows không chứa điểm từng khung


# --- kiểm tra dữ liệu vào ---------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_sample_fps_must_be_positive(bad):
    with pytest.raises(ValueError, match="sample_fps"):
        a_result(sample_fps=bad)


def test_verify_results_must_align_with_windows():
    with pytest.raises(ValueError, match="cùng độ dài"):
        a_result(candidate_windows=(a_window(), a_window(30, 40)), verify_results=(a_verify(),))


def test_pending_verification_is_allowed():
    """Tier-1 xong, tier-2 chưa chạy: có cửa sổ nhưng chưa có kết quả VLM -> hợp lệ."""
    r = a_result(verify_results=())
    assert len(r.candidate_windows) == 1 and r.verify_results == ()


# --- đọc file hỏng / lạ ------------------------------------------------------------

def test_unsupported_schema_version_is_rejected(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({"schema_version": 999, "video_id": "v1", "sample_fps": 1.0}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        RunResult.from_json(p)


def test_missing_required_field_is_rejected(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "video_id": "v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu trường"):
        RunResult.from_json(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunResult.from_json(tmp_path / "nope.json")


def test_non_object_json_is_rejected(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        RunResult.from_json(p)


def test_unknown_fields_are_ignored_for_forward_compat(tmp_path):
    """File do phiên bản mới ghi thêm trường lạ -> code cũ vẫn đọc được, không nổ."""
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "video_id": "v1", "sample_fps": 2.0,
        "anomaly_scores": [0.5], "future_field": {"anything": True},
    }), encoding="utf-8")
    assert RunResult.from_json(p).video_id == "v1"


def test_old_file_without_camera_still_loads(tmp_path):
    """Alert trước khi có trường camera -> camera = "" (không raise)."""
    p = tmp_path / "v1.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "video_id": "v1", "sample_fps": 1.0,
        "anomaly_scores": [],
        "alerts": [{"video_id": "v1", "start_s": 0, "end_s": 1, "event_type": "Fighting",
                    "severity": "high", "responders": ["security"], "protocol": "P",
                    "path": ["event:Fighting"]}],
    }), encoding="utf-8")
    alert = RunResult.from_json(p).alerts[0]
    assert alert.camera == "" and alert.event_type is EventType.FIGHTING


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    a_result().to_json(tmp_path / "v1.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1.json"]


def test_json_is_human_readable(tmp_path):
    text = a_result().to_json(tmp_path / "v1.json").read_text(encoding="utf-8")
    assert text.lstrip().startswith("{") and "\n  " in text     # indent=2


def test_metadata_stamps_schema_and_timestamp():
    meta = make_metadata(model="siglip2", commit="abc", note="x")
    assert meta["model"] == "siglip2" and meta["note"] == "x"
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["timestamp"].endswith("+00:00")                   # UTC có offset rõ ràng


def test_default_results_dir_points_into_repo():
    assert DEFAULT_RESULTS_DIR.name == "results"


def test_committed_sample_run_matches_current_schema():
    """Nếu ai đó đổi schema mà quên sinh lại file mẫu, test này bắt được ngay."""
    p = DEFAULT_RESULTS_DIR / "sample_run.json"
    if not p.is_file():
        pytest.skip("chưa có results/sample_run.json")
    r = RunResult.from_json(p)
    assert r.video_id == "sample_run"
    assert r.alerts and all(a.camera for a in r.alerts)
    assert "GIẢ" in r.metadata.get("note", "")           # file mẫu phải tự khai là dữ liệu giả

# --- đầu vào xấu: chặn ở biên, không ghi file hỏng ----------------------------------

@pytest.mark.parametrize("bad", ["../evil", "a/b", "a\\b", "", "..", "."])
def test_video_id_with_path_chars_is_rejected(bad, tmp_path):
    with pytest.raises(ValueError, match="video_id"):
        result_path(bad, tmp_path)
    with pytest.raises(ValueError, match="video_id"):
        windows_path(bad, tmp_path)

def test_traversal_video_id_writes_nothing_outside_root(tmp_path):
    root = tmp_path / "results"
    with pytest.raises(ValueError):
        RunResult("../evil", 1.0, (0.1,)).to_json(root=root)
    assert not (tmp_path / "evil.json").exists()

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(bad):
    with pytest.raises(ValueError, match="NaN/Inf"):
        RunResult("v1", 1.0, (0.1, bad, 0.3))

def test_non_finite_scores_in_numpy_are_rejected():
    with pytest.raises(ValueError, match="NaN/Inf"):
        RunResult("v1", 1.0, np.array([0.1, np.nan]))

def test_written_json_is_strict(tmp_path):
    """File phải parse được bằng parser nghiêm ngặt (JS/jq): không NaN/Infinity."""
    p = RunResult("v1", 1.0, (0.1, 0.2)).to_json(root=tmp_path)
    json.loads(p.read_text(encoding="utf-8"),
               parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

@pytest.mark.parametrize("start,end", [(5.0, 2.0), (-1.0, 2.0), (0.0, float("inf"))])
def test_invalid_windows_are_rejected(start, end):
    with pytest.raises(ValueError, match="Cửa sổ"):
        RunResult("v1", 1.0, (0.1,), (Window("v1", start, end, 0.9),))

def test_sample_fps_nan_is_rejected():
    with pytest.raises(ValueError, match="sample_fps"):
        RunResult("v1", float("nan"), (0.1,))
