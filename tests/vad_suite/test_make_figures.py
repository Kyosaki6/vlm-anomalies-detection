# -*- coding: utf-8 -*-
"""Test C7f — `scripts/make_figures.py`. Không cần matplotlib, không GPU, không video thật.

Điều phải khóa là LUẬT CHỌN, không phải hình đẹp: nếu ai đó sửa luật (đổi "nhiều giây nhất" thành
"đẹp nhất") thì test phải đỏ. Mỗi danh mục có một test riêng, cộng test danh mục rỗng.
"""
import json

import pytest

from scripts.make_figures import (
    CATEGORIES,
    FigureError,
    _load_runs,
    _threshold_from,
    main,
    select_cases,
)
from vad.datasets.ucf_crime import Annotation
from vad.results import RunResult
from vad.types import EventType, VerifyResult, Window


# --- dụng cụ ---------------------------------------------------------------------

def _ann(video_id: str, label: str, segments=((0, 900),)) -> Annotation:
    return Annotation(video_id=video_id, label=label, segments=tuple(segments))


def _verify(anomaly: bool, etype: EventType = EventType.FIGHTING, caption: str = "x") -> VerifyResult:
    return VerifyResult(is_anomaly=anomaly, event_type=etype, caption=caption, confidence=0.9)


def _run(video_id: str, windows, verdicts, scores=(0.5, 1.5), metadata=None) -> RunResult:
    return RunResult(video_id=video_id, sample_fps=1.0, anomaly_scores=tuple(scores),
                     candidate_windows=tuple(windows), verify_results=tuple(verdicts),
                     metadata=metadata if metadata is not None else {"threshold": 1.0})


def _window(video_id: str, start: float, end: float, score: float) -> Window:
    return Window(video_id=video_id, start_s=start, end_s=end, score=score)


def _by_category(picks):
    return {p.category: p for p in picks}


def test_returns_one_pick_per_category_including_empty_ones():
    picks, empty = select_cases({}, {})
    assert [p.category for p in picks] == list(CATEGORIES)
    assert empty == list(CATEGORIES)
    assert all(p.missing for p in picks)


# --- 1) bắt đúng ---------------------------------------------------------------

def test_confirmed_hit_takes_largest_gt_overlap_among_confirmed_only():
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 0, 10, 2.0), _window(vid, 20, 60, 1.9), _window(vid, 100, 120, 5.0)],
               [_verify(True), _verify(True), _verify(False)])       # cửa sổ 3 bị loại
    pick = _by_category(select_cases({vid: run}, {vid: _ann(vid, "Fighting", [(0, 1800)])})[0])["confirmed_hit"]
    assert pick.window_index == 1          # overlap 40 s > 10 s; cửa sổ 100-120 bị loại
    assert pick.overlap_s == pytest.approx(40.0)
    assert pick.is_confirmed is True
    assert pick.label == "Fighting"


def test_confirmed_hit_ignores_windows_outside_gt():
    vid = "Fighting001_x264"
    # GT chỉ 0..30 s; cửa sổ duy nhất được xác nhận nằm 100..110 s -> không phải "bắt đúng"
    run = _run(vid, [_window(vid, 100, 110, 2.0)], [_verify(True)])
    picks, empty = select_cases({vid: run}, {vid: _ann(vid, "Fighting", [(0, 900)])})
    assert "confirmed_hit" in empty


# --- 2) & 3) báo động nhầm ------------------------------------------------------

def test_removed_false_alarm_only_looks_at_rejected_windows_on_normal_videos():
    vid = "Normal_Videos_001_x264"
    run = _run(vid, [_window(vid, 0, 10, 1.5), _window(vid, 20, 30, 2.8), _window(vid, 40, 50, 3.9)],
               [_verify(False), _verify(False), _verify(True)])       # cửa sổ điểm cao nhất bị XÁC NHẬN
    pick = _by_category(select_cases({vid: run}, {vid: _ann(vid, "Normal", ())})[0])["removed_false_alarm"]
    assert pick.window_index == 1          # 2.8 là cao nhất trong các cửa sổ BỊ LOẠI
    assert pick.is_confirmed is False
    assert pick.label == "Normal"


def test_surviving_false_alarm_is_a_normal_video_with_a_confirmed_window():
    vid = "Normal_Videos_002_x264"
    run = _run(vid, [_window(vid, 0, 10, 3.2)], [_verify(True, EventType.BURGLARY, "người mở cửa")])
    pick = _by_category(select_cases({vid: run}, {vid: _ann(vid, "Normal", ())})[0])["surviving_false_alarm"]
    assert pick.window_index == 0
    assert pick.is_confirmed is True
    assert pick.caption == "người mở cửa"


# --- 4) bỏ sót ------------------------------------------------------------------

def test_missed_incident_is_a_video_with_no_windows_and_best_score_inside_gt():
    # cả hai video đều không có cửa sổ; chọn video có điểm TB trong đoạn GT cao hơn
    # đoạn GT 0..900 frame = 0..30 s nhưng chuỗi điểm chỉ có 4 mẫu -> lấy cả 4 mẫu (cắt theo mảng)
    shallow = _run("Shooting001_x264", [], [], scores=(0.2, 0.2, 0.2, 0.2))
    deep = _run("Shooting002_x264", [], [], scores=(0.2, 1.0, 1.2, 0.2))
    anns = {"Shooting001_x264": _ann("Shooting001_x264", "Shooting"),
            "Shooting002_x264": _ann("Shooting002_x264", "Shooting")}
    pick = _by_category(select_cases({"Shooting001_x264": shallow,
                                      "Shooting002_x264": deep}, anns)[0])["missed_incident"]
    assert pick.video_id == "Shooting002_x264"
    assert pick.window_index is None
    assert pick.evidence == pytest.approx((0.2 + 1.0 + 1.2 + 0.2) / 4)


def test_missed_incident_ignores_videos_that_do_have_windows():
    caught = _run("Shooting001_x264", [_window("Shooting001_x264", 0, 10, 2.0)], [_verify(True)],
                  scores=(2.0, 2.0))
    picks, empty = select_cases({"Shooting001_x264": caught},
                                {"Shooting001_x264": _ann("Shooting001_x264", "Shooting")})
    assert "missed_incident" in empty


# --- 5) mất cửa sổ đúng ---------------------------------------------------------

def test_lost_true_positive_requires_gt_overlap_among_rejected_windows():
    vid = "Robbery001_x264"
    run = _run(vid, [_window(vid, 100, 150, 3.0), _window(vid, 10, 40, 2.0)],
               [_verify(False, EventType.NORMAL), _verify(False, EventType.NORMAL)])
    # GT là 0..60 s -> chỉ cửa sổ 10..40 giao (30 s); cửa sổ điểm cao hơn nằm ngoài GT
    pick = _by_category(select_cases({vid: run}, {vid: _ann(vid, "Robbery", [(0, 1800)])})[0])["lost_true_positive"]
    assert pick.window_index == 1
    assert pick.overlap_s == pytest.approx(30.0)


def test_lost_true_positive_ignores_confirmed_windows():
    vid = "Robbery002_x264"
    run = _run(vid, [_window(vid, 10, 40, 2.0)], [_verify(True)])
    picks, empty = select_cases({vid: run}, {vid: _ann(vid, "Robbery", [(0, 1800)])})
    assert "lost_true_positive" in empty


# --- ngưỡng & đọc file ----------------------------------------------------------

def test_threshold_comes_from_metadata_and_rejects_mixed_values():
    a = _run("v1", [], [], metadata={"threshold": 1.1686})
    b = _run("v2", [], [], metadata={"threshold": 1.1686})
    assert _threshold_from({"a": a, "b": b}, None) == pytest.approx(1.1686)
    c = _run("v3", [], [], metadata={"threshold": 2.0})
    with pytest.raises(FigureError):
        _threshold_from({"a": a, "c": c}, None)


def test_missing_threshold_must_not_be_guessed():
    run = _run("v1", [], [], metadata={})
    with pytest.raises(FigureError):
        _threshold_from({"v1": run}, None)
    assert _threshold_from({"v1": run}, 1.1686) == pytest.approx(1.1686)
    with pytest.raises(FigureError):
        _threshold_from({"v1": run}, float("nan"))


def test_load_runs_skips_index_file_and_names_the_broken_one(tmp_path):
    good = tmp_path / "Fighting001_x264.json"
    good.write_text(json.dumps(_run("Fighting001_x264", [], []).to_dict()), encoding="utf-8")
    (tmp_path / "_index.json").write_text('{"not": "a run"}', encoding="utf-8")
    runs = _load_runs(tmp_path)
    assert list(runs) == ["Fighting001_x264"]

    (tmp_path / "Broken_x264.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(FigureError) as err:
        _load_runs(tmp_path)
    assert "Broken_x264.json" in str(err.value)


# --- CLI dry-run và smoke vẽ hình ----------------------------------------------

def _write_fixture(tmp_path):
    ann_path = tmp_path / "ann.txt"
    ann_path.write_text("Fighting001_x264 Fighting 0 900\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    run = _run("Fighting001_x264", [_window("Fighting001_x264", 0, 30, 2.0)],
               [_verify(True, EventType.FIGHTING, "hai người đánh nhau")])
    (results / "Fighting001_x264.json").write_text(json.dumps(run.to_dict()), encoding="utf-8")
    return ann_path, results


def test_dry_run_writes_nothing(tmp_path):
    ann_path, results = _write_fixture(tmp_path)
    out = tmp_path / "figures"
    code = main(["--annotation", str(ann_path), "--results-dir", str(results),
                 "--out-dir", str(out), "--dry-run"])
    assert code == 0
    assert not out.exists()


def test_plot_writes_a_png_and_selection_json(tmp_path):
    pytest.importorskip("matplotlib")
    ann_path, results = _write_fixture(tmp_path)
    out = tmp_path / "figures"
    code = main(["--annotation", str(ann_path), "--results-dir", str(results), "--out-dir", str(out)])
    assert code == 0
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 1                    # chỉ danh mục 'confirmed_hit' có dữ liệu
    assert pngs[0].stat().st_size > 5000      # PNG thật, không phải file rỗng
    payload = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert payload["selection"][0]["category"] == "confirmed_hit"
    assert "removed_false_alarm" in payload["empty_categories"]
    assert payload["figures"] == [pngs[0].name]
