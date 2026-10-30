# -*- coding: utf-8 -*-
"""Test scripts/calibrate_threshold.py (TH5): ngưỡng từ calib_normal, đo vận hành trên test. CPU, không tải model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import calibrate_threshold as ct  # noqa: E402
import run_benchmark as rb  # noqa: E402
from vad import pipeline as pl  # noqa: E402
from vad.datasets.ucf_crime import Annotation  # noqa: E402

# --- quantile_threshold -------------------------------------------------------------------

@pytest.mark.parametrize("fpr", [0.01, 0.05, 0.2, 0.5])
def test_threshold_never_exceeds_budget_on_continuous_scores(fpr):
    x = np.random.default_rng(0).standard_normal(1000)
    thr = ct.quantile_threshold(x, fpr)
    assert ct.achieved_fpr(x, thr) <= fpr + 1e-12
    assert ct.achieved_fpr(x, thr) > fpr - 0.01                 # và sát hạn mức, không quá thận trọng

def test_threshold_with_heavy_ties_stays_within_budget():
    x = np.round(np.random.default_rng(1).random(2000), 1)       # chỉ 11 giá trị
    thr = ct.quantile_threshold(x, 0.05)
    assert ct.achieved_fpr(x, thr) <= 0.05

def test_all_equal_scores_threshold_above_max_so_nothing_alarms():
    x = np.full(100, 0.7)
    thr = ct.quantile_threshold(x, 0.05)
    assert thr > 0.7 and ct.achieved_fpr(x, thr) == 0.0

def test_threshold_changes_with_non_default_budget():
    x = np.random.default_rng(2).standard_normal(5000)
    assert ct.quantile_threshold(x, 0.01) > ct.quantile_threshold(x, 0.20)

@pytest.mark.parametrize("bad", [np.array([]), np.array([0.1, np.nan]), np.array([0.1, np.inf])])
def test_hostile_calib_scores_rejected(bad):
    with pytest.raises(ct.CalibrationError):
        ct.quantile_threshold(bad, 0.05)

@pytest.mark.parametrize("fpr", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_target_fpr_rejected(fpr):
    with pytest.raises(ct.CalibrationError):
        ct.quantile_threshold(np.arange(10.0), fpr)

# --- cấu hình không hiệu chỉnh được -------------------------------------------------------

@pytest.mark.parametrize("aid", ["A3", "A4", "A5", "A5c"])
def test_uncalibratable_configs_refused(aid):
    with pytest.raises(ct.CalibrationError):
        ct.check_calibratable(pl.ABLATIONS[aid])

@pytest.mark.parametrize("aid", ["A0", "A1", "A2", "A3c", "A4c"])
def test_calibratable_configs_accepted(aid):
    ct.check_calibratable(pl.ABLATIONS[aid])

# --- operating_point ----------------------------------------------------------------------

def _row(vid, n_frames, fps=30.0):
    return {"video_id": vid, "n_frames": n_frames, "fps": fps}

def _ann(vid, *segs):
    return Annotation(video_id=vid, label="Abuse" if segs else "Normal", segments=tuple(segs))

CFG = pl.PipelineConfig(model_id="m", sample_fps=1.0, min_len_s=1.0, merge_gap_s=0.0)

def test_operating_point_counts_false_alarms_recall_and_latency():
    rows = [_row("A", 600), _row("N", 3600)]                     # A: 20 s, anomalía giây 10..14 (frame 300..449)
    ann = {"A": _ann("A", (300, 449)), "N": _ann("N")}
    a = np.zeros(20)
    a[12:15] = 1.0                                               # phát hiện từ giây 12 -> trễ 2 s
    n = np.zeros(120)
    n[30:33] = 1.0
    n[80:82] = 1.0                                               # 2 cửa sổ báo giả trên 120 s
    op = ct.operating_point(rows, ann, [a, n], CFG, threshold=0.5)
    assert (op["incidents"], op["incidents_detected"], op["incident_recall"]) == (1, 1, 1.0)
    assert op["mean_latency_s"] == pytest.approx(2.0)
    assert op["false_alarm_windows"] == 2
    assert op["fa_per_hour"] == pytest.approx(2 / (120 / 3600.0))    # 3600 frame / 30 fps = 120 s = 60 FA/h
    assert op["normal_sample_fpr"] == pytest.approx(5 / 120)
    assert op["windows_total"] == 3

def test_missed_incident_has_no_latency_and_zero_recall():
    rows = [_row("A", 600), _row("N", 300)]
    ann = {"A": _ann("A", (300, 449)), "N": _ann("N")}
    op = ct.operating_point(rows, ann, [np.zeros(20), np.zeros(10)], CFG, threshold=0.5)
    assert op["incident_recall"] == 0.0 and op["mean_latency_s"] is None and op["fa_per_hour"] == 0.0

def test_window_outside_the_incident_is_not_a_hit():
    rows = [_row("A", 600), _row("N", 300)]
    ann = {"A": _ann("A", (300, 449)), "N": _ann("N")}
    a = np.zeros(20)
    a[2:4] = 1.0                                                 # cảnh báo trước sự cố (giây 2..4), sự cố giây 10..15
    op = ct.operating_point(rows, ann, [a, np.zeros(10)], CFG, threshold=0.5)
    assert op["incidents_detected"] == 0

def test_two_segments_in_one_video_are_two_incidents():
    rows = [_row("A", 900), _row("N", 300)]
    ann = {"A": _ann("A", (60, 119), (600, 659)), "N": _ann("N")}
    a = np.zeros(30)
    a[2:4] = 1.0                                                 # chỉ trúng đoạn đầu (giây 2..4)
    op = ct.operating_point(rows, ann, [a, np.zeros(10)], CFG, threshold=0.5)
    assert (op["incidents"], op["incidents_detected"]) == (2, 1)

def test_25fps_video_uses_its_own_fps_for_gt_seconds():
    rows = [_row("A", 500, fps=25.0), _row("N", 250, fps=25.0)]  # A dài 20 s; sự cố frame 250..374 = giây 10..15
    ann = {"A": _ann("A", (250, 374)), "N": _ann("N")}
    a = np.zeros(20)
    a[10:12] = 1.0
    op = ct.operating_point(rows, ann, [a, np.zeros(10)], CFG, threshold=0.5)
    assert op["incidents_detected"] == 1 and op["mean_latency_s"] == pytest.approx(0.0)

def test_no_normal_video_in_test_is_an_error_not_division_by_zero():
    rows = [_row("A", 600)]
    with pytest.raises(ct.CalibrationError):
        ct.operating_point(rows, {"A": _ann("A", (300, 449))}, [np.zeros(20)], CFG, threshold=0.5)

# --- _score_videos fail loud --------------------------------------------------------------

def test_score_videos_rejects_length_mismatch(tmp_path):
    from vad.embed.cache import save_embeddings
    cfg = pl.PipelineConfig(model_id="m", sample_fps=1.0)
    save_embeddings(np.ones((5, 4), dtype=np.float32), "m", "V", 1.0, tmp_path)
    prompts = (np.ones((1, 4), np.float32), np.ones((1, 4), np.float32))
    with pytest.raises(rb.BenchmarkInputError, match="V"):
        ct._score_videos([_row("V", 3000)], cfg, prompts, tmp_path, None)
