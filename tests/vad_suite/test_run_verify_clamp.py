# -*- coding: utf-8 -*-
"""A5b — cửa sổ trôi quá cuối video (Phước, sửa lỗi sau lượt chạy 307 cửa sổ).

Lượt A5b để lại 2 cửa sổ lỗi vì tier-1 xuất cửa sổ dài hơn video:
  RoadAccidents002_x264  video 11.567 s, cửa sổ 11.0..12.0
  Vandalism017_x264      video 33.700 s, cửa sổ 32.0..34.0
Code cũ raise "thời điểm nằm ngoài video" -> window bị bỏ. Sửa: cắt biên cửa sổ về đúng độ dài
video TRƯỚC khi chia chunk; cửa sổ nằm hoàn toàn ngoài video vẫn raise (không bịa).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.run_verify as rv  # noqa: E402
from scripts.run_verify import clamp_window_to_video, plan_window_frames  # noqa: E402
from vad.types import Window  # noqa: E402

N_FRAMES, FPS = 347, 30.0          # video 11.5667 s, đúng ca RoadAccidents002
DURATION = N_FRAMES / FPS


def _window(start_s: float, end_s: float) -> Window:
    return Window(video_id="RoadAccidents002_x264", start_s=start_s, end_s=end_s, score=1.2)


def test_clamp_keeps_normal_window_unchanged():
    assert clamp_window_to_video(2.0, 5.0, DURATION) == (2.0, 5.0)


def test_clamp_trims_tail_to_video_end():
    start, end = clamp_window_to_video(11.0, 12.0, DURATION)
    assert start == 11.0
    assert end == pytest.approx(DURATION)


def test_clamp_rejects_window_fully_outside_video():
    with pytest.raises(ValueError, match="hoàn toàn ngoài video"):
        clamp_window_to_video(12.0, 13.0, DURATION)
    with pytest.raises(ValueError):
        clamp_window_to_video(-5.0, 0.0, DURATION)


def test_plan_clamps_bounds_and_returns_valid_distinct_frames(monkeypatch):
    """Ca thật: cửa sổ 11.0..12.0 trên video 347 frame -> phải ra 4 frame hợp lệ, khác nhau."""
    monkeypatch.setattr(rv, "probe_video", lambda _p: (N_FRAMES, FPS))
    plans = plan_window_frames("dummy.mp4", [_window(11.0, 12.0)], k=4)
    assert len(plans) == 1
    chunks = plans[0]
    assert not isinstance(chunks, ValueError), chunks
    (c_start, c_end), times, indices = chunks[0]
    assert c_start == 11.0
    assert c_end == pytest.approx(DURATION)          # biên phải đã bị cắt về cuối video
    assert all(0 <= i < N_FRAMES for i in indices)
    assert len(set(indices)) == 4                    # 4 frame khác nhau
    assert all(c_start < t < c_end for t in times)


def test_plan_normal_window_keeps_original_bounds(monkeypatch):
    """Không được đổi hành vi của 305 cửa sổ đã chạy đúng (biên giữ nguyên)."""
    monkeypatch.setattr(rv, "probe_video", lambda _p: (N_FRAMES, FPS))
    chunks = plan_window_frames("dummy.mp4", [_window(2.0, 5.0)], k=4)[0]
    assert not isinstance(chunks, ValueError)
    assert [(round(a, 6), round(b, 6)) for (a, b), _, _ in chunks] == [(2.0, 5.0)]


def test_plan_reports_error_when_window_too_short_after_clamp(monkeypatch):
    """Cửa sổ chỉ còn < 1 frame sau khi cắt -> ValueError cho RIÊNG window đó, không raise cả lượt."""
    monkeypatch.setattr(rv, "probe_video", lambda _p: (N_FRAMES, FPS))
    plans = plan_window_frames("dummy.mp4", [_window(DURATION - 0.001, 12.0)], k=4)
    assert isinstance(plans[0], ValueError)
