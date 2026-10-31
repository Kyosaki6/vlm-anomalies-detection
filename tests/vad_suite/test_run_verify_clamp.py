# -*- coding: utf-8 -*-
"""A5b — cửa sổ trôi quá cuối video (Phước, sửa lỗi sau lượt chạy 307 cửa sổ).

48/307 cửa sổ có `end_s > duration` (lệch 0.03..0.93 s) do tier-1 xuất lố cuối video.
Chỉ 2 cửa sổ có mốc thời gian vượt hẳn video nên code cũ raise:
  RoadAccidents002_x264  video 11.567 s, cửa sổ 11.0..12.0
  Vandalism017_x264      video 33.700 s, cửa sổ 32.0..34.0
46 cửa sổ còn lại code cũ vẫn chạy, nhưng sau khi cắt biên về đúng độ dài video thì mốc
thời gian/frame indices đổi (chunk count không đổi) — thay đổi đã biết, chấp nhận.
Cửa sổ nằm hoàn toàn ngoài video vẫn raise (không bịa frame).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_verify as rv  # noqa: E402
from run_verify import clamp_window_to_video, plan_window_frames  # noqa: E402
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
    """Cửa sổ nằm hoàn toàn trong video -> biên giữ nguyên (không phải '305 cửa sổ' như bản cũ)."""
    monkeypatch.setattr(rv, "probe_video", lambda _p: (N_FRAMES, FPS))
    chunks = plan_window_frames("dummy.mp4", [_window(2.0, 5.0)], k=4)[0]
    assert not isinstance(chunks, ValueError)
    assert [(round(a, 6), round(b, 6)) for (a, b), _, _ in chunks] == [(2.0, 5.0)]


def test_plan_clamps_window_just_past_end_even_when_old_times_were_valid(monkeypatch):
    """46/48 ca thật: end_s vượt duration nhưng mốc cũ vẫn nằm trong video (code cũ KHÔNG lỗi).

    Video 10.0 s (300 frame @30fps), cửa sổ 2.0..10.4: mốc cũ 3.05/5.15/7.25/9.35 đều < 10.0
    nên code cũ chạy được; sau clamp biên phải = 10.0 -> mốc đổi thành 3/5/7/9.
    """
    monkeypatch.setattr(rv, "probe_video", lambda _p: (300, 30.0))
    chunks = plan_window_frames("dummy.mp4", [_window(2.0, 10.4)], k=4)[0]
    assert not isinstance(chunks, ValueError), chunks
    (c_start, c_end), times, indices = chunks[0]
    assert (c_start, c_end) == (2.0, 10.0)      # metadata["chunks"] lưu biên đã clamp
    assert times == [3.0, 5.0, 7.0, 9.0]
    assert indices == [90, 150, 210, 270]


def test_plan_reports_error_when_window_too_short_after_clamp(monkeypatch):
    """Cửa sổ chỉ còn < 1 frame sau khi cắt -> ValueError cho RIÊNG window đó, không raise cả lượt."""
    monkeypatch.setattr(rv, "probe_video", lambda _p: (N_FRAMES, FPS))
    plans = plan_window_frames("dummy.mp4", [_window(DURATION - 0.001, 12.0)], k=4)
    assert isinstance(plans[0], ValueError)
