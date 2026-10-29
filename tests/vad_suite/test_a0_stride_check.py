# -*- coding: utf-8 -*-
"""Test helper của scripts/a0_stride_check.py (không tải model)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.a0_stride_check import embed_video, expand_repeat, iter_stride_frames  # noqa: E402


def _video(path: Path, n: int) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), (i * 5) % 255, dtype=np.uint8))
    w.release()


class _Emb:
    def embed_images(self, frames):
        return np.stack([np.array([1.0, float(f.mean())]) for f in frames]).astype(np.float32)


def test_stride_picks_every_16th_frame(tmp_path):
    v = tmp_path / "v.mp4"
    _video(v, 50)
    assert len(list(iter_stride_frames(v, 16))) == 4   # khung 0,16,32,48


def test_embed_video_batches_and_counts(tmp_path):
    v = tmp_path / "v.mp4"
    _video(v, 50)
    assert embed_video(v, _Emb(), 16, batch=3).shape == (4, 2)   # batch lẻ vẫn đủ mẫu


def test_expand_repeat_truncates_like_lavad():
    out = expand_repeat(np.array([1.0, 2.0, 3.0]), 16, 40)
    assert len(out) == 40 and out[0] == 1.0 and out[15] == 1.0 and out[16] == 2.0 and out[39] == 3.0


def test_expand_repeat_pads_when_video_shorter_than_container():
    out = expand_repeat(np.array([1.0, 2.0]), 16, 40)   # chỉ đủ 32 khung -> đệm mẫu cuối
    assert len(out) == 40 and (out[32:] == 2.0).all()


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        list(iter_stride_frames(tmp_path / "nope.mp4", 16))
