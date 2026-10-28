# -*- coding: utf-8 -*-
"""Video sampler (track T1): đọc video -> các khung RGB ở `sample_fps`, không giữ cả video trong RAM.

Căn chỉnh với `vad.temporal.ops.expand_to_frames`: mẫu k đại diện cho khoảng [k/sample_fps, (k+1)/sample_fps)
và lấy khung đầu khoảng, tức khung gốc round(k * fps / sample_fps). Số mẫu = ceil(n_frames * sample_fps / fps).

Đọc tuần tự (grab) thay vì seek: seek theo frame không chính xác với nhiều codec (nhất là UCF-Crime x264).
Đánh đổi: giải mã mọi khung nên chậm hơn seek, nhưng đúng.

`n_frames` từ container có thể sai vài khung; khi đọc hết sớm hơn dự kiến ta dừng, không raise.
Người gọi phải so `len(gt) == n_frames_thực` (frame_level_auc đã kiểm tra và báo lỗi nếu lệch).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("Thiếu opencv: pip install -r requirements.txt") from e


def probe_video(path: str | Path) -> tuple[int, float]:
    """-> (n_frames theo container, fps). Không mở được -> RuntimeError; không có file -> FileNotFoundError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV không mở được video: {p}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0 or not math.isfinite(fps):
        raise RuntimeError(f"Video có fps không hợp lệ ({fps}): {p}")
    return n, fps


def sample_indices(n_frames: int, fps: float, sample_fps: float) -> list[int]:
    """Chỉ số khung gốc cần lấy cho mỗi mẫu (đã kẹp vào [0, n_frames-1])."""
    if sample_fps <= 0 or fps <= 0:
        raise ValueError("sample_fps và fps phải > 0")
    if n_frames <= 0:
        return []
    count = math.ceil(n_frames * sample_fps / fps)
    return [min(int(round(k * fps / sample_fps)), n_frames - 1) for k in range(count)]


def iter_sampled_frames(path: str | Path, sample_fps: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
    """Sinh (thời_điểm_giây, khung RGB uint8 HxWx3). Thời điểm = k / sample_fps."""
    n_frames, fps = probe_video(path)
    targets = sample_indices(n_frames, fps, sample_fps)
    cap = cv2.VideoCapture(str(path))
    try:
        t_iter = iter(enumerate(targets))
        nxt = next(t_iter, None)
        frame_idx = 0
        while nxt is not None:
            k, target = nxt
            if not cap.grab():
                return  # container báo nhiều khung hơn thực tế: dừng, không raise
            if frame_idx == target:
                ok, bgr = cap.retrieve()
                if not ok:
                    return
                yield k / sample_fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                nxt = next(t_iter, None)
                # nhiều mẫu có thể trùng cùng một khung khi sample_fps > fps
                while nxt is not None and nxt[1] == frame_idx:
                    yield nxt[0] / sample_fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    nxt = next(t_iter, None)
            frame_idx += 1
    finally:
        cap.release()
