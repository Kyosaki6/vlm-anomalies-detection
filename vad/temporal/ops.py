# -*- coding: utf-8 -*-
"""Tầng thời gian cho điểm tier-1 (track T1, Thịnh). Thuần numpy, test được trên CPU.

Thứ tự đề xuất: robust_normalize -> window_topk_pool -> ema/median_smooth -> candidate_windows.
Mọi hàm nhận/trả mảng 1-D float, không sửa mảng đầu vào.
"""
from __future__ import annotations

import numpy as np

from vad.types import Window


def robust_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(x - median) / (1.4826 * MAD). Chuẩn hoá theo từng video để bù khác biệt camera/ánh sáng."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)


def ema(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Trung bình trượt mũ. alpha trong (0, 1]; alpha=1 = không làm mượt."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha phải trong (0, 1], nhận {alpha}")
    x = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(x)
    if x.size == 0:
        return out
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def median_smooth(scores: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median filter, kernel lẻ; biên được đệm bằng giá trị mép."""
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"kernel phải là số lẻ >= 1, nhận {kernel}")
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0 or kernel == 1:
        return x.copy()
    pad = kernel // 2
    padded = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return np.median(windows, axis=1)


def window_topk_pool(scores: np.ndarray, window: int, k: int = 3) -> np.ndarray:
    """Tại mỗi t: trung bình k giá trị lớn nhất trong cửa sổ trung tâm `window` mẫu."""
    if window < 1 or k < 1:
        raise ValueError("window và k phải >= 1")
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    half = window // 2
    out = np.empty_like(x)
    for t in range(x.size):
        seg = x[max(0, t - half): t + half + 1]
        kk = min(k, seg.size)
        out[t] = np.sort(seg)[-kk:].mean()
    return out


def candidate_windows(scores: np.ndarray, sample_fps: float, threshold: float, video_id: str = "",
                      min_len_s: float = 1.0, merge_gap_s: float = 2.0) -> list[Window]:
    """Đoạn liên tiếp có score >= threshold -> Window. Gộp đoạn cách nhau < merge_gap_s, bỏ đoạn < min_len_s.

    Mỗi mẫu t phủ [t/fps, (t+1)/fps). Score của Window = max trong đoạn.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps phải > 0")
    x = np.asarray(scores, dtype=np.float64)
    above = x >= threshold
    runs: list[list[int]] = []
    t = 0
    while t < above.size:
        if above[t]:
            s = t
            while t + 1 < above.size and above[t + 1]:
                t += 1
            runs.append([s, t])
        t += 1
    merge_gap = int(round(merge_gap_s * sample_fps))
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for s, e in merged:
        start_s, end_s = s / sample_fps, (e + 1) / sample_fps
        if end_s - start_s >= min_len_s:
            out.append(Window(video_id=video_id, start_s=start_s, end_s=end_s,
                              score=float(x[s:e + 1].max())))
    return out


def expand_to_frames(scores: np.ndarray, sample_fps: float, fps: float, n_frames: int) -> np.ndarray:
    """Điểm theo mẫu (sample_fps) -> điểm theo từng frame gốc, để so với GT frame-level.

    Frame f lấy điểm của mẫu gần nhất về phía trước: idx = floor(f / fps * sample_fps), cắt vào biên.
    """
    if sample_fps <= 0 or fps <= 0:
        raise ValueError("sample_fps và fps phải > 0")
    x = np.asarray(scores, dtype=np.float64)
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Không có điểm nào để mở rộng")
    idx = np.floor(np.arange(n_frames) / fps * sample_fps).astype(int)
    return x[np.clip(idx, 0, x.size - 1)]
