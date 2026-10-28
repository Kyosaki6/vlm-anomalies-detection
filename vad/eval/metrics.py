# -*- coding: utf-8 -*-
"""Chỉ số đánh giá benchmark VAD: AUC frame-level (micro) + bootstrap CI theo VIDEO.

Dùng lại `decision.roc_auc` (Mann-Whitney có xử lý điểm trùng). Khoảng tin cậy lấy mẫu lại theo video,
KHÔNG theo frame, vì frame liền kề tương quan mạnh -> CI theo frame quá hẹp, tự lừa mình.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np

import decision


def frame_level_auc(scores_per_video: Sequence[np.ndarray], labels_per_video: Sequence[np.ndarray]) -> float:
    """Ghép mọi frame của mọi video rồi tính một AUC (micro). NaN nếu chỉ có một lớp."""
    if len(scores_per_video) != len(labels_per_video):
        raise ValueError("Số video điểm và nhãn lệch nhau")
    for i, (s, y) in enumerate(zip(scores_per_video, labels_per_video)):
        if len(s) != len(y):
            raise ValueError(f"Video #{i}: {len(s)} điểm nhưng {len(y)} nhãn (lệch fps/frame?)")
    if not scores_per_video:
        return float("nan")
    scores = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores_per_video])
    labels = np.concatenate([np.asarray(y, dtype=bool) for y in labels_per_video])
    return decision.roc_auc(scores.tolist(), labels.tolist())


def video_bootstrap_auc_ci(scores_per_video: Sequence[np.ndarray], labels_per_video: Sequence[np.ndarray],
                           n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """CI percentile cho AUC micro, lấy mẫu lại có hoàn lại các VIDEO."""
    n = len(scores_per_video)
    if n == 0:
        raise ValueError("Không có video")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci phải trong (0, 1)")
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_bootstraps):
        idx = [rng.randrange(n) for _ in range(n)]
        v = frame_level_auc([scores_per_video[i] for i in idx], [labels_per_video[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    tail = (1.0 - ci) / 2.0
    return decision._percentile(vals, tail), decision._percentile(vals, 1.0 - tail)


def false_alarms_per_hour(n_false_alarms: int, normal_seconds: float) -> float:
    """Số cảnh báo sai trên giờ video bình thường."""
    if normal_seconds <= 0:
        raise ValueError("normal_seconds phải > 0")
    return n_false_alarms / (normal_seconds / 3600.0)


def detection_latency_s(gt_start_s: float, first_alert_s: float | None) -> float | None:
    """Độ trễ phát hiện (giây). Không có cảnh báo -> None. Báo sớm hơn GT -> 0.0 (không âm)."""
    if first_alert_s is None:
        return None
    return max(0.0, first_alert_s - gt_start_s)
