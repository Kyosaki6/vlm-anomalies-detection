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


# --- Bản nhanh (numpy) cho benchmark ~1.1 triệu frame ---------------------------------------------------
# `frame_level_auc` đi qua `decision.roc_auc` (vòng lặp Python): ~4.6 s cho 1.1M frame -> bootstrap 1000 lần
# mất hàng giờ. Bản dưới đây cho cùng kết quả (kiểm bằng test), chỉ khác tốc độ.

def _concat_checked(scores_per_video: Sequence[np.ndarray],
                    labels_per_video: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ghép mọi video -> (điểm, nhãn bool, chỉ số video của từng frame). Báo lỗi nếu lệch độ dài hoặc NaN/Inf."""
    if len(scores_per_video) != len(labels_per_video):
        raise ValueError("Số video điểm và nhãn lệch nhau")
    if not scores_per_video:
        raise ValueError("Không có video")
    lengths = []
    for i, (s, y) in enumerate(zip(scores_per_video, labels_per_video)):
        if len(s) != len(y):
            raise ValueError(f"Video #{i}: {len(s)} điểm nhưng {len(y)} nhãn (lệch fps/frame?)")
        lengths.append(len(s))
    scores = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores_per_video])
    labels = np.concatenate([np.asarray(y, dtype=bool) for y in labels_per_video])
    if not np.isfinite(scores).all():
        raise ValueError("Điểm có NaN/Inf: AUC sẽ sai lặng lẽ")
    vid = np.repeat(np.arange(len(lengths)), lengths)
    return scores, labels, vid


def _auc_from_groups(group_pos: np.ndarray, group_neg: np.ndarray) -> float:
    """AUC từ số (có trọng số) positive/negative trong từng nhóm điểm trùng, nhóm đã sắp tăng dần theo điểm.

    P(điểm pos > điểm neg) + 0.5 * P(trùng): chính là Mann-Whitney có xử lý ties.
    """
    n_pos, n_neg = float(group_pos.sum()), float(group_neg.sum())
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    neg_below = np.cumsum(group_neg) - group_neg
    return float((group_pos * (neg_below + 0.5 * group_neg)).sum() / (n_pos * n_neg))


def fast_frame_level_auc(scores_per_video: Sequence[np.ndarray], labels_per_video: Sequence[np.ndarray]) -> float:
    """Như `frame_level_auc` (micro, tie-aware) nhưng bằng numpy. NaN nếu chỉ có một lớp."""
    scores, labels, _ = _concat_checked(scores_per_video, labels_per_video)
    _, inv = np.unique(scores, return_inverse=True)
    n_groups = int(inv.max()) + 1
    pos = np.bincount(inv, weights=labels.astype(np.float64), minlength=n_groups)
    neg = np.bincount(inv, weights=(~labels).astype(np.float64), minlength=n_groups)
    return _auc_from_groups(pos, neg)


def fast_video_bootstrap_auc_ci(scores_per_video: Sequence[np.ndarray], labels_per_video: Sequence[np.ndarray],
                                n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Như `video_bootstrap_auc_ci` (cùng seed -> cùng chỉ số video được rút) nhưng không sắp xếp lại mỗi lần.

    Mẫu lại có hoàn lại = mỗi video có một số lần xuất hiện `m_v`; AUC micro của mẫu đó bằng AUC có trọng số
    với trọng số frame = `m_v`. Nhóm điểm trùng tính MỘT lần, mỗi lần bootstrap chỉ còn 2 phép bincount.
    """
    n = len(scores_per_video)
    if n == 0:
        raise ValueError("Không có video")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci phải trong (0, 1)")
    scores, labels, vid = _concat_checked(scores_per_video, labels_per_video)
    _, inv = np.unique(scores, return_inverse=True)
    n_groups = int(inv.max()) + 1
    pos_f, neg_f = labels.astype(np.float64), (~labels).astype(np.float64)
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_bootstraps):
        counts = np.bincount([rng.randrange(n) for _ in range(n)], minlength=n).astype(np.float64)
        w = counts[vid]
        v = _auc_from_groups(np.bincount(inv, weights=w * pos_f, minlength=n_groups),
                             np.bincount(inv, weights=w * neg_f, minlength=n_groups))
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
