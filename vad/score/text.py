# -*- coding: utf-8 -*-
"""Điểm tier-1 từ embedding đã cache (track T1). Chỉ numpy: chạy CPU, không cần model.

Prompt dùng cho PHÁT HIỆN phải GENERIC (không gọi tên lớp) để tránh rò nhãn; tên lớp chỉ dùng ở event typer.
Xem configs/prompts.yaml.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LOGIT_SCALE = 100.0


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def prompt_centroid(text_embs: np.ndarray) -> np.ndarray:
    """Ensemble nhiều paraphrase: trung bình embedding rồi chuẩn hoá lại -> (D,)."""
    embs = _l2(np.atleast_2d(text_embs))
    return _l2(embs.mean(axis=0))


def text_anomaly_scores(frame_embs: np.ndarray, normal_embs: np.ndarray, abnormal_embs: np.ndarray,
                        logit_scale: float = DEFAULT_LOGIT_SCALE) -> np.ndarray:
    """Xác suất 'bất thường' theo từng frame: softmax 2 lớp trên cosine * logit_scale. -> (T,)

    frame_embs: (T, D); normal_embs/abnormal_embs: (P, D) hoặc (D,), mỗi bên được ensemble.
    """
    f = _l2(frame_embs)
    n = prompt_centroid(normal_embs)
    a = prompt_centroid(abnormal_embs)
    margin = (f @ a - f @ n) * logit_scale
    return 1.0 / (1.0 + np.exp(-np.clip(margin, -60.0, 60.0)))


def memory_anomaly_scores(frame_embs: np.ndarray, normal_bank: np.ndarray) -> np.ndarray:
    """1 - max cosine tới ngân hàng embedding bình thường (few-shot normal reference). -> (T,)"""
    bank = _l2(np.atleast_2d(normal_bank))
    if bank.shape[0] == 0:
        raise ValueError("normal_bank rỗng")
    sims = _l2(frame_embs) @ bank.T
    return 1.0 - np.clip(sims.max(axis=1), -1.0, 1.0)
