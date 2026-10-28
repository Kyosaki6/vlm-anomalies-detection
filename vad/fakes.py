# -*- coding: utf-8 -*-
"""Bản giả (deterministic, CPU, không tải model) của các Protocol — để mọi track làm việc từ ngày 1.

Thay bằng bản thật khi sẵn sàng; test hợp đồng ở tests/vad/test_contracts.py chạy với cả hai.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from vad.types import EventType, VerifyResult


class FakeEmbedder:
    """Embedding giả: hash nội dung -> vector ngẫu nhiên cố định, L2-normalized. Cùng đầu vào = cùng vector."""

    model_id = "fake-embedder-v0"

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, key: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_images(self, frames: Sequence[Any]) -> np.ndarray:
        return np.stack([self._vec(np.asarray(f).tobytes()) for f in frames]) if len(frames) else \
            np.empty((0, self.dim), dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t.encode("utf-8")) for t in texts]) if len(texts) else \
            np.empty((0, self.dim), dtype=np.float32)


class FakeVerifier:
    """Luôn trả một kết quả cố định (mặc định: Fighting, tin cậy 0.9). Dùng để dựng alert/dashboard."""

    def __init__(self, event_type: EventType = EventType.FIGHTING, confidence: float = 0.9,
                 is_anomaly: bool = True):
        self._r = VerifyResult(is_anomaly=is_anomaly, event_type=event_type,
                               caption="[fake] two people are fighting near a car", confidence=confidence)

    def verify(self, frames: Sequence[Any], context: dict) -> VerifyResult:
        return self._r
