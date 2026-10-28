# -*- coding: utf-8 -*-
"""vad/interfaces.py — Protocol cho các thành phần thay thế được.

Mỗi track viết bản thật; `vad/fakes.py` có bản giả để track khác làm việc ngay.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from vad.types import Alert, EventType, VerifyResult, Window


@runtime_checkable
class Embedder(Protocol):
    """Image/text encoder dùng chung không gian vector. Đầu ra đã L2-normalize, float32."""

    model_id: str

    def embed_images(self, frames: Sequence[Any]) -> np.ndarray:
        """frames: list PIL.Image hoặc ndarray HxWx3 RGB uint8 -> (N, D)."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """-> (N, D)."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """Tier-2: VLM sinh ngôn ngữ xác minh một cửa sổ nghi ngờ."""

    def verify(self, frames: Sequence[Any], context: dict) -> VerifyResult:
        """frames: vài khung đại diện của Window. context: {'video_id', 'start_s', 'end_s', ...}."""
        ...


@runtime_checkable
class AlertPolicy(Protocol):
    """Biến (sự kiện đã xác minh + ngữ cảnh) thành Alert."""

    def to_alert(self, window: Window, result: VerifyResult, camera: str | None = None,
                 zone: str | None = None, hour: int | None = None) -> Alert | None:
        """Trả None nếu không đáng cảnh báo (bình thường / dưới ngưỡng)."""
        ...


__all__ = ["Embedder", "Verifier", "AlertPolicy", "EventType"]
