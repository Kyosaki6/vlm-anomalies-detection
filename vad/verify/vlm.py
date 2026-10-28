# -*- coding: utf-8 -*-
"""Tier-2: VLM sinh ngôn ngữ THẬT (track T2, Phước). Bắt buộc theo đề tài đã đăng ký.

Việc cần làm (chưa làm, KHÔNG giả vờ có):
  1. Chọn model (ứng viên lớp 2-3B, 4-bit, <6 GB VRAM — phải ĐO, không đoán); ghi quyết định vào notes/.
  2. Prompt lấy từ configs/prompts.yaml (khoá `verifier`); ép JSON, parse bằng `parse_verifier_json`.
  3. Ghi kết quả mỗi Window ra file riêng để chạy lại được (Colab có thể rớt).
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from vad.types import EventType, VerifyResult


def parse_verifier_json(raw: str) -> VerifyResult:
    """Parse đầu ra VLM -> VerifyResult. Chịu được text thừa quanh JSON và ```json fence.

    Hỏng/thiếu khoá -> ValueError (người gọi quyết định retry hay đánh dấu lỗi; không đoán bừa).
    """
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Không thấy JSON trong đầu ra VLM: {raw[:80]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON hỏng: {e}") from e
    for key in ("is_anomaly", "event_type", "caption", "confidence"):
        if key not in data:
            raise ValueError(f"Thiếu khoá {key!r} trong JSON VLM")
    conf = float(data["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence ngoài [0, 1]: {conf}")
    return VerifyResult(
        is_anomaly=bool(data["is_anomaly"]),
        event_type=EventType.from_label(str(data["event_type"])),
        caption=str(data["caption"]),
        confidence=conf,
    )


class VLMVerifier:
    def __init__(self, model_id: str, device: str | None = None, quantization: str = "4bit"):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization

    def verify(self, frames: Sequence[Any], context: dict) -> VerifyResult:
        raise NotImplementedError("VLMVerifier.verify: chưa nối model (việc của T2, Phước)")
