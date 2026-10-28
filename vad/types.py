# -*- coding: utf-8 -*-
"""vad/types.py — hợp đồng dữ liệu (frozen dataclass) giữa 5 track.

Mọi track chỉ import từ đây + `vad.interfaces`. Đổi file này = báo cả nhóm.
Python 3.10 compatible (CI chạy 3.10): không dùng StrEnum.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    """13 lớp bất thường của UCF-Crime + NORMAL. Giá trị = tên lớp gốc của dataset."""
    ABUSE = "Abuse"
    ARREST = "Arrest"
    ARSON = "Arson"
    ASSAULT = "Assault"
    BURGLARY = "Burglary"
    EXPLOSION = "Explosion"
    FIGHTING = "Fighting"
    ROAD_ACCIDENT = "RoadAccidents"
    ROBBERY = "Robbery"
    SHOOTING = "Shooting"
    SHOPLIFTING = "Shoplifting"
    STEALING = "Stealing"
    VANDALISM = "Vandalism"
    NORMAL = "Normal"
    UNKNOWN = "Unknown"  # VLM nói bất thường nhưng không map được lớp nào

    @classmethod
    def from_label(cls, label: str) -> "EventType":
        """Tên lớp dataset/VLM -> EventType; không nhận ra -> UNKNOWN (không raise)."""
        key = "".join(ch for ch in label.lower() if ch.isalnum())
        for member in cls:
            if "".join(ch for ch in member.value.lower() if ch.isalnum()) == key:
                return member
        # UCF viết 'RoadAccidents', người viết hay gõ 'road accident'
        if key in ("roadaccident", "roadaccidents", "trafficaccident"):
            return cls.ROAD_ACCIDENT
        return cls.UNKNOWN


@dataclass(frozen=True)
class VideoRef:
    """Một video trong manifest. `n_frames`/`fps` là của video GỐC (30 fps với UCF)."""
    video_id: str            # ví dụ 'Abuse028_x264' (không đuôi file)
    path: str                # đường dẫn tuyệt đối hoặc tương đối theo dataset root
    dataset: str             # 'ucf_crime' | 'shanghaitech' | 'own'
    split: str               # 'test' | 'calib_normal' | 'demo'
    n_frames: int | None = None
    fps: float = 30.0


@dataclass(frozen=True)
class Window:
    """Cửa sổ nghi ngờ trong một video (giây tính từ đầu video)."""
    video_id: str
    start_s: float
    end_s: float
    score: float             # điểm tier-1 đã chuẩn hoá, cao hơn = bất thường hơn

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class VerifyResult:
    """Đầu ra tier-2 (VLM sinh ngôn ngữ) cho một Window."""
    is_anomaly: bool
    event_type: EventType
    caption: str             # mô tả tiếng Anh ngắn: VLM nói gì thấy trong cảnh
    confidence: float        # 0..1


@dataclass(frozen=True)
class Alert:
    """Cảnh báo cuối cùng hiển thị trên dashboard."""
    video_id: str
    start_s: float
    end_s: float
    event_type: EventType
    severity: str            # 'low' | 'medium' | 'high' | 'critical'
    responders: tuple[str, ...]
    protocol: str
    path: tuple[str, ...]    # đường đi trong policy graph, để giải thích
    caption: str = ""
    confidence: float = 0.0
