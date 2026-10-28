# -*- coding: utf-8 -*-
"""UCF-Crime: parse temporal annotation -> frame-level ground truth (track T3, Huy).

Định dạng `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` (mỗi dòng, cách nhau bởi khoảng trắng):

    Abuse028_x264.mp4  Abuse  165  240  -1  -1
    <tên video>        <lớp>  <start1> <end1> <start2> <end2>

- Số âm = không có đoạn đó (sentinel -1).
- Video nguồn 30 fps; frame đánh số từ 0 (quy ước phổ biến: đoạn [start, end] đóng; kiểm lại với LAVAD/RTFM
  trước khi chốt, và ghi quyết định vào notes/).
- Video test BÌNH THƯỜNG không có trong file -> nhãn toàn 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vad.types import EventType


@dataclass(frozen=True)
class Annotation:
    video_id: str                       # không đuôi .mp4
    label: str                          # tên lớp gốc, ví dụ 'Abuse'
    segments: tuple[tuple[int, int], ...]  # các đoạn (start, end) frame, đã bỏ sentinel

    @property
    def event_type(self) -> EventType:
        return EventType.from_label(self.label)


def parse_annotation_line(line: str) -> Annotation | None:
    """Một dòng -> Annotation. Dòng rỗng -> None. Dòng hỏng -> ValueError (không nuốt lỗi)."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) < 4:
        raise ValueError(f"Dòng annotation thiếu cột (cần >= 4): {line!r}")
    name, label = parts[0], parts[1]
    try:
        nums = [int(x) for x in parts[2:]]
    except ValueError as e:
        raise ValueError(f"Cột frame không phải số nguyên: {line!r}") from e
    segments: list[tuple[int, int]] = []
    for i in range(0, len(nums) - 1, 2):
        start, end = nums[i], nums[i + 1]
        if start < 0 or end < 0:      # sentinel -1: không có đoạn này
            continue
        if end < start:
            raise ValueError(f"Đoạn ngược (end < start) ở {name}: {start}..{end}")
        segments.append((start, end))
    video_id = name[:-4] if name.lower().endswith(".mp4") else name
    return Annotation(video_id=video_id, label=label, segments=tuple(segments))


def load_annotations(path: str | Path) -> dict[str, Annotation]:
    """Đọc cả file annotation -> {video_id: Annotation}. Chịu BOM (utf-8-sig)."""
    out: dict[str, Annotation] = {}
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                ann = parse_annotation_line(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            if ann is not None:
                out[ann.video_id] = ann
    return out


def frame_labels(n_frames: int, segments: tuple[tuple[int, int], ...]) -> np.ndarray:
    """bool[n_frames]: True = bất thường. Đoạn vượt n_frames bị cắt (không raise)."""
    if n_frames < 0:
        raise ValueError("n_frames phải >= 0")
    y = np.zeros(n_frames, dtype=bool)
    for start, end in segments:
        if start >= n_frames:
            continue
        y[start:min(end, n_frames - 1) + 1] = True
    return y


def load_gt(annotations: dict[str, Annotation], video_id: str, n_frames: int) -> np.ndarray:
    """GT frame-level cho một video; video không có trong annotation = bình thường (toàn False)."""
    ann = annotations.get(video_id)
    return frame_labels(n_frames, ann.segments if ann else ())
