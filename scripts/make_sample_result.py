# -*- coding: utf-8 -*-
"""Sinh `results/sample_run.json` từ dữ liệu GIẢ — để dashboard có file thật để đọc khi chưa có GPU.

    python scripts/make_sample_result.py

Không cần GPU, không tải model. File sinh ra là ví dụ về ĐÚNG cấu trúc, KHÔNG phải kết quả thật:
`metadata.note` ghi rõ điều đó, và dashboard phải hiển thị nguyên văn ghi chú này.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.fakes import FakeVerifier  # noqa: E402
from vad.graph.alerts import build_alert_feed  # noqa: E402
from vad.graph.policy import GraphPolicy  # noqa: E402
from vad.results import RunResult, make_metadata  # noqa: E402
from vad.temporal.ops import candidate_windows  # noqa: E402
from vad.types import EventType, VerifyResult, Window  # noqa: E402

VIDEO_ID = "sample_run"
SAMPLE_FPS = 1.0


def main() -> int:
    rng = np.random.default_rng(0)
    # 120 giây: nền thấp, có 3 đoạn nhô lên (giây 20-30, 55-65, 95-105)
    scores = rng.normal(0.10, 0.02, 120).clip(0, 1)
    for start, end, peak in ((20, 30, 0.82), (55, 65, 0.71), (95, 105, 0.90)):
        scores[start:end] = rng.normal(peak, 0.03, end - start).clip(0, 1)

    windows = candidate_windows(scores, SAMPLE_FPS, threshold=0.5, video_id=VIDEO_ID)
    fake = FakeVerifier(event_type=EventType.FIGHTING, confidence=0.88)
    detections = [(w, fake.verify([], {"video_id": VIDEO_ID})) for w in windows]

    # Cửa sổ thứ hai là tai nạn, để bản mẫu có hai loại sự kiện khác nhau
    if len(detections) > 1:
        w = detections[1][0]
        detections[1] = (w, VerifyResult(True, EventType.ROAD_ACCIDENT,
                                         "a car hits a pedestrian at the crossing", 0.81))

    policy = GraphPolicy()
    alerts = build_alert_feed(detections, policy, camera="gate_a", zone="school_gate", hour=12)

    result = RunResult(
        video_id=VIDEO_ID,
        sample_fps=SAMPLE_FPS,
        anomaly_scores=tuple(float(s) for s in scores),
        candidate_windows=tuple(windows),
        verify_results=tuple(r for _, r in detections),
        alerts=tuple(alerts),
        metadata=make_metadata(
            model="google/siglip2-base-patch16-384",
            commit="sample",
            note="DỮ LIỆU GIẢ (FakeVerifier + điểm ngẫu nhiên có chủ đích). Không phải kết quả thật.",
        ),
    )
    path = result.to_json()
    result.save_windows()
    print(f"đã ghi {path}")
    print(f"  {len(scores)} khung @ {SAMPLE_FPS} fps, {len(windows)} cửa sổ ứng viên, {len(alerts)} cảnh báo")
    for a in alerts:
        print(f"  - {a.event_type.value:<14} {a.severity:<8} {a.start_s:>5.0f}-{a.end_s:<5.0f}s "
              f"-> {', '.join(a.responders) or '-'} [{a.protocol}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

