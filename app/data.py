# -*- coding: utf-8 -*-
"""Bộ nạp dữ liệu cho dashboard (Task E1): đọc `results/*.json`, KHÔNG import streamlit.

Tách khỏi `streamlit_app.py` để test được trên CPU thuần. Dữ liệu giả chỉ có khi gọi `demo_run()`
(dashboard chỉ gọi khi có cờ `--demo`) và luôn tự khai trong `metadata["note"]` + `metadata["demo"]`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from vad.results import DEFAULT_RESULTS_DIR, RunResult, result_path
from vad.types import Alert, VerifyResult, Window

DEMO_NOTE = "DEMO DATA: synthetic scores and a fake verifier. Not a real result."


def list_video_ids(root: str | Path = DEFAULT_RESULTS_DIR) -> list[str]:
    """`video_id` của mọi file `results/<id>.json` (bỏ thư mục con `windows/`). Sắp xếp theo tên."""
    base = Path(root)
    if not base.is_dir():
        return []
    non_video = ("ablation_", "threshold_", "a0_", "typing_eval")
    return sorted(p.stem for p in base.glob("*.json") if p.is_file() and not p.stem.startswith(non_video))


def load_run(video_id: str, root: str | Path = DEFAULT_RESULTS_DIR) -> RunResult:
    """Đọc một lần chạy. Lỗi (thiếu file, sai schema_version, video_id kiểu đường dẫn) được ném nguyên vẹn."""
    return RunResult.from_json(result_path(video_id, root))


def is_demo(run: RunResult) -> bool:
    return bool(run.metadata.get("demo"))


def is_placeholder(run: RunResult) -> bool:
    """Kết quả giả/mẫu: tự khai `demo` hoặc có `metadata.note` (như `results/sample_run.json`)."""
    return is_demo(run) or bool(str(run.metadata.get("note", "")).strip())


def run_note(run: RunResult) -> str:
    """Ghi chú của file kết quả, nguyên văn (UI phải hiển thị, không được giấu)."""
    return str(run.metadata.get("note", "")).strip()


def duration_s(run: RunResult) -> float:
    return len(run.anomaly_scores) / run.sample_fps


def timeline(run: RunResult) -> tuple[np.ndarray, np.ndarray]:
    """(t_giây, điểm) mỗi mẫu t phủ [t/fps, (t+1)/fps) -> vẽ tại t/fps."""
    scores = np.asarray(run.anomaly_scores, dtype=np.float64)
    return np.arange(scores.size) / run.sample_fps, scores


def detections(run: RunResult) -> list[tuple[Window, VerifyResult | None]]:
    """Ghép cửa sổ với kết quả tier-2 theo thứ tự. Chưa chạy tier-2 -> `None` (UI ghi 'not verified')."""
    if run.verify_results:
        return list(zip(run.candidate_windows, run.verify_results))
    return [(w, None) for w in run.candidate_windows]


def alerts_by_severity(alerts: Sequence[Alert], min_severity: str = "low") -> list[Alert]:
    """Lọc theo mức nghiêm trọng tối thiểu, mới nhất lên đầu (theo `start_s` giảm dần)."""
    order = ("low", "medium", "high", "critical")
    if min_severity not in order:
        raise ValueError(f"min_severity phải thuộc {order}, nhận {min_severity!r}")
    floor = order.index(min_severity)
    kept = [a for a in alerts if a.severity in order and order.index(a.severity) >= floor]
    return sorted(kept, key=lambda a: a.start_s, reverse=True)


def threshold_crossings(run: RunResult, threshold: float) -> int:
    """Số đoạn liên tiếp có điểm >= threshold (để hiện cạnh thanh trượt ngưỡng)."""
    if not math.isfinite(threshold):
        raise ValueError(f"threshold phải hữu hạn, nhận {threshold}")
    above = np.asarray(run.anomaly_scores, dtype=np.float64) >= threshold
    if above.size == 0:
        return 0
    return int(above[0]) + int(np.count_nonzero(above[1:] & ~above[:-1]))


def demo_run(video_id: str = "demo_video", seconds: int = 120, fps: float = 1.0, seed: int = 0) -> RunResult:
    """DỮ LIỆU GIẢ cho `--demo`. Có cửa sổ nhưng KHÔNG có verify/alert: không bịa kết quả tier-2."""
    from vad.temporal.ops import candidate_windows

    rng = np.random.default_rng(seed)
    scores = np.clip(rng.normal(0.1, 0.03, int(seconds * fps)), 0.0, 1.0)
    lo, hi = int(45 * fps), int(60 * fps)
    scores[lo:hi] = np.clip(scores[lo:hi] + 0.7, 0.0, 1.0)
    windows = tuple(candidate_windows(scores, fps, threshold=0.5, video_id=video_id))
    return RunResult(video_id=video_id, sample_fps=fps, anomaly_scores=scores, candidate_windows=windows,
                     metadata={"model": "demo", "demo": True, "note": DEMO_NOTE})


__all__ = [
    "DEMO_NOTE", "list_video_ids", "load_run", "is_demo", "is_placeholder", "run_note", "duration_s",
    "timeline", "detections", "alerts_by_severity", "threshold_crossings", "demo_run",
]
