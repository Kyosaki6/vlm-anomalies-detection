# -*- coding: utf-8 -*-
"""Hợp đồng file kết quả (Task C1): một lần chạy một video -> một file JSON.

Vì sao có file này: dashboard KHÔNG đọc code của pipeline, nó đọc kết quả đã lưu. Nếu mỗi track tự
bịa một cấu trúc JSON thì dashboard phải viết lại mỗi lần ai đó đổi tên trường. Ở đây chốt một lần:

    results/<video_id>.json          -- toàn bộ kết quả của video: điểm từng khung, cửa sổ ứng viên,
                                        kết quả VLM, cảnh báo, metadata
    results/windows/<video_id>.json  -- chỉ phần tier-2 (cửa sổ + kết quả VLM), tách riêng để chạy lại
                                        bước đắt tiền mà không phải chạy lại tier-1

Quy ước:
  - `schema_version` tăng khi đổi cấu trúc. File cũ đọc bằng code mới -> raise, không đoán.
  - Trường lạ trong JSON bị bỏ qua (để file mới đọc được bằng code cũ trong lúc chuyển tiếp).
  - Ghi nguyên tử (tmp + rename) để tắt máy giữa chừng không để lại file JSON hỏng.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vad.types import Alert, EventType, VerifyResult, Window

SCHEMA_VERSION = 1
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SCORE_DECIMALS = 6          # đủ chính xác, tránh JSON phình vì 17 chữ số thập phân


# --- đường dẫn -------------------------------------------------------------------

def result_path(video_id: str, root: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    return Path(root) / f"{video_id}.json"


def windows_path(video_id: str, root: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    return Path(root) / "windows" / f"{video_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Chưa có file kết quả: {p}")
    with open(p, encoding="utf-8-sig") as f:      # utf-8-sig: chịu được BOM trên Windows
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"File kết quả phải là JSON object, nhận {type(data).__name__}: {p}")
    return data


def make_metadata(model: str = "", commit: str = "", **extra: Any) -> dict[str, Any]:
    """Metadata chuẩn cho mọi lần chạy: model nào, commit nào, lúc nào."""
    meta: dict[str, Any] = {
        "model": model,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
    }
    meta.update(extra)
    return meta


# --- chuyển đổi dataclass <-> dict ------------------------------------------------

def _window_to_dict(w: Window) -> dict[str, Any]:
    return {"video_id": w.video_id, "start_s": w.start_s, "end_s": w.end_s, "score": w.score}


def _window_from_dict(d: dict[str, Any]) -> Window:
    return Window(video_id=str(d["video_id"]), start_s=float(d["start_s"]),
                  end_s=float(d["end_s"]), score=float(d["score"]))


def _verify_to_dict(v: VerifyResult) -> dict[str, Any]:
    return {"is_anomaly": bool(v.is_anomaly), "event_type": v.event_type.value,
            "caption": v.caption, "confidence": float(v.confidence)}


def _verify_from_dict(d: dict[str, Any]) -> VerifyResult:
    return VerifyResult(is_anomaly=bool(d["is_anomaly"]),
                        event_type=EventType.from_label(str(d["event_type"])),
                        caption=str(d["caption"]), confidence=float(d["confidence"]))


def _alert_to_dict(a: Alert) -> dict[str, Any]:
    return {
        "video_id": a.video_id, "start_s": a.start_s, "end_s": a.end_s,
        "event_type": a.event_type.value, "severity": a.severity,
        "responders": list(a.responders), "protocol": a.protocol, "path": list(a.path),
        "caption": a.caption, "confidence": a.confidence, "camera": a.camera,
    }


def _alert_from_dict(d: dict[str, Any]) -> Alert:
    return Alert(
        video_id=str(d["video_id"]), start_s=float(d["start_s"]), end_s=float(d["end_s"]),
        event_type=EventType.from_label(str(d["event_type"])), severity=str(d["severity"]),
        responders=tuple(d.get("responders", ())), protocol=str(d["protocol"]),
        path=tuple(d.get("path", ())), caption=str(d.get("caption", "")),
        confidence=float(d.get("confidence", 0.0)),
        camera=str(d.get("camera", "")),        # file cũ chưa có trường này -> ""
    )


# --- kết quả một lần chạy ---------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    """Toàn bộ kết quả của MỘT video. Đây là thứ dashboard đọc."""

    video_id: str
    sample_fps: float
    anomaly_scores: tuple[float, ...]
    candidate_windows: tuple[Window, ...] = ()
    verify_results: tuple[VerifyResult, ...] = ()
    alerts: tuple[Alert, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        # Cho phép truyền thẳng np.ndarray cho tiện (pipeline trả numpy).
        if isinstance(self.anomaly_scores, np.ndarray):
            object.__setattr__(self, "anomaly_scores",
                               tuple(float(x) for x in np.asarray(self.anomaly_scores).ravel()))
        if self.sample_fps <= 0:
            raise ValueError(f"sample_fps phải > 0, nhận {self.sample_fps}")
        if self.verify_results and len(self.verify_results) != len(self.candidate_windows):
            raise ValueError(
                "verify_results phải rỗng hoặc cùng độ dài với candidate_windows "
                f"({len(self.verify_results)} vs {len(self.candidate_windows)})"
            )

    # --- serialize ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "sample_fps": self.sample_fps,
            "n_frames": len(self.anomaly_scores),
            "anomaly_scores": [round(float(s), SCORE_DECIMALS) for s in self.anomaly_scores],
            "candidate_windows": [_window_to_dict(w) for w in self.candidate_windows],
            "verify_results": [_verify_to_dict(v) for v in self.verify_results],
            "alerts": [_alert_to_dict(a) for a in self.alerts],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version không hỗ trợ: {version!r} (code này đọc bản {SCHEMA_VERSION}). "
                "File này do phiên bản khác tạo — chạy lại hoặc nâng cấp code."
            )
        try:
            return cls(
                video_id=str(data["video_id"]),
                sample_fps=float(data["sample_fps"]),
                anomaly_scores=tuple(float(s) for s in data.get("anomaly_scores", ())),
                candidate_windows=tuple(_window_from_dict(w) for w in data.get("candidate_windows", ())),
                verify_results=tuple(_verify_from_dict(v) for v in data.get("verify_results", ())),
                alerts=tuple(_alert_from_dict(a) for a in data.get("alerts", ())),
                metadata=dict(data.get("metadata", {})),
                schema_version=version,
            )
        except KeyError as e:
            raise ValueError(f"File kết quả thiếu trường bắt buộc: {e}") from e

    # --- lưu / đọc ---

    def to_json(self, path: str | Path | None = None,
                root: str | Path = DEFAULT_RESULTS_DIR) -> Path:
        """Ghi ra `path`, hoặc `results/<video_id>.json` nếu không truyền. Trả về đường dẫn đã ghi."""
        target = Path(path) if path is not None else result_path(self.video_id, root)
        return _write_json_atomic(target, self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> "RunResult":
        return cls.from_dict(_read_json(Path(path)))

    def save_windows(self, root: str | Path = DEFAULT_RESULTS_DIR) -> Path:
        """Ghi riêng phần tier-2 ra `results/windows/<video_id>.json`."""
        payload = {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "sample_fps": self.sample_fps,
            "candidate_windows": [_window_to_dict(w) for w in self.candidate_windows],
            "verify_results": [_verify_to_dict(v) for v in self.verify_results],
            "metadata": dict(self.metadata),
        }
        return _write_json_atomic(windows_path(self.video_id, root), payload)

    @classmethod
    def load_windows(cls, video_id: str, root: str | Path = DEFAULT_RESULTS_DIR) -> "RunResult":
        """Đọc file windows (không có anomaly_scores) thành RunResult rút gọn."""
        data = _read_json(windows_path(video_id, root))
        return cls.from_dict({**data, "anomaly_scores": []})