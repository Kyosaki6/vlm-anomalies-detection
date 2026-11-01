# -*- coding: utf-8 -*-
"""Bộ nạp dữ liệu cho dashboard (Task E1): đọc `results/*.json`, KHÔNG import streamlit.

Tách khỏi `streamlit_app.py` để test được trên CPU thuần. Dữ liệu giả chỉ có khi gọi `demo_run()`
(dashboard chỉ gọi khi có cờ `--demo`) và luôn tự khai trong `metadata["note"]` + `metadata["demo"]`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from vad.results import DEFAULT_RESULTS_DIR, SCHEMA_VERSION, RunResult, result_path
from vad.types import Alert, VerifyResult, Window

DEMO_NOTE = "DEMO DATA: synthetic scores and a fake verifier. Not a real result."


def _is_run_file(path: Path) -> bool:
    """File là một `RunResult` về mặt khung: JSON object có `schema_version` == SCHEMA_VERSION.

    File khác loại nằm chung thư mục (`ablation_A*.json`, `threshold_A4c.json`, `typing_eval.json`,
    `a0_stride16*.json`) không có `schema_version` -> bị bỏ. JSON hỏng/không đọc được -> bỏ, không ném.
    """
    try:
        with path.open(encoding="utf-8-sig") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(obj, dict) and obj.get("schema_version") == SCHEMA_VERSION


def default_results_dir() -> Path:
    """`results/run_A4c/` nếu có, ngược lại `results/`. Biến `VAD_RESULTS_DIR` do caller xử lý trước."""
    preferred = Path(DEFAULT_RESULTS_DIR) / "run_A4c"
    return preferred if preferred.is_dir() else Path(DEFAULT_RESULTS_DIR)


def list_video_ids(root: str | Path = DEFAULT_RESULTS_DIR) -> list[str]:
    """`video_id` của mọi file kết quả chạy `<root>/<id>.json` (bỏ thư mục con `windows/` và file không phải run).

    Chỉ giữ file có `schema_version` khớp (xem `_is_run_file`). Sắp xếp theo tên.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.json") if p.is_file() and _is_run_file(p))


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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = ("manifests/ucf_test.csv", "manifests/own_demo.csv")
VIDEO_MISSING_TEXT = "video not on this machine"


def default_threshold(run: RunResult) -> tuple[float | None, str]:
    """(tau, nguồn) từ `metadata.threshold` / `metadata.threshold_source`. Không có -> (None, "")
    để UI tự lùi về trung điểm. KHÔNG tự bịa tau: giá trị không hữu hạn coi như thiếu."""
    raw = run.metadata.get("threshold")
    try:
        tau = float(raw)
    except (TypeError, ValueError):
        return None, ""
    if not math.isfinite(tau):
        return None, ""
    return tau, str(run.metadata.get("threshold_source", "") or "unknown")


def _read_run_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8-sig") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) and obj.get("schema_version") == SCHEMA_VERSION else None


def video_summaries(root: str | Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Một dòng mỗi file run: video_id, số cửa sổ, số alert, tier2. Dùng để lọc 'có >= 1 cửa sổ' (114/290).

    Đọc JSON thô (không dựng RunResult) nên rẻ hơn nhiều khi quét 290 file; UI nên cache kết quả.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.glob("*.json")):
        if not p.is_file():
            continue
        obj = _read_run_json(p)
        if obj is None:
            continue
        out.append({"video_id": p.stem,
                    "n_windows": len(obj.get("candidate_windows") or ()),
                    "n_alerts": len(obj.get("alerts") or ()),
                    "tier2": str((obj.get("metadata") or {}).get("tier2", ""))})
    return out


def filter_video_ids(summaries: Sequence[dict], with_windows_only: bool = True) -> list[str]:
    """Mặc định chỉ giữ video có >= 1 cửa sổ ứng viên; `with_windows_only=False` -> tất cả."""
    return [s["video_id"] for s in summaries if not with_windows_only or s["n_windows"] >= 1]


def find_video_path(video_id: str, manifests: Sequence[str | Path] = DEFAULT_MANIFESTS,
                    repo_root: str | Path = REPO_ROOT) -> Path | None:
    """Đường dẫn video gốc qua manifest, hoặc None nếu manifest không có / file không nằm trên máy này.

    Video UCF không nằm trong git -> thiếu file là bình thường, UI hiện `VIDEO_MISSING_TEXT`, không crash.
    """
    import csv

    root = Path(repo_root)
    for m in manifests:
        mp = Path(m) if Path(m).is_absolute() else root / m
        if not mp.is_file():
            continue
        try:
            with mp.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("video_id") == video_id and row.get("path"):
                        p = Path(row["path"])
                        p = p if p.is_absolute() else root / p
                        return p if p.is_file() else None
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
    return None


def verdict_label(v: VerifyResult | None) -> str:
    """'not verified' (chưa chạy tier-2) | 'confirmed' | 'rejected' theo `is_anomaly` (quy tắc A6, không đọc event_type)."""
    if v is None:
        return "not verified"
    return "confirmed" if v.is_anomaly else "rejected"


def window_bands(run: RunResult) -> list[dict]:
    """Dải cửa sổ cho biểu đồ: start/end/score/verdict (2 màu confirmed vs rejected + 'not verified')."""
    return [{"start (s)": w.start_s, "end (s)": w.end_s, "tier-1 score": w.score, "verdict": verdict_label(v)}
            for w, v in detections(run)]


def vlm_rejected_count(run: RunResult) -> int:
    """Số cửa sổ bị VLM bác (tier 2 đã chạy và `is_anomaly == False`). Chưa verify -> 0."""
    return sum(1 for _, v in detections(run) if v is not None and not v.is_anomaly)


def detection_rows(run: RunResult) -> list[dict]:
    """Bảng tier-1 vs tier-2. Cột 'VLM type (info)' là loại VLM nói, chỉ để tham khảo: loại của alert do typer quyết."""
    rows = []
    for w, v in detections(run):
        rows.append({"start (s)": w.start_s, "end (s)": w.end_s, "tier-1 score": round(w.score, 3),
                     "VLM verdict": verdict_label(v),
                     "VLM type (info)": "" if v is None else v.event_type.value,
                     "confidence": None if v is None else v.confidence,
                     "caption": "" if v is None else v.caption})
    return rows


def vlm_type_for_alert(run: RunResult, alert: Alert) -> str:
    """Loại VLM của cửa sổ sinh ra alert, tra theo (start_s, end_s) trong `metadata.event_typing.windows`,
    rồi lùi về cửa sổ có cùng khoảng. Không thấy -> ''."""
    typing = run.metadata.get("event_typing")
    if isinstance(typing, dict):
        for item in typing.get("windows") or ():
            try:
                if float(item["start_s"]) == alert.start_s and float(item["end_s"]) == alert.end_s:
                    return str(item.get("vlm_event_type", ""))
            except (KeyError, TypeError, ValueError):
                continue
    for w, v in detections(run):
        if v is not None and w.start_s == alert.start_s and w.end_s == alert.end_s:
            return v.event_type.value
    return ""


def alert_card(run: RunResult, alert: Alert) -> dict:
    """Mọi trường thẻ cảnh báo, thuần dữ liệu (UI chỉ vẽ). `type` = loại typer (Alert.event_type)."""
    return {
        "type": alert.event_type.value,
        "vlm_type": vlm_type_for_alert(run, alert),
        "severity": alert.severity,
        "start_s": alert.start_s,
        "end_s": alert.end_s,
        "camera": alert.camera,
        "responders": list(alert.responders),
        "protocol": alert.protocol,
        "caption": alert.caption,
        "confidence": alert.confidence,
        "path": list(alert.path),
    }


PRE_REGISTERED_CLAIM = ("Tier 2 is evaluated on FA/hour and window precision; "
                        "AUC can only move through the 307 windows.")
_ABLATION_ORDER = ("A0", "A1", "A2", "A3", "A3c", "A4", "A4c", "A5", "A5c", "A6")


def _ablation_sort_key(name: str) -> tuple[int, str]:
    return (_ABLATION_ORDER.index(name), name) if name in _ABLATION_ORDER else (len(_ABLATION_ORDER), name)


def load_ablations(root: str | Path = DEFAULT_RESULTS_DIR) -> tuple[list[dict], list[str]]:
    """(dòng bảng, tên file bị bỏ). Mỗi dòng: ablation, auc, ci_lo, ci_hi (đơn vị %) + ghi chú.

    File hỏng hoặc thiếu `auc_micro`/`ci_*` bị bỏ và LIỆT KÊ lại (UI hiện cảnh báo); không bao giờ điền 0.
    """
    base = Path(root)
    rows: list[dict] = []
    skipped: list[str] = []
    if not base.is_dir():
        return rows, skipped
    for p in sorted(base.glob("ablation_*.json")):
        try:
            with p.open(encoding="utf-8-sig") as f:
                obj = json.load(f)
            name = str(obj["ablation"])
            auc, lo, hi = (float(obj[k]) for k in ("auc_micro", "ci_lo", "ci_hi"))
        except (OSError, ValueError, KeyError, TypeError):
            skipped.append(p.name)
            continue
        if not all(math.isfinite(x) for x in (auc, lo, hi)):
            skipped.append(p.name)
            continue
        cfg = obj.get("config") if isinstance(obj.get("config"), dict) else {}
        rows.append({"ablation": name, "AUC (%)": round(100 * auc, 2),
                     "CI low (%)": round(100 * lo, 2), "CI high (%)": round(100 * hi, 2),
                     "prompt set": cfg.get("prompt_set", ""),
                     "memory": "" if cfg.get("use_memory") is None else ("yes" if cfg["use_memory"] else "no"),
                     "tier": "1+2" if name.startswith("A6") else "1"})
    rows.sort(key=lambda r: _ablation_sort_key(r["ablation"]))
    return rows, skipped


def load_a6(root: str | Path = DEFAULT_RESULTS_DIR) -> dict | None:
    """`ablation_A6.json` nguyên vẹn, hoặc None nếu chưa chạy / hỏng (UI hiện 'not run', không số 0)."""
    p = Path(root) / "ablation_A6.json"
    try:
        with p.open(encoding="utf-8-sig") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) and "operating_before" in obj and "operating_after" in obj else None


def operating_rows(a6: dict) -> list[dict]:
    """Bảng tier 1 -> tier 2: windows, precision, FA/hour (cả hai số), recall. Giá trị thiếu -> None."""
    b, a = a6["operating_before"], a6["operating_after"]

    def row(label: str, key: str, scale: float = 1.0, nd: int = 2) -> dict:
        def f(d: dict):
            x = d.get(key)
            return None if x is None else round(float(x) * scale, nd)
        return {"metric": label, "tier 1 (A4c)": f(b), "tier 1+2 (A6)": f(a)}

    return [
        row("windows", "windows", nd=0),
        row("true-positive windows", "true_positive_windows", nd=0),
        row("window precision (%)", "window_precision", 100),
        row("false-alarm windows", "false_alarm_windows", nd=0),
        row("FA/hour (all normal videos)", "fa_per_hour"),
        row("FA/hour (excl. videos 935/924)", "fa_per_hour_excl_long"),
        row("incident recall (%)", "incident_recall", 100),
        row("mean latency (s)", "mean_latency_s"),
    ]


def cost_rows(a6: dict) -> list[dict]:
    """Khối chi phí tier 2 từ `ablation_A6.json["cost"]`. Thiếu khoá -> bỏ dòng, không điền 0."""
    c = a6.get("cost") or {}
    spec = [("s/window median", "s_per_window_median", 2), ("s/window p90", "s_per_window_p90", 2),
            ("windows chunked (%)", "pct_windows_chunked", 1), ("peak VRAM (GB)", "peak_vram_gb", 2)]
    return [{"metric": label, "value": round(float(c[k]), nd)} for label, k, nd in spec if c.get(k) is not None]


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
    "DEMO_NOTE", "default_results_dir", "list_video_ids", "load_run", "is_demo", "is_placeholder", "run_note", "duration_s",
    "timeline", "detections", "alerts_by_severity", "threshold_crossings", "demo_run",
    "REPO_ROOT", "VIDEO_MISSING_TEXT", "PRE_REGISTERED_CLAIM", "default_threshold", "video_summaries",
    "filter_video_ids", "find_video_path", "verdict_label", "window_bands", "vlm_rejected_count", "detection_rows",
    "vlm_type_for_alert", "alert_card", "load_ablations", "load_a6", "operating_rows", "cost_rows",
]
