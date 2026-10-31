# -*- coding: utf-8 -*-
"""Đánh giá A6 (tầng 2) theo ĐÚNG `notes/a6-protocol.md` (TH7b). Quy tắc nằm ở note đó; lệch note = lỗi của script.

    python scripts/eval_tier2.py --manifest manifests/ucf_test.csv --threshold-file results/threshold_A4c.json \\
        --windows-dir results/windows --verify-dir results/verify --out results/ablation_A6.json

Ý chính:
  * verdict của window = `is_anomaly` trong `results/verify/<video_id>/window_NNNN.json` (đã OR theo chunk, D-1);
  * window bị VLM bác: điểm các mẫu trong window = 0.0 (median calib_normal sau chuẩn hoá, < tau); phần còn lại giữ nguyên;
  * AUC micro + CI bootstrap theo VIDEO (cùng seed/giao thức `ablation_A4c.json`), thêm delta ghép cặp so với A4c;
  * số liệu vận hành trước/sau ở CÙNG tau: FA/hour, precision window, recall sự cố, độ trễ, window bị bác.

Thiếu file verify, window lệch, `window_index` thừa/trùng, verifier giả (`metadata.fake`) = LỖI, không có verdict mặc định.
Kết quả giả (`--allow-fake`) tự khai trong JSON (`ablation = "A6-fake"`) và không bao giờ được ghi vào `results/ablation_A6.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import decision  # noqa: E402
import export_windows as ew  # noqa: E402
import run_benchmark as rb  # noqa: E402
from vad.datasets.ucf_crime import Annotation, load_annotations, load_gt  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR  # noqa: E402
from vad.eval import metrics as mt  # noqa: E402
from vad.eval.metrics import (detection_latency_s, false_alarms_per_hour, fast_frame_level_auc,  # noqa: E402
                              fast_video_bootstrap_auc_ci)
from vad.pipeline import ABLATIONS, DEFAULT_PROMPTS, PipelineConfig, score_cached_video, with_threshold  # noqa: E402
from vad.results import DEFAULT_RESULTS_DIR, RunResult  # noqa: E402
from vad.temporal.ops import candidate_windows, expand_to_frames  # noqa: E402
from vad.types import Window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_NOTE = ROOT / "notes" / "a6-protocol.md"
DEFAULT_WINDOWS_DIR = DEFAULT_RESULTS_DIR / "windows"
DEFAULT_VERIFY_DIR = DEFAULT_RESULTS_DIR / "verify"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "ablation_A6.json"
DEMOTED_SCORE = 0.0       # median calib_normal sau chuẩn hoá (a6-protocol §2)
LONG_NORMAL_VIDEOS = ("Normal_Videos_935_x264", "Normal_Videos_924_x264")  # a6-protocol §3: số FA/hour thứ hai
FP_TOL = 1e-6

class Tier2EvalError(ValueError):
    """Đầu vào A6 sai: thiếu/thừa verdict, window lệch, verifier giả, tau không hợp lệ..."""

@dataclass(frozen=True)
class VideoInput:
    """Một video test: điểm tier-1 cấp MẪU (sample_fps), GT, và window đóng băng của video đó."""
    video_id: str
    fps: float
    n_frames: int
    scores: np.ndarray                         # (T,) điểm A4c cuối, trước khi expand ra frame
    segments: tuple[tuple[int, int], ...]      # GT (start, end) frame, bao gồm hai đầu
    windows: tuple[Window, ...]

    @property
    def is_normal(self) -> bool:
        return not self.segments

@dataclass(frozen=True)
class WindowVerdict:
    is_anomaly: bool
    fake: bool = False
    model: str = ""
    elapsed_s: float | None = None
    n_chunks: int = 1
    chunk_seconds: float = 0.0
    prompt_hash: str | None = None

# --- demote (a6-protocol §2) ------------------------------------------------------------

def window_sample_span(w: Window, sample_fps: float, n_samples: int) -> tuple[int, int]:
    """[lo, hi) mẫu mà window phủ: floor(start*fps) <= t < ceil(end*fps), cắt vào biên (cùng span `candidate_windows`)."""
    lo = int(math.floor(w.start_s * sample_fps + 1e-9))
    hi = int(math.ceil(w.end_s * sample_fps - 1e-9))
    return max(0, lo), min(n_samples, hi)

def demote(scores: np.ndarray, sample_fps: float, rejected: Sequence[Window]) -> np.ndarray:
    """Bản sao `scores`; mẫu trong window bị bác = 0.0. Không đụng gì khác."""
    out = np.array(scores, dtype=np.float64, copy=True)
    for w in rejected:
        lo, hi = window_sample_span(w, sample_fps, out.size)
        out[lo:hi] = DEMOTED_SCORE
    return out

# --- verdict ------------------------------------------------------------------------------

def verify_file(verify_dir: str | Path, video_id: str, window_index: int) -> Path:
    """Cùng đường dẫn `run_verify.output_path` (có test khoá hai bên khớp nhau)."""
    if (not video_id or video_id in {".", ".."} or "/" in video_id or "\\" in video_id or "\x00" in video_id):
        raise Tier2EvalError(f"video_id không hợp lệ làm tên thư mục: {video_id!r}")
    return Path(verify_dir) / video_id / f"window_{window_index:04d}.json"

def _same_window(a: Window, b: Window) -> bool:
    return (a.video_id == b.video_id and abs(a.start_s - b.start_s) <= 1e-9 and abs(a.end_s - b.end_s) <= 1e-9
            and abs(a.score - b.score) <= FP_TOL)

def load_verdicts(verify_dir: str | Path, videos: Sequence[VideoInput],
                  allow_fake: bool = False) -> dict[tuple[str, int], WindowVerdict]:
    """Verdict cho MỌI window đóng băng. Thiếu file / lệch window / file thừa / verifier giả -> `Tier2EvalError`."""
    root = Path(verify_dir)
    errors: list[str] = []
    out: dict[tuple[str, int], WindowVerdict] = {}
    for v in videos:
        for index, w in enumerate(v.windows):
            path = verify_file(root, v.video_id, index)
            if not path.is_file():
                errors.append(f"{v.video_id}[{index}]: thiếu {path}")
                continue
            try:
                run = RunResult.from_json(path)
            except (TypeError, ValueError, OSError) as e:
                errors.append(f"{v.video_id}[{index}]: không đọc được {path} ({e})")
                continue
            meta = run.metadata
            if run.video_id != v.video_id or meta.get("window_index") != index:
                errors.append(f"{v.video_id}[{index}]: file nói video {run.video_id!r}, window_index "
                              f"{meta.get('window_index')!r}")
                continue
            if len(run.candidate_windows) != 1 or len(run.verify_results) != 1:
                errors.append(f"{v.video_id}[{index}]: cần đúng 1 window + 1 verify_result trong {path}")
                continue
            if not _same_window(run.candidate_windows[0], w):
                errors.append(f"{v.video_id}[{index}]: window trong verify {run.candidate_windows[0]} khác window "
                              f"đóng băng {w} (tầng 1 đã đổi sau khi verify?)")
                continue
            fake = bool(meta.get("fake", False))
            if fake and not allow_fake:
                errors.append(f"{v.video_id}[{index}]: verifier GIẢ, không được dùng cho A6 thật")
                continue
            chunks = meta.get("chunks") or []
            chunk_seconds = sum(float(c["end_s"]) - float(c["start_s"]) for c in chunks) if chunks else w.duration_s
            cfg = meta.get("verifier_config") or {}
            elapsed = meta.get("elapsed_s")
            out[(v.video_id, index)] = WindowVerdict(
                is_anomaly=bool(run.verify_results[0].is_anomaly), fake=fake, model=str(meta.get("model", "")),
                elapsed_s=float(elapsed) if isinstance(elapsed, (int, float)) else None,
                n_chunks=int(meta.get("n_chunks") or max(1, len(chunks))), chunk_seconds=float(chunk_seconds),
                prompt_hash=cfg.get("prompt_hash"))
        on_disk = sorted(p.name for p in (root / v.video_id).glob("window_*.json")) if (root / v.video_id).is_dir() else []
        expected = {verify_file(root, v.video_id, i).name for i in range(len(v.windows))}
        extra = [n for n in on_disk if n not in expected]
        if extra:
            errors.append(f"{v.video_id}: file verify thừa (window_index ngoài 0..{len(v.windows) - 1}): {extra}")
    known = {v.video_id for v in videos if v.windows}
    if root.is_dir():
        stray = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in known)
        if stray:
            errors.append(f"thư mục verify của video ngoài manifest/không có window: {stray}")
    if errors:
        head = "\n  ".join(errors[:20])
        raise Tier2EvalError(f"{len(errors)} lỗi verdict (chỉ hiện 20 đầu):\n  {head}")
    models = {x.model for x in out.values()}
    if len(models) > 1:
        raise Tier2EvalError(f"verify_dir trộn nhiều verifier: {sorted(models)}")
    return out

# --- metric vận hành (a6-protocol §3) -------------------------------------------------------------

def _overlaps(w: Window, g0: float, g1: float) -> bool:
    return w.start_s < g1 and w.end_s > g0           # cùng quy tắc `calibrate_threshold.operating_point`

def operating_metrics(videos: Sequence[VideoInput], kept: Mapping[str, Sequence[Window]],
                      incident_filter: set[tuple[str, int]] | None = None) -> dict[str, Any]:
    """Số liệu vận hành cho tập window `kept[video_id]`. Trả luôn tử/mẫu số để báo cáo."""
    normal_s = normal_s_excl = 0.0
    fa = fa_excl = 0
    tp_win = n_win = 0
    incidents = detected = 0
    detected_ids: set[tuple[str, int]] = set()
    lat_by_incident: dict[tuple[str, int], float] = {}
    for v in videos:
        wins = list(kept.get(v.video_id, ()))
        n_win += len(wins)
        if v.is_normal:
            dur = v.n_frames / v.fps
            normal_s += dur
            fa += len(wins)
            if v.video_id not in LONG_NORMAL_VIDEOS:
                normal_s_excl += dur
                fa_excl += len(wins)
            continue
        spans = [(a / v.fps, (b + 1) / v.fps) for a, b in v.segments]
        for w in wins:
            if any(_overlaps(w, g0, g1) for g0, g1 in spans):
                tp_win += 1
        for i, (g0, g1) in enumerate(spans):
            key = (v.video_id, i)
            incidents += 1
            hit = [w for w in wins if _overlaps(w, g0, g1)]
            if hit:
                detected += 1
                detected_ids.add(key)
                lat_by_incident[key] = float(detection_latency_s(g0, min(w.start_s for w in hit)))
    if normal_s <= 0:
        raise Tier2EvalError("không có video bình thường để đo FA/hour")
    lats = [lat_by_incident[k] for k in detected_ids if incident_filter is None or k in incident_filter]
    return {
        "windows": n_win, "true_positive_windows": tp_win,
        "window_precision": (tp_win / n_win) if n_win else None,
        "false_alarm_windows": fa, "normal_hours": normal_s / 3600.0,
        "fa_per_hour": false_alarms_per_hour(fa, normal_s),
        "false_alarm_windows_excl_long": fa_excl, "normal_hours_excl_long": normal_s_excl / 3600.0,
        "fa_per_hour_excl_long": false_alarms_per_hour(fa_excl, normal_s_excl) if normal_s_excl > 0 else None,
        "excluded_long_videos": list(LONG_NORMAL_VIDEOS),
        "incidents": incidents, "incidents_detected": detected,
        "incident_recall": (detected / incidents) if incidents else None,
        "latency_n": len(lats),
        "mean_latency_s": float(np.mean(lats)) if lats else None,
        "median_latency_s": float(np.median(lats)) if lats else None,
        "_detected_ids": detected_ids,
    }

def _public(m: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if not k.startswith("_")}

# --- AUC ghép cặp (a6-protocol §2, thứ cấp) -------------------------------------------------------

def paired_auc_delta_ci(scores_after: Sequence[np.ndarray], scores_before: Sequence[np.ndarray],
                        labels: Sequence[np.ndarray], n_bootstraps: int = 1000, ci: float = 0.95,
                        seed: int = 42) -> tuple[float, float]:
    """CI bootstrap theo VIDEO của AUC(after) - AUC(before), cùng các lần rút video cho cả hai.

    Rút video bằng đúng dãy `random.Random(seed).randrange(n)` của `fast_video_bootstrap_auc_ci`, nên mỗi lần
    rút ở đây trùng lần rút của CI không ghép cặp.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError("ci phải trong (0, 1)")
    n = len(labels)
    s_a, y, vid = mt._concat_checked(scores_after, labels)
    s_b, _, _ = mt._concat_checked(scores_before, labels)
    _, inv_a = np.unique(s_a, return_inverse=True)
    _, inv_b = np.unique(s_b, return_inverse=True)
    ga, gb = int(inv_a.max()) + 1, int(inv_b.max()) + 1
    pos, neg = y.astype(np.float64), (~y).astype(np.float64)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_bootstraps):
        counts = np.bincount([rng.randrange(n) for _ in range(n)], minlength=n).astype(np.float64)
        w = counts[vid]
        a = mt._auc_from_groups(np.bincount(inv_a, weights=w * pos, minlength=ga),
                                np.bincount(inv_a, weights=w * neg, minlength=ga))
        b = mt._auc_from_groups(np.bincount(inv_b, weights=w * pos, minlength=gb),
                                np.bincount(inv_b, weights=w * neg, minlength=gb))
        if not (math.isnan(a) or math.isnan(b)):
            deltas.append(a - b)
    if not deltas:
        return float("nan"), float("nan")
    deltas.sort()
    tail = (1.0 - ci) / 2.0
    return decision._percentile(deltas, tail), decision._percentile(deltas, 1.0 - tail)

# --- đánh giá ---------------------------------------------------------------------------------------

def _nn(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

def check_frozen_windows(videos: Sequence[VideoInput], config: PipelineConfig, tau: float) -> None:
    """Window đóng băng phải đúng bằng window dựng lại từ điểm A4c ở tau (chống lệch cache/config)."""
    bad: list[str] = []
    for v in videos:
        again = candidate_windows(v.scores, config.sample_fps, tau, video_id=v.video_id,
                                  min_len_s=config.min_len_s, merge_gap_s=config.merge_gap_s)
        if len(again) != len(v.windows) or not all(_same_window(a, b) for a, b in zip(again, v.windows)):
            bad.append(v.video_id)
    if bad:
        raise Tier2EvalError(f"window đóng băng khác window dựng lại từ điểm A4c ở tau={tau} ở {len(bad)} video: {bad[:10]}")

def evaluate_a6(videos: Sequence[VideoInput], verdicts: Mapping[tuple[str, int], WindowVerdict],
                config: PipelineConfig, tau: float, labels: Sequence[np.ndarray] | None = None,
                n_bootstraps: int = 1000, seed: int = 42, paired: bool = True) -> dict[str, Any]:
    """Tính toàn bộ A6 từ dữ liệu trong bộ nhớ. `labels` None -> dựng từ GT của `videos`."""
    if not (math.isfinite(tau) and tau > DEMOTED_SCORE):
        raise Tier2EvalError(f"tau phải hữu hạn và > {DEMOTED_SCORE} (điểm bị hạ phải nằm dưới ngưỡng), nhận {tau!r}")
    fps_s = config.sample_fps
    missing = [(v.video_id, i) for v in videos for i in range(len(v.windows)) if (v.video_id, i) not in verdicts]
    if missing:
        raise Tier2EvalError(f"thiếu verdict cho {len(missing)} window: {missing[:10]}")
    n_expected = sum(len(v.windows) for v in videos)
    extra = set(verdicts) - {(v.video_id, i) for v in videos for i in range(len(v.windows))}
    if extra:
        raise Tier2EvalError(f"verdict của window không tồn tại: {sorted(extra)[:10]}")

    if labels is None:
        labels = [load_gt({v.video_id: Annotation(v.video_id, "", v.segments)}, v.video_id, v.n_frames) for v in videos]
    before_f: list[np.ndarray] = []
    after_f: list[np.ndarray] = []
    kept: dict[str, list[Window]] = {}
    rejected_all: dict[str, list[Window]] = {}
    lost: list[dict[str, Any]] = []
    split = {"rejected_in_anomalous_videos": 0, "rejected_in_normal_videos": 0,
             "rejected_overlapping_gt": 0, "rejected_not_overlapping_gt": 0}
    for v in videos:
        rej = [w for i, w in enumerate(v.windows) if not verdicts[(v.video_id, i)].is_anomaly]
        keep = [w for i, w in enumerate(v.windows) if verdicts[(v.video_id, i)].is_anomaly]
        kept[v.video_id] = keep
        rejected_all[v.video_id] = rej
        after = demote(v.scores, fps_s, rej)
        # Sau khi hạ điểm, window dựng lại ở cùng tau phải đúng bằng tập window được xác nhận.
        again = candidate_windows(after, fps_s, tau, video_id=v.video_id, min_len_s=config.min_len_s,
                                  merge_gap_s=config.merge_gap_s)
        if len(again) != len(keep) or not all(_same_window(a, b) for a, b in zip(again, keep)):
            raise Tier2EvalError(f"{v.video_id}: window còn lại sau khi hạ điểm không khớp window được xác nhận")
        before_f.append(expand_to_frames(v.scores, fps_s, v.fps, v.n_frames))
        after_f.append(expand_to_frames(after, fps_s, v.fps, v.n_frames))
        spans = [(a / v.fps, (b + 1) / v.fps) for a, b in v.segments]
        for w in rej:
            hit = any(_overlaps(w, g0, g1) for g0, g1 in spans)
            split["rejected_in_normal_videos" if v.is_normal else "rejected_in_anomalous_videos"] += 1
            split["rejected_overlapping_gt" if hit else "rejected_not_overlapping_gt"] += 1
            if hit:
                lost.append({"video_id": v.video_id, "start_s": w.start_s, "end_s": w.end_s})

    all_windows = {v.video_id: list(v.windows) for v in videos}
    before = operating_metrics(videos, all_windows)
    after_m = operating_metrics(videos, kept)
    common = before["_detected_ids"] & after_m["_detected_ids"]
    before_common = operating_metrics(videos, all_windows, incident_filter=common)
    after_common = operating_metrics(videos, kept, incident_filter=common)
    if after_m["incidents_detected"] > before["incidents_detected"]:
        raise Tier2EvalError("recall sau tầng 2 lớn hơn trước: tầng 2 chỉ được bỏ window (lỗi trong script)")

    auc_before = fast_frame_level_auc(before_f, labels)
    auc_after = fast_frame_level_auc(after_f, labels)
    lo, hi = fast_video_bootstrap_auc_ci(after_f, labels, n_bootstraps=n_bootstraps, seed=seed)
    out: dict[str, Any] = {
        "n_windows": n_expected,
        "n_confirmed": sum(len(k) for k in kept.values()),
        "n_rejected": sum(len(r) for r in rejected_all.values()),
        "auc_micro": _nn(auc_after), "ci_lo": _nn(lo), "ci_hi": _nn(hi),
        "auc_micro_a4c": _nn(auc_before),
        "delta_auc": _nn(auc_after - auc_before),
        "operating_before": _public(before), "operating_after": _public(after_m),
        "latency_common_incidents": {"n": len(common), "before": {"mean_latency_s": before_common["mean_latency_s"],
                                                                  "median_latency_s": before_common["median_latency_s"]},
                                     "after": {"mean_latency_s": after_common["mean_latency_s"],
                                               "median_latency_s": after_common["median_latency_s"]}},
        "rejected_split": split, "lost_true_positive_windows": lost,
        "n_videos": len(videos), "n_frames": int(sum(len(s) for s in after_f)),
    }
    if paired:
        dlo, dhi = paired_auc_delta_ci(after_f, before_f, labels, n_bootstraps=n_bootstraps, seed=seed)
        out["delta_auc_ci_lo"], out["delta_auc_ci_hi"] = _nn(dlo), _nn(dhi)
    return out

def cost_summary(verdicts: Mapping[tuple[str, int], WindowVerdict], n_videos: int,
                 peak_vram_gb: float | None = None) -> dict[str, Any]:
    """Chi phí từ metadata các file verify (không cần log): s/window, % chunk, chunk-call, giây video đưa cho VLM."""
    vs = list(verdicts.values())
    el = sorted(x.elapsed_s for x in vs if x.elapsed_s is not None)
    p90 = el[min(len(el) - 1, math.ceil(0.9 * len(el)) - 1)] if el else None
    return {
        "windows_per_video": len(vs) / n_videos if n_videos else None,
        "chunk_calls": sum(x.n_chunks for x in vs),
        "windows_chunked": sum(1 for x in vs if x.n_chunks > 1),
        "pct_windows_chunked": 100.0 * sum(1 for x in vs if x.n_chunks > 1) / len(vs) if vs else None,
        "video_seconds_shown": sum(x.chunk_seconds for x in vs),
        "s_per_window_median": statistics.median(el) if el else None,
        "s_per_window_p90": p90, "s_per_window_n": len(el),
        "s_per_window_total": sum(el) if el else None,
        "peak_vram_gb": peak_vram_gb,
    }

# --- I/O ----------------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""

def protocol_fingerprint(path: Path = PROTOCOL_NOTE) -> dict[str, str]:
    if not path.is_file():
        raise Tier2EvalError(f"thiếu {path}: A6 không được chạy khi chưa có giao thức viết trước")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    return {"path": rel, "sha256": sha, "git_commit": _git("log", "-1", "--format=%h", "--", rel) or "uncommitted"}

def load_videos(rows: Sequence[Mapping[str, Any]], annotations: Mapping[str, Annotation], config: PipelineConfig,
                tau: float, stats: Any, prompts: tuple[np.ndarray, np.ndarray], cache_dir: str | Path,
                windows_dir: str | Path) -> list[VideoInput]:
    """Dựng điểm A4c cấp mẫu từ cache, đọc window đóng băng, kiểm index. Mọi video trong manifest."""
    wdir = Path(windows_dir)
    index_path = wdir / ew.INDEX_NAME
    cfg = with_threshold(config, tau)
    videos: list[VideoInput] = []
    for r in rows:
        vid = r["video_id"]
        res = score_cached_video(vid, cfg, prompts[0], prompts[1], cache_dir, calib_stats=stats)
        want = rb.expected_samples(r["n_frames"], r["fps"], cfg.sample_fps)
        if abs(len(res.scores) - want) > rb.LENGTH_TOLERANCE or not len(res.scores) or not np.isfinite(res.scores).all():
            raise rb.BenchmarkInputError(f"{vid}: điểm xấu hoặc lệch độ dài (cache {len(res.scores)}, cần ~{want})")
        wfile = wdir / f"{vid}.json"
        wins: tuple[Window, ...] = ()
        if wfile.is_file():
            run = RunResult.from_json(wfile)
            if run.video_id != vid:
                raise Tier2EvalError(f"{wfile} có video_id {run.video_id!r}, manifest nói {vid!r}")
            wins = tuple(run.candidate_windows)
        ann = annotations.get(vid)
        videos.append(VideoInput(vid, float(r["fps"]), int(r["n_frames"]), np.asarray(res.scores, dtype=np.float64),
                                 tuple(ann.segments) if ann else (), wins))
    if index_path.is_file():
        with open(index_path, encoding="utf-8-sig") as f:
            idx = json.load(f)
        total = sum(len(v.windows) for v in videos)
        if idx.get("n_windows") != total:
            raise Tier2EvalError(f"{index_path} nói {idx.get('n_windows')} window, đọc được {total} từ {wdir}")
        if abs(float(idx.get("threshold", tau)) - tau) > 1e-9:
            raise Tier2EvalError(f"tau trong {index_path} ({idx.get('threshold')}) khác tau của file ngưỡng ({tau})")
    return videos

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--threshold-file", required=True)
    p.add_argument("--annotations", default=str(rb.DEFAULT_ANNOTATIONS))
    p.add_argument("--ablation", default="A4c")
    p.add_argument("--windows-dir", default=str(DEFAULT_WINDOWS_DIR))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--prompt-cache-dir", default=str(rb.DEFAULT_PROMPT_CACHE))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-paired", action="store_true", help="bỏ CI delta ghép cặp (nhanh hơn)")
    p.add_argument("--peak-vram-gb", type=float, default=None, help="đỉnh VRAM từ log A5b, chỉ để ghi lại")
    p.add_argument("--allow-fake", action="store_true",
                   help="chấp nhận output FakeVerifier (chỉ dev/test); kết quả tự khai là 'A6-fake'")
    args = p.parse_args(argv)

    if args.ablation not in ABLATIONS:
        p.error(f"ablation không có: {args.ablation}; hợp lệ: {sorted(ABLATIONS)}")
    out_path = Path(args.out)
    if args.allow_fake and out_path.resolve() == DEFAULT_OUT.resolve():
        p.error("--allow-fake không được ghi vào results/ablation_A6.json; dùng --out khác")
    config = ABLATIONS[args.ablation]
    tau, stats = ew.load_threshold_file(args.threshold_file, args.ablation, config)
    proto = protocol_fingerprint()
    rows = rb.read_benchmark_manifest(args.manifest)
    annotations = load_annotations(args.annotations)
    rb.check_inputs(rows, annotations, config, args.cache_dir)
    prompts = rb.load_or_embed_prompts(config, args.prompts, args.prompt_cache_dir, device=args.device)
    videos = load_videos(rows, annotations, config, tau, stats, prompts, args.cache_dir, args.windows_dir)
    check_frozen_windows(videos, config, tau)
    verdicts = load_verdicts(args.verify_dir, videos, allow_fake=args.allow_fake)
    fake = any(x.fake for x in verdicts.values())
    labels = [load_gt(annotations, v.video_id, v.n_frames) for v in videos]
    res = evaluate_a6(videos, verdicts, config, tau, labels=labels, n_bootstraps=args.n_bootstraps, seed=args.seed,
                      paired=not args.no_paired)
    hashes = sorted({x.prompt_hash for x in verdicts.values() if x.prompt_hash})
    payload = {
        "ablation": "A6-fake" if fake else "A6", "source_ablation": args.ablation, "threshold": tau,
        "threshold_source": "calib_normal", "manifest": args.manifest, "seed": args.seed,
        "n_bootstraps": args.n_bootstraps, "verifier": next(iter({x.model for x in verdicts.values()}), ""),
        "prompt_hash": hashes[0] if len(hashes) == 1 else hashes, "verify_dir": str(args.verify_dir),
        "fake_verifier": fake, "protocol_note": proto,
        **res, "cost": cost_summary(verdicts, len(videos), args.peak_vram_gb),
        "note": ("KẾT QUẢ GIẢ: verdict từ FakeVerifier, không phải số liệu A6." if fake else ""),
        "commit": rb.git_commit(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rb.write_json_atomic(out_path, payload)
    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
    b, a = payload["operating_before"], payload["operating_after"]
    print(f"{payload['ablation']} tau={tau:.4f}: {res['n_confirmed']}/{res['n_windows']} window được xác nhận, "
          f"{res['n_rejected']} bị bác")
    print(f"  AUC {f(res['auc_micro'] * 100)} [{f(res['ci_lo'] * 100)}, {f(res['ci_hi'] * 100)}]  "
          f"(A4c {f(res['auc_micro_a4c'] * 100)}, delta {f(res['delta_auc'] * 100)})")
    print(f"  FA/hour {f(b['fa_per_hour'])} -> {f(a['fa_per_hour'])}  precision {f(b['window_precision'], 3)} -> "
          f"{f(a['window_precision'], 3)}  recall {b['incidents_detected']}/{b['incidents']} -> "
          f"{a['incidents_detected']}/{a['incidents']} -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
