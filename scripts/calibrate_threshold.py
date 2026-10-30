# -*- coding: utf-8 -*-
"""Hiệu chỉnh ngưỡng báo động trên `calib_normal` (TH5) rồi kiểm tra ngưỡng đó trên test (chỉ ĐỌC, không chỉnh).

    python scripts/calibrate_threshold.py --manifest manifests/ucf_test.csv \\
        --calib-manifest manifests/ucf_calib_normal.csv --ablation A4c --target-fpr 0.05

Ngưỡng = phân vị (1 - FPR) của điểm tier-1 gộp trên MỌI mẫu của các video `calib_normal`. Tức là tối đa `FPR`
mẫu bình thường vượt ngưỡng. KHÔNG dùng test để chọn ngưỡng; test chỉ để đo FPR/FA-per-hour/recall tại ngưỡng đã chốt.

Từ chối (lỗi rõ ràng, không lặng lẽ chạy):
  - config chuẩn hoá THEO VIDEO (A3/A4/A5): median mọi video = 0, ngưỡng cố định không có nghĩa. Dùng A3c/A4c.
  - config dùng memory (A5c): điểm calib phải chấm leave-one-video-out, chưa hỗ trợ ở script này.
Ghi results/threshold_<AID>.json (nguyên tử).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_benchmark as rb  # noqa: E402
from vad.datasets.ucf_crime import Annotation, load_annotations  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR  # noqa: E402
from vad.eval.metrics import detection_latency_s, false_alarms_per_hour  # noqa: E402
from vad.pipeline import (ABLATIONS, DEFAULT_PROMPTS, CalibStats, PipelineConfig, score_cached_video,  # noqa: E402
                          with_threshold)
from vad.temporal.ops import candidate_windows  # noqa: E402

DEFAULT_TARGET_FPR = 0.05
BANK_MAX_FRAMES = 20000


class CalibrationError(ValueError):
    """Không thể hiệu chỉnh một cách có nghĩa (config sai loại, dữ liệu rỗng/NaN...)."""


def check_calibratable(config: PipelineConfig) -> None:
    if config.per_video_norm and config.norm_ref == "video":
        raise CalibrationError("config chuẩn hoá theo video: median mỗi video = 0 nên ngưỡng cố định không có nghĩa "
                               "giữa các video. Dùng biến thể 'c' (A3c/A4c) hoặc config không chuẩn hoá.")
    if config.use_memory:
        raise CalibrationError("config dùng memory: điểm calib phải chấm leave-one-video-out, script này chưa hỗ trợ.")


def quantile_threshold(pooled_scores: np.ndarray, target_fpr: float) -> float:
    """Ngưỡng nhỏ nhất sao cho không quá `target_fpr` mẫu gộp có điểm >= ngưỡng (xem `achieved_fpr` để biết thực tế)."""
    if not 0.0 < target_fpr < 1.0:
        raise CalibrationError(f"target_fpr phải trong (0, 1), nhận {target_fpr}")
    x = np.asarray(pooled_scores, dtype=np.float64).ravel()
    if x.size == 0:
        raise CalibrationError("không có điểm calib nào")
    if not np.isfinite(x).all():
        raise CalibrationError("điểm calib có NaN/Inf")
    xs = np.sort(x)
    n = xs.size
    max_above = int(np.floor(target_fpr * n))
    uniq = np.unique(xs)
    count_ge = n - np.searchsorted(xs, uniq, side="left")      # số mẫu có điểm >= từng giá trị, giảm dần
    ok = uniq[count_ge <= max_above]
    if ok.size:
        return float(ok[0])
    return float(np.nextafter(xs[-1], np.inf))                  # hạn mức quá chặt (hoặc hoà ở đỉnh): đặt ngay trên max


def achieved_fpr(scores: np.ndarray, threshold: float) -> float:
    x = np.asarray(scores, dtype=np.float64)
    return float(np.mean(x >= threshold)) if x.size else float("nan")


def _score_videos(rows: Sequence[dict[str, Any]], config: PipelineConfig, prompts: tuple[np.ndarray, np.ndarray],
                  cache_dir: str | Path, stats: CalibStats | None) -> list[np.ndarray]:
    out = []
    bad = []
    for r in rows:
        t = score_cached_video(r["video_id"], config, prompts[0], prompts[1], cache_dir, calib_stats=stats).scores
        want = rb.expected_samples(r["n_frames"], r["fps"], config.sample_fps)
        if abs(len(t) - want) > rb.LENGTH_TOLERANCE or len(t) == 0 or not np.isfinite(t).all():
            bad.append(r["video_id"])
            continue
        out.append(t)
    if bad:
        raise rb.BenchmarkInputError(f"Điểm xấu hoặc lệch độ dài ở {len(bad)} video: {bad}")
    return out


def operating_point(rows: Sequence[dict[str, Any]], annotations: dict[str, Annotation],
                    scores_pv: Sequence[np.ndarray], config: PipelineConfig, threshold: float) -> dict[str, Any]:
    """Số liệu vận hành tại ngưỡng (đọc, không chỉnh): FPR mẫu, FA/hour, recall sự cố, độ trễ, tải tầng 2."""
    cfg = with_threshold(config, threshold)
    n_norm_samples = n_norm_above = n_anom_samples = n_anom_above = 0
    normal_seconds = 0.0
    fa_windows = 0
    windows_total = 0
    incidents = detected = 0
    latencies: list[float] = []
    for r, s in zip(rows, scores_pv):
        vid, fps = r["video_id"], r["fps"]
        segs = annotations[vid].segments
        wins = candidate_windows(s, cfg.sample_fps, threshold, video_id=vid, min_len_s=cfg.min_len_s,
                                 merge_gap_s=cfg.merge_gap_s)
        windows_total += len(wins)
        above = int(np.count_nonzero(s >= threshold))
        if not segs:
            n_norm_samples += len(s)
            n_norm_above += above
            normal_seconds += r["n_frames"] / fps
            fa_windows += len(wins)
            continue
        n_anom_samples += len(s)
        n_anom_above += above
        for (a, b) in segs:
            g0, g1 = a / fps, (b + 1) / fps
            incidents += 1
            hit = [w for w in wins if w.start_s < g1 and w.end_s > g0]
            if hit:
                detected += 1
                lat = detection_latency_s(g0, min(w.start_s for w in hit))
                latencies.append(float(lat))
    if normal_seconds <= 0:
        raise CalibrationError("test không có video bình thường để đo FA/hour")
    return {
        "threshold": float(threshold),
        "normal_sample_fpr": n_norm_above / n_norm_samples if n_norm_samples else None,
        "anomalous_sample_tpr": n_anom_above / n_anom_samples if n_anom_samples else None,
        "fa_per_hour": false_alarms_per_hour(fa_windows, normal_seconds),
        "false_alarm_windows": fa_windows,
        "normal_hours": normal_seconds / 3600.0,
        "incidents": incidents,
        "incidents_detected": detected,
        "incident_recall": detected / incidents if incidents else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "windows_total": windows_total,
        "windows_per_video": windows_total / len(rows) if rows else None,
    }


def calibrate(config: PipelineConfig, calib_rows: Sequence[dict[str, Any]], prompts: tuple[np.ndarray, np.ndarray],
              cache_dir: str | Path, target_fpr: float, stats: CalibStats | None) -> dict[str, Any]:
    check_calibratable(config)
    calib_scores = _score_videos(calib_rows, config, prompts, cache_dir, stats)
    pooled = np.concatenate(calib_scores)
    thr = quantile_threshold(pooled, target_fpr)
    return {"threshold": thr, "target_fpr": target_fpr, "achieved_calib_fpr": achieved_fpr(pooled, thr),
            "n_calib_videos": len(calib_rows), "n_calib_samples": int(pooled.size)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="manifest test (chỉ để ĐO tại ngưỡng đã chốt)")
    p.add_argument("--calib-manifest", required=True)
    p.add_argument("--annotations", default=str(rb.DEFAULT_ANNOTATIONS))
    p.add_argument("--ablation", default="A4c")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--out-dir", default=str(rb.DEFAULT_RESULTS))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--prompt-cache-dir", default=str(rb.DEFAULT_PROMPT_CACHE))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.ablation not in ABLATIONS:
        p.error(f"ablation không có: {args.ablation}; hợp lệ: {sorted(ABLATIONS)}")
    config = ABLATIONS[args.ablation]
    check_calibratable(config)

    calib_rows = rb.read_benchmark_manifest(args.calib_manifest)
    test_rows = rb.read_benchmark_manifest(args.manifest)
    annotations = load_annotations(args.annotations)
    prompts = rb.load_or_embed_prompts(config, args.prompts, args.prompt_cache_dir, device=args.device)

    stats = None
    if config.per_video_norm and config.norm_ref == "calib":
        stats = rb.build_calib_stats(args.calib_manifest, config, args.cache_dir, prompts, BANK_MAX_FRAMES)
    cal = calibrate(config, calib_rows, prompts, args.cache_dir, args.target_fpr, stats)

    rb.check_inputs(test_rows, annotations, config, args.cache_dir)
    test_scores = _score_videos(test_rows, config, prompts, args.cache_dir, stats)
    op = operating_point(test_rows, annotations, test_scores, config, cal["threshold"])

    res = {
        "ablation": args.ablation, "config": asdict(config), "calib_stats": asdict(stats) if stats else None,
        "calibration": cal, "test_at_threshold": op,
        "note": "Ngưỡng chọn CHỈ từ calib_normal; test chỉ để đo. FPR test có thể lệch FPR calib (khác phân phối).",
        "commit": rb.git_commit(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = rb.write_json_atomic(Path(args.out_dir) / f"threshold_{args.ablation}.json", res)
    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
    print(f"{args.ablation}: tau={cal['threshold']:.4f} (target FPR {args.target_fpr:.0%}, calib achieved "
          f"{cal['achieved_calib_fpr']:.2%} on {cal['n_calib_videos']} videos / {cal['n_calib_samples']} samples)")
    print(f"  test normal sample FPR {f(op['normal_sample_fpr'] and op['normal_sample_fpr'] * 100)}%  "
          f"FA/hour {f(op['fa_per_hour'])}  incident recall {f(op['incident_recall'] and op['incident_recall'] * 100)}% "
          f"({op['incidents_detected']}/{op['incidents']})  mean latency {f(op['mean_latency_s'], 1)} s  "
          f"windows {op['windows_total']} ({f(op['windows_per_video'], 1)}/video) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
