# -*- coding: utf-8 -*-
"""Đóng băng tầng 1 và xuất cửa sổ ứng viên cho tầng 2 (TH6): results/windows/<video_id>.json + _index.json.

    python scripts/export_windows.py --manifest manifests/ucf_test.csv --calib-manifest manifests/ucf_calib_normal.csv \\
        --ablation A4c --threshold-file results/threshold_A4c.json

Ngưỡng và thống kê chuẩn hoá lấy từ file `scripts/calibrate_threshold.py` đã ghi (calib_normal), KHÔNG tính lại trên test.
Chỉ ghi video có >= 1 cửa sổ (tầng 2 không có việc ở video khác); `_index.json` liệt kê mọi cửa sổ theo thứ tự cố định
để Phước chạy/tiếp tục (resume) theo chỉ số. Kết quả tất định: cùng cache + cùng config = cùng file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_benchmark as rb  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR  # noqa: E402
from vad.pipeline import ABLATIONS, DEFAULT_PROMPTS, CalibStats, PipelineConfig, score_cached_video, with_threshold  # noqa: E402
from vad.results import DEFAULT_RESULTS_DIR, RunResult, make_metadata, windows_path  # noqa: E402

INDEX_NAME = "_index.json"


class ExportError(ValueError):
    """File ngưỡng không khớp config / không hợp lệ."""


def load_threshold_file(path: str | Path, ablation: str, config: PipelineConfig) -> tuple[float, CalibStats | None]:
    with open(path, encoding="utf-8-sig") as f:
        d = json.load(f)
    if d.get("ablation") != ablation:
        raise ExportError(f"{path} là của ablation {d.get('ablation')!r}, không phải {ablation!r}")
    if d.get("config") != asdict(config):
        raise ExportError(f"config trong {path} khác ABLATIONS[{ablation!r}] hiện tại: ngưỡng cũ không dùng được, "
                          "chạy lại scripts/calibrate_threshold.py")
    thr = d["calibration"]["threshold"]
    if not isinstance(thr, (int, float)) or not np.isfinite(thr):
        raise ExportError(f"ngưỡng không hợp lệ trong {path}: {thr!r}")
    cs = d.get("calib_stats")
    return float(thr), (CalibStats(**cs) if cs else None)


def export_windows(rows: Sequence[dict[str, Any]], config: PipelineConfig, prompts: tuple[np.ndarray, np.ndarray],
                   cache_dir: str | Path, out_root: str | Path, threshold: float, stats: CalibStats | None,
                   ablation: str, commit: str = "") -> dict[str, Any]:
    cfg = with_threshold(config, threshold)
    bad = [r["video_id"] for r in rows
           if not rb.has_cached(cfg.model_id, r["video_id"], cfg.sample_fps, cache_dir)]
    if bad:
        raise rb.BenchmarkInputError(f"Thiếu cache embedding cho {len(bad)} video: {bad}")
    entries: list[dict[str, Any]] = []
    n_videos_with = 0
    for r in rows:
        vid = r["video_id"]
        res = score_cached_video(vid, cfg, prompts[0], prompts[1], cache_dir, calib_stats=stats)
        if not np.isfinite(res.scores).all():
            raise rb.BenchmarkInputError(f"Điểm NaN/Inf ở {vid}")
        if not res.windows:
            continue
        n_videos_with += 1
        run = RunResult(video_id=vid, sample_fps=cfg.sample_fps, anomaly_scores=(), candidate_windows=res.windows,
                        metadata=make_metadata(model=cfg.model_id, commit=commit, tier="1", ablation=ablation,
                                               threshold=threshold, fps=r["fps"], n_frames=r["n_frames"]))
        run.save_windows(out_root)
        for i, w in enumerate(res.windows):
            entries.append({"video_id": vid, "window_index": i, "start_s": w.start_s, "end_s": w.end_s,
                            "score": w.score, "duration_s": w.duration_s})
    index = {"ablation": ablation, "threshold": threshold, "n_videos_total": len(rows),
             "n_videos_with_windows": n_videos_with, "n_windows": len(entries), "commit": commit,
             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "windows": entries}
    rb.write_json_atomic(Path(out_root) / "windows" / INDEX_NAME, index)
    return index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ablation", default="A4c")
    p.add_argument("--threshold-file", required=True)
    p.add_argument("--out-root", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--prompt-cache-dir", default=str(rb.DEFAULT_PROMPT_CACHE))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.ablation not in ABLATIONS:
        p.error(f"ablation không có: {args.ablation}; hợp lệ: {sorted(ABLATIONS)}")
    config = ABLATIONS[args.ablation]
    thr, stats = load_threshold_file(args.threshold_file, args.ablation, config)
    rows = rb.read_benchmark_manifest(args.manifest)
    prompts = rb.load_or_embed_prompts(config, args.prompts, args.prompt_cache_dir, device=args.device)
    idx = export_windows(rows, config, prompts, args.cache_dir, args.out_root, thr, stats, args.ablation,
                         commit=rb.git_commit())
    print(f"{args.ablation} tau={thr:.4f}: {idx['n_windows']} cửa sổ trong {idx['n_videos_with_windows']}/"
          f"{idx['n_videos_total']} video -> {Path(args.out_root) / 'windows'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
