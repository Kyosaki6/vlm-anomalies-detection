# -*- coding: utf-8 -*-
"""Benchmark tier-1 trên UCF-Crime test (TH3): embedding đã cache -> điểm -> AUC frame-level micro + CI theo VIDEO.

    python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A0
    python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --all --n-bootstraps 1000
    python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A0 --limit 5   # chạy thử

Không nạp model khi benchmark: embedding prompt được cache ở cache/prompts/<model>__<set>.npz (lần đầu mới cần model,
chạy CPU là đủ). Thiếu cache video, lệch độ dài, điểm NaN/rỗng = LỖI, không bao giờ lặng lẽ bỏ qua video.
Ghi results/ablation_<AID>.json (ghi nguyên tử).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.ucf_crime import Annotation, load_annotations, load_gt  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR, _slug, has_cached, load_embeddings  # noqa: E402
from vad.eval.metrics import fast_frame_level_auc, fast_video_bootstrap_auc_ci  # noqa: E402
from vad.pipeline import (ABLATIONS, DEFAULT_PROMPTS, CalibStats, PipelineConfig, build_memory_bank,  # noqa: E402
                          compute_calib_stats, load_detection_prompts, score_cached_video)
from vad.temporal.ops import expand_to_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "datasets" / "ucf_crime" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
DEFAULT_PROMPT_CACHE = ROOT / "cache" / "prompts"
DEFAULT_RESULTS = ROOT / "results"
LENGTH_TOLERANCE = 2


class BenchmarkInputError(ValueError):
    """Đầu vào benchmark hỏng (thiếu cache, lệch độ dài, điểm xấu...). Luôn dừng, không bỏ qua video."""


# --- manifest ------------------------------------------------------------------------

def read_benchmark_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("n_frames") or not r.get("fps"):
                raise BenchmarkInputError(f"Manifest thiếu n_frames/fps ở {r.get('video_id')!r}: benchmark cần cả hai")
            rows.append({"video_id": r["video_id"], "n_frames": int(r["n_frames"]), "fps": float(r["fps"])})
    ids = [r["video_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise BenchmarkInputError("Manifest có video_id trùng nhau")
    return rows


def select_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """`--limit N`: lấy N video cách đều (manifest xếp theo lớp, lấy N đầu sẽ toàn video bất thường)."""
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise BenchmarkInputError("--limit phải >= 1")
    return [rows[i] for i in np.linspace(0, len(rows) - 1, limit).round().astype(int)]


# --- prompt cache --------------------------------------------------------------------

def prompt_cache_path(model_id: str, prompt_set: str, cache_dir: str | Path = DEFAULT_PROMPT_CACHE) -> Path:
    return Path(cache_dir) / f"{_slug(model_id)}__{prompt_set}.npz"


def _default_embedder_factory(model_id: str, device: str | None):
    def make():
        from vad.embed.siglip import SigLIPEmbedder
        return SigLIPEmbedder(model_id=model_id, device=device)
    return make


def load_or_embed_prompts(config: PipelineConfig, prompts_path: str | Path = DEFAULT_PROMPTS,
                          cache_dir: str | Path = DEFAULT_PROMPT_CACHE,
                          embedder_factory: Callable[[], Any] | None = None,
                          device: str | None = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(normal, abnormal) embedding của prompt. Cache khớp (cùng model, set, cùng CHỮ prompt) thì không nạp model."""
    normal_txt, abnormal_txt = load_detection_prompts(config.prompt_set, prompts_path)
    path = prompt_cache_path(config.model_id, config.prompt_set, cache_dir)
    if path.is_file():
        with np.load(path) as z:
            same = (list(z["normal_texts"]) == normal_txt and list(z["abnormal_texts"]) == abnormal_txt
                    and str(z["model_id"]) == config.model_id)
            if same:
                print(f"prompt cache hit: {path.name}")
                return z["normal"].astype(np.float32), z["abnormal"].astype(np.float32)
        print(f"prompt cache STALE (prompts.yaml hoặc model đổi): {path.name}, tính lại")

    factory = embedder_factory or _default_embedder_factory(config.model_id, device)
    embedder = factory()
    if embedder.model_id != config.model_id:
        raise BenchmarkInputError(f"embedder là {embedder.model_id!r} nhưng config cần {config.model_id!r}")
    normal = np.asarray(embedder.embed_texts(normal_txt), dtype=np.float32)
    abnormal = np.asarray(embedder.embed_texts(abnormal_txt), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, normal=normal, abnormal=abnormal, normal_texts=np.array(normal_txt),
                 abnormal_texts=np.array(abnormal_txt), model_id=np.array(config.model_id))
    os.replace(tmp, path)
    print(f"prompt cache miss: đã tính và lưu {path.name}")
    return normal, abnormal


# --- chấm điểm -----------------------------------------------------------------------

def shift_labels(y: np.ndarray, k: int) -> np.ndarray:
    """Dịch nhãn k frame sang phải (k<0: sang trái), giữ độ dài. Chỉ để chẩn đoán lệch GT (--label-shift)."""
    if k == 0:
        return y
    out = np.zeros_like(y)
    if abs(k) >= len(y):
        return out
    if k > 0:
        out[k:] = y[:-k]
    else:
        out[:k] = y[-k:]
    return out


def expected_samples(n_frames: int, fps: float, sample_fps: float) -> int:
    return math.ceil(n_frames * sample_fps / fps)


def check_inputs(rows: list[dict[str, Any]], annotations: dict[str, Annotation], config: PipelineConfig,
                 cache_dir: str | Path) -> None:
    """Kiểm TRƯỚC khi chấm: thiếu cache / thiếu annotation / lệch độ dài -> một lỗi liệt kê đủ mọi video."""
    missing = [r["video_id"] for r in rows if not has_cached(config.model_id, r["video_id"], config.sample_fps, cache_dir)]
    if missing:
        raise BenchmarkInputError(f"Thiếu cache embedding ({config.model_id}, fps {config.sample_fps:g}) "
                                  f"cho {len(missing)} video: {missing}")
    unannotated = [r["video_id"] for r in rows if r["video_id"] not in annotations]
    if unannotated:
        raise BenchmarkInputError(f"{len(unannotated)} video không có trong file annotation (sẽ bị coi nhầm là bình "
                                  f"thường): {unannotated}")


def score_all(rows: list[dict[str, Any]], annotations: dict[str, Annotation], config: PipelineConfig,
              normal_embs: np.ndarray, abnormal_embs: np.ndarray, cache_dir: str | Path,
              normal_bank: np.ndarray | None = None, label_shift: int = 0,
              calib_stats: CalibStats | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """-> (điểm theo frame, nhãn theo frame) cho từng video, cùng thứ tự `rows`."""
    scores_pv: list[np.ndarray] = []
    labels_pv: list[np.ndarray] = []
    bad_len: list[str] = []
    bad_score: list[str] = []
    for r in rows:
        vid = r["video_id"]
        res = score_cached_video(vid, config, normal_embs, abnormal_embs, cache_dir, normal_bank=normal_bank,
                                 calib_stats=calib_stats)
        t = res.scores
        want = expected_samples(r["n_frames"], r["fps"], config.sample_fps)
        if abs(len(t) - want) > LENGTH_TOLERANCE:
            bad_len.append(f"{vid} (cache {len(t)} mẫu, cần ~{want})")
            continue
        if len(t) == 0 or not np.isfinite(t).all():
            bad_score.append(vid)
            continue
        frame_scores = expand_to_frames(t, config.sample_fps, r["fps"], r["n_frames"])
        labels = shift_labels(load_gt(annotations, vid, r["n_frames"]), label_shift)
        scores_pv.append(frame_scores)
        labels_pv.append(labels)
    if bad_len:
        raise BenchmarkInputError(f"Lệch độ dài cache vs n_frames/fps (> {LENGTH_TOLERANCE} mẫu) ở {len(bad_len)} "
                                  f"video: {bad_len}")
    if bad_score:
        raise BenchmarkInputError(f"Điểm rỗng hoặc NaN/Inf ở {len(bad_score)} video: {bad_score}")
    return scores_pv, labels_pv


def _nan_to_none(x: float) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


def evaluate(rows: list[dict[str, Any]], scores_pv: list[np.ndarray], labels_pv: list[np.ndarray],
             n_bootstraps: int, seed: int) -> dict[str, Any]:
    """AUC micro (giao thức LAVAD) + CI bootstrap theo video + AUC từng video + macro để đối chiếu."""
    micro = fast_frame_level_auc(scores_pv, labels_pv)
    lo, hi = fast_video_bootstrap_auc_ci(scores_pv, labels_pv, n_bootstraps=n_bootstraps, seed=seed)
    per_video = []
    for r, s, y in zip(rows, scores_pv, labels_pv):
        per_video.append({"video_id": r["video_id"], "n_frames": int(len(s)), "n_anomalous_frames": int(y.sum()),
                          "auc": _nan_to_none(fast_frame_level_auc([s], [y]))})
    macro_vals = [v["auc"] for v in per_video if v["auc"] is not None]
    n_anom = sum(1 for y in labels_pv if y.any())
    return {
        "auc_micro": _nan_to_none(micro), "ci_lo": _nan_to_none(lo), "ci_hi": _nan_to_none(hi),
        "auc_macro_mean_per_video": _nan_to_none(float(np.mean(macro_vals))) if macro_vals else None,
        "n_videos": len(rows), "n_videos_anomalous": n_anom, "n_videos_normal": len(rows) - n_anom,
        "n_frames": int(sum(len(s) for s in scores_pv)),
        "n_anomalous_frames": int(sum(int(y.sum()) for y in labels_pv)),
        "per_video": per_video,
    }


# --- đầu ra --------------------------------------------------------------------------

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)
    return path


def result_file(out_dir: str | Path, ablation_id: str, tag: str = "") -> Path:
    return Path(out_dir) / f"ablation_{ablation_id}{('_' + tag) if tag else ''}.json"


def run_ablation(ablation_id: str, config: PipelineConfig, rows: list[dict[str, Any]],
                 annotations: dict[str, Annotation], cache_dir: str | Path, prompts: tuple[np.ndarray, np.ndarray],
                 n_bootstraps: int = 1000, seed: int = 42, normal_bank: np.ndarray | None = None,
                 label_shift: int = 0, manifest: str = "", commit: str | None = None,
                 calib_stats: CalibStats | None = None) -> dict[str, Any]:
    """Một ablation -> dict kết quả (chưa ghi đĩa)."""
    t0 = time.time()
    check_inputs(rows, annotations, config, cache_dir)
    scores_pv, labels_pv = score_all(rows, annotations, config, prompts[0], prompts[1], cache_dir,
                                     normal_bank=normal_bank, label_shift=label_shift, calib_stats=calib_stats)
    t_score = time.time() - t0
    ev = evaluate(rows, scores_pv, labels_pv, n_bootstraps, seed)
    return {
        "ablation": ablation_id, "config": asdict(config), "manifest": manifest,
        "seed": seed, "n_bootstraps": n_bootstraps, "label_shift": label_shift,
        "calib_stats": asdict(calib_stats) if calib_stats is not None else None,
        "protocol": "micro-pooled frame-level AUC (LAVAD), CI bootstrap over videos",
        "commit": commit if commit is not None else git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": {"score": round(t_score, 2), "total": round(time.time() - t0, 2)},
        **ev,
    }


def build_calib_stats(manifest: str | Path, config: PipelineConfig, cache_dir: str | Path,
                      prompts: tuple[np.ndarray, np.ndarray], max_frames: int) -> CalibStats:
    rows = read_benchmark_manifest(manifest)
    missing = [r["video_id"] for r in rows if not has_cached(config.model_id, r["video_id"], config.sample_fps, cache_dir)]
    if missing:
        raise BenchmarkInputError(f"Thiếu cache calib_normal cho thống kê chuẩn hoá ({len(missing)} video): {missing}")
    return compute_calib_stats([load_embeddings(config.model_id, r["video_id"], config.sample_fps, cache_dir)
                                for r in rows], prompts[0], prompts[1], config, bank_max_frames=max_frames)

def build_bank(manifest: str | Path, config: PipelineConfig, cache_dir: str | Path, max_frames: int) -> np.ndarray:
    rows = read_benchmark_manifest(manifest)
    missing = [r["video_id"] for r in rows if not has_cached(config.model_id, r["video_id"], config.sample_fps, cache_dir)]
    if missing:
        raise BenchmarkInputError(f"Thiếu cache calib_normal cho memory bank ({len(missing)} video): {missing}")
    return build_memory_bank([load_embeddings(config.model_id, r["video_id"], config.sample_fps, cache_dir)
                              for r in rows], max_frames=max_frames)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    p.add_argument("--ablation", action="append", default=[], help=f"một trong {sorted(ABLATIONS)}; lặp lại được")
    p.add_argument("--all", action="store_true", help="chạy mọi ablation (A0..A5, A3c..A5c)")
    p.add_argument("--limit", type=int, default=None, help="chỉ chạy N video cách đều (chạy thử)")
    p.add_argument("--out-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--prompt-cache-dir", default=str(DEFAULT_PROMPT_CACHE))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--device", default="cpu", help="thiết bị cho text tower khi cache prompt chưa có")
    p.add_argument("--calib-manifest", default=None, help="manifest calib_normal, bắt buộc cho ablation dùng memory (A5)")
    p.add_argument("--bank-max-frames", type=int, default=20000)
    p.add_argument("--label-shift", type=int, default=0, help="CHẨN ĐOÁN: dịch GT k frame (bước 2 của bảng debug A0)")
    p.add_argument("--logit-scale", type=float, default=None, help="CHẨN ĐOÁN: ghi đè logit_scale của config")
    p.add_argument("--tag", default="", help="hậu tố tên file, để chạy chẩn đoán không đè kết quả chính")
    args = p.parse_args(argv)

    ids = list(ABLATIONS) if args.all else list(dict.fromkeys(args.ablation))
    if not ids:
        p.error("cần --ablation <ID> hoặc --all")
    unknown = [a for a in ids if a not in ABLATIONS]
    if unknown:
        p.error(f"ablation không có: {unknown}; hợp lệ: {sorted(ABLATIONS)}")
    diagnostic = args.label_shift != 0 or args.logit_scale is not None
    if diagnostic and not args.tag:
        p.error("chạy chẩn đoán (--label-shift/--logit-scale) phải kèm --tag để không đè kết quả chính")

    rows = select_rows(read_benchmark_manifest(args.manifest), args.limit)
    annotations = load_annotations(args.annotations)
    print(f"{len(rows)} video, {len(annotations)} annotation, ablation: {ids}")

    for aid in ids:
        config = ABLATIONS[aid]
        if args.logit_scale is not None:
            config = replace(config, logit_scale=args.logit_scale)
        prompts = load_or_embed_prompts(config, args.prompts, args.prompt_cache_dir, device=args.device)
        bank = None
        if config.use_memory:
            if not args.calib_manifest:
                p.error(f"{aid} dùng normal memory: cần --calib-manifest manifests/ucf_calib_normal.csv")
            bank = build_bank(args.calib_manifest, config, args.cache_dir, args.bank_max_frames)
        stats = None
        if config.per_video_norm and config.norm_ref == "calib":
            if not args.calib_manifest:
                p.error(f"{aid} chuẩn hoá theo calib_normal: cần --calib-manifest manifests/ucf_calib_normal.csv")
            stats = build_calib_stats(args.calib_manifest, config, args.cache_dir, prompts, args.bank_max_frames)
        res = run_ablation(aid, config, rows, annotations, args.cache_dir, prompts, n_bootstraps=args.n_bootstraps,
                           seed=args.seed, normal_bank=bank, label_shift=args.label_shift, manifest=str(args.manifest),
                           calib_stats=stats)
        res["limit"] = args.limit
        res["note"] = ("CHẨN ĐOÁN, không phải kết quả chính" if diagnostic else
                       ("chạy thử --limit, không phải kết quả chính" if args.limit else ""))
        out = write_json_atomic(result_file(args.out_dir, aid, args.tag), res)
        auc, lo, hi = res["auc_micro"], res["ci_lo"], res["ci_hi"]
        fmt = lambda v: "nan" if v is None else f"{100 * v:.2f}"  # noqa: E731
        print(f"{aid}: AUC micro {fmt(auc)}  CI95 [{fmt(lo)}, {fmt(hi)}]  macro {fmt(res['auc_macro_mean_per_video'])}  "
              f"({res['n_videos_anomalous']} bất thường + {res['n_videos_normal']} bình thường, "
              f"{res['n_frames']} frame, {res['elapsed_s']['total']}s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
