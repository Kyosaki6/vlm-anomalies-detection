# -*- coding: utf-8 -*-
"""A0 gate, kiểm tra #3 (TH4): lấy mẫu đúng giao thức LAVAD (mỗi 16 khung gốc) thay vì 1 fps của cache.

    python scripts/a0_stride_check.py --manifest manifests/ucf_test.csv --limit 20     # đo tốc độ + số sơ bộ
    python scripts/a0_stride_check.py --manifest manifests/ucf_test.csv                # đủ 290 video (CPU, resumable)

Cache 1 fps cố định nên không thể "lấy mẫu lại"; phải giải mã lại video. Embedding ghi ở cache/stride16/ (KHÔNG đụng
cache/embeddings/). Điểm A0 = đúng config `ABLATIONS["A0"]`; mở rộng ra frame bằng np.repeat(stride) cắt về n_frames
như LAVAD (src/eval.py:78,146). Ghi results/a0_stride16[_limitN].json. Đây là CHẨN ĐOÁN, không thay A0 chính.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from scripts.run_benchmark import (DEFAULT_ANNOTATIONS, DEFAULT_PROMPT_CACHE, DEFAULT_RESULTS,  # noqa: E402
                                   BenchmarkInputError, evaluate, load_or_embed_prompts, read_benchmark_manifest,
                                   select_rows, write_json_atomic)
from vad.datasets.ucf_crime import load_annotations, load_gt  # noqa: E402
from vad.pipeline import ABLATIONS, run_tier1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRIDE_CACHE = ROOT / "cache" / "stride16"


def iter_stride_frames(path: str | Path, stride: int):
    """RGB khung 0, stride, 2*stride, ... (đọc tuần tự bằng grab, chỉ retrieve khung cần)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV không mở được video: {path}")
    try:
        i = 0
        while cap.grab():
            if i % stride == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def embed_video(path: str | Path, embedder, stride: int, batch: int) -> np.ndarray:
    chunks, buf = [], []
    for fr in iter_stride_frames(path, stride):
        buf.append(fr)
        if len(buf) >= batch:
            chunks.append(embedder.embed_images(buf))
            buf = []
    if buf:
        chunks.append(embedder.embed_images(buf))
    if not chunks:
        raise RuntimeError(f"Video không có khung nào: {path}")
    return np.concatenate(chunks, axis=0)


def expand_repeat(scores: np.ndarray, stride: int, n_frames: int) -> np.ndarray:
    """np.repeat(scores, stride)[:n_frames] như LAVAD. Thiếu khung (video thật ngắn hơn n_frames) -> lặp mẫu cuối."""
    out = np.repeat(np.asarray(scores, dtype=np.float64), stride)[:n_frames]
    if len(out) < n_frames:
        out = np.concatenate([out, np.full(n_frames - len(out), out[-1])])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(ROOT / "manifests" / "ucf_test.csv"))
    p.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache-dir", default=str(STRIDE_CACHE))
    p.add_argument("--out-dir", default=str(DEFAULT_RESULTS))
    args = p.parse_args(argv)

    config = ABLATIONS["A0"]
    rows_all = read_benchmark_manifest(args.manifest)
    paths = {}
    import csv
    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            paths[r["video_id"]] = ROOT / r["path"]
    rows = select_rows(rows_all, args.limit)
    annotations = load_annotations(args.annotations)
    normal, abnormal = load_or_embed_prompts(config, cache_dir=DEFAULT_PROMPT_CACHE, device=args.device)

    from vad.embed.siglip import SigLIPEmbedder
    embedder = SigLIPEmbedder(model_id=config.model_id, device=args.device, batch_size=args.batch)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scores_pv, labels_pv, done_frames = [], [], 0
    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        f = cache / f"{vid}__stride{args.stride}.npy"
        t1 = time.time()
        if f.is_file():
            embs = np.load(f)
        else:
            embs = embed_video(paths[vid], embedder, args.stride, args.batch)
            tmp = f.with_name(f.name + ".tmp.npy")
            np.save(tmp, embs.astype(np.float16))
            tmp.replace(f)
            embs = embs.astype(np.float16)
        embs = embs.astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        want = -(-r["n_frames"] // args.stride)
        if abs(len(embs) - want) > 2:
            raise BenchmarkInputError(f"{vid}: {len(embs)} mẫu, cần ~{want}")
        s = run_tier1(embs, normal, abnormal, config, video_id=vid).scores
        if not np.isfinite(s).all():
            raise BenchmarkInputError(f"{vid}: điểm NaN/Inf")
        scores_pv.append(expand_repeat(s, args.stride, r["n_frames"]))
        labels_pv.append(load_gt(annotations, vid, r["n_frames"]))
        done_frames += len(embs)
        print(f"[{i}/{len(rows)}] {vid}: {len(embs)} mẫu, {time.time() - t1:.1f}s (tổng {time.time() - t0:.0f}s)", flush=True)

    ev = evaluate(rows, scores_pv, labels_pv, args.n_bootstraps, args.seed)
    ev.pop("per_video")
    res = {"check": "A0 stride sampling (LAVAD every 16th frame)", "stride": args.stride, "config": config.model_id,
           "limit": args.limit, "elapsed_s": round(time.time() - t0, 1), **ev,
           "note": "CHẨN ĐOÁN kiểm tra #3 của a0-gate.md, không thay kết quả A0 chính"}
    tag = f"_limit{args.limit}" if args.limit else ""
    out = write_json_atomic(Path(args.out_dir) / f"a0_stride{args.stride}{tag}.json", res)
    fmt = lambda v: "nan" if v is None else f"{100 * v:.2f}"  # noqa: E731
    print(f"A0@stride{args.stride}: micro {fmt(ev['auc_micro'])} CI95 [{fmt(ev['ci_lo'])}, {fmt(ev['ci_hi'])}] "
          f"macro {fmt(ev['auc_macro_mean_per_video'])} ({ev['n_videos_anomalous']}+{ev['n_videos_normal']} video) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
