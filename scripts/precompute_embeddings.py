# -*- coding: utf-8 -*-
"""Tính embedding cho một shard video, ghi cache resumable. Mỗi máy chạy một shard khác nhau.

    python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 1/4 --dry-run
    python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 2/4 --fps 1 --dtype float32
    # SigLIP 2 + CLIP B/32 (baseline A0) trong MỘT lượt giải mã video:
    python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 2/4 \\
        --model google/siglip2-base-patch16-384 --model openai/clip-vit-base-patch32

Manifest CSV cột: video_id,path,dataset,split[,n_frames,fps]. `--dry-run` chỉ liệt kê việc sẽ làm (không cần GPU/model).
Mọi shard PHẢI dùng cùng danh sách --model, --fps; dtype tính toán có thể khác (1060 dùng float32), cache luôn lưu fp16.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.sharding import parse_shard_spec, select_shard  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached, save_embeddings  # noqa: E402
from vad.embed.siglip import DEFAULT_MODEL_ID  # noqa: E402
from vad.io.video import iter_sampled_frames  # noqa: E402
from vad.types import VideoRef  # noqa: E402


def read_manifest(path: str | Path) -> list[VideoRef]:
    refs: list[VideoRef] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            refs.append(VideoRef(
                video_id=row["video_id"], path=row["path"], dataset=row["dataset"], split=row["split"],
                n_frames=int(row["n_frames"]) if row.get("n_frames") else None,
                fps=float(row["fps"]) if row.get("fps") else 30.0,
            ))
    ids = [r.video_id for r in refs]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest có video_id trùng nhau")
    return refs


def plan_work(refs: list[VideoRef], shard_index: int, n_shards: int, model_id: str | Sequence[str], fps: float,
              cache_dir: Path) -> tuple[list[VideoRef], list[VideoRef]]:
    """-> (cần làm, đã có cache) trong shard này. Với nhiều model: "đã có" = MỌI model đều có cache."""
    model_ids = [model_id] if isinstance(model_id, str) else list(model_id)
    mine = set(select_shard([r.video_id for r in refs], shard_index, n_shards))
    todo, done = [], []
    for r in refs:
        if r.video_id not in mine:
            continue
        cached = all(has_cached(m, r.video_id, fps, cache_dir) for m in model_ids)
        (done if cached else todo).append(r)
    return todo, done

def process_video_multi(ref: VideoRef, embedders: Sequence, sample_fps: float, cache_dir: Path) -> int:
    """Video -> cache .npy cho MỖI embedder, giải mã video đúng MỘT lần (G8: giải mã 30 fps mới là phần đắt).

    Embedder đã có cache cho video này thì bỏ qua (resume theo từng model). Trả số khung đã lấy mẫu
    (0 = video hỏng/không có khung, hoặc mọi model đã có cache -> không ghi gì).
    Giữ tối đa `batch` khung trong RAM một lúc (batch = nhỏ nhất trong các embedder): không nạp cả video.
    """
    pending = [e for e in embedders if not has_cached(e.model_id, ref.video_id, sample_fps, cache_dir)]
    if not pending:
        return 0
    ids = [e.model_id for e in pending]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Hai embedder cùng model_id sẽ ghi đè cache của nhau: {ids}")
    batch_size = min(getattr(e, "batch_size", 8) for e in pending)

    batch: list = []
    chunks: dict[str, list[np.ndarray]] = {e.model_id: [] for e in pending}

    def flush() -> None:
        for e in pending:
            chunks[e.model_id].append(e.embed_images(batch))

    for _, frame in iter_sampled_frames(ref.path, sample_fps):
        batch.append(frame)
        if len(batch) >= batch_size:
            flush()
            batch = []
    if batch:
        flush()
    n_chunks = {len(c) for c in chunks.values()}
    if n_chunks == {0}:
        return 0
    # Ghi cache chỉ SAU KHI mọi model xong: một model lỗi giữa chừng không để lại cache lệch nhau.
    n = 0
    for e in pending:
        embs = np.concatenate(chunks[e.model_id], axis=0)
        save_embeddings(embs, e.model_id, ref.video_id, sample_fps, cache_dir)
        n = int(embs.shape[0])
    return n

def process_video(ref: VideoRef, embedder, sample_fps: float, cache_dir: Path) -> int:
    """Bản một model của `process_video_multi` (giữ API cũ)."""
    return process_video_multi(ref, [embedder], sample_fps, cache_dir)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shard", default="1/1", help="i/N, đếm từ 1 (ví dụ 2/4)")
    p.add_argument("--model", action="append", default=None,
                   help=f"HF model id; lặp lại để embed nhiều model trong một lượt giải mã (mặc định {DEFAULT_MODEL_ID})")
    p.add_argument("--fps", type=float, default=1.0, help="tần suất lấy mẫu (khung/giây)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    models = list(dict.fromkeys(args.model or [DEFAULT_MODEL_ID]))   # bỏ trùng, giữ thứ tự
    idx, n = parse_shard_spec(args.shard)
    refs = read_manifest(args.manifest)
    todo, done = plan_work(refs, idx, n, models, args.fps, Path(args.cache_dir))
    print(f"shard {idx + 1}/{n}: {len(todo)} cần làm, {len(done)} đã có cache, tổng manifest {len(refs)}")
    if args.dry_run:
        for r in todo:
            print(f"  [todo] {r.video_id}  ({r.path})")
        return 0

    from vad.embed.siglip import SigLIPEmbedder

    embedders = [SigLIPEmbedder(model_id=m, device=args.device, dtype=args.dtype) for m in models]
    # KHÔNG gọi embedder.load() ở đây: lần encode đầu tiên sẽ tự nạp. Nhờ vậy chạy với video hỏng/
    # không tồn tại vẫn báo lỗi mà không cần tải model (CI không bao giờ được tải model).
    total_frames, failed = 0, []
    t0 = time.time()
    for i, r in enumerate(todo, start=1):
        t1 = time.time()
        try:
            n_frames = process_video_multi(r, embedders, args.fps, Path(args.cache_dir))
        except Exception as e:  # video hỏng không được làm sập cả shard
            failed.append(r.video_id)
            print(f"  [{i}/{len(todo)}] LỖI {r.video_id}: {type(e).__name__}: {e}")
            continue
        total_frames += n_frames
        rate = n_frames / max(time.time() - t1, 1e-9)
        print(f"  [{i}/{len(todo)}] {r.video_id}: {n_frames} khung, {rate:.1f} khung/s")

    dt = time.time() - t0
    print(f"\nxong: {len(todo) - len(failed)}/{len(todo)} video, {total_frames} khung, "
          f"{dt / 60:.1f} phút, TB {total_frames / max(dt, 1e-9):.1f} khung/s")
    if failed:
        print(f"lỗi: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
