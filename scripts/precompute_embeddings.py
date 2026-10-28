# -*- coding: utf-8 -*-
"""Tính embedding cho một shard video, ghi cache resumable. Mỗi máy chạy một shard khác nhau.

    python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 1/4 --dry-run
    python scripts/precompute_embeddings.py --manifest manifests/ucf_test.csv --shard 2/4 --fps 1 --dtype float32

Manifest CSV cột: video_id,path,dataset,split[,n_frames,fps]. `--dry-run` chỉ liệt kê việc sẽ làm (không cần GPU/model).
Mọi shard PHẢI dùng cùng --model, --fps; dtype tính toán có thể khác (1060 dùng float32), cache luôn lưu fp16.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.sharding import parse_shard_spec, select_shard  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached  # noqa: E402
from vad.embed.siglip import DEFAULT_MODEL_ID  # noqa: E402
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


def plan_work(refs: list[VideoRef], shard_index: int, n_shards: int, model_id: str, fps: float,
              cache_dir: Path) -> tuple[list[VideoRef], list[VideoRef]]:
    """-> (cần làm, đã có cache) trong shard này."""
    mine = set(select_shard([r.video_id for r in refs], shard_index, n_shards))
    todo, done = [], []
    for r in refs:
        if r.video_id not in mine:
            continue
        (done if has_cached(model_id, r.video_id, fps, cache_dir) else todo).append(r)
    return todo, done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shard", default="1/1", help="i/N, đếm từ 1 (ví dụ 2/4)")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--fps", type=float, default=1.0, help="tần suất lấy mẫu (khung/giây)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    idx, n = parse_shard_spec(args.shard)
    refs = read_manifest(args.manifest)
    todo, done = plan_work(refs, idx, n, args.model, args.fps, Path(args.cache_dir))
    print(f"shard {idx + 1}/{n}: {len(todo)} cần làm, {len(done)} đã có cache, tổng manifest {len(refs)}")
    if args.dry_run:
        for r in todo:
            print(f"  [todo] {r.video_id}  ({r.path})")
        return 0
    raise NotImplementedError("Chạy thật cần video sampler + SigLIPEmbedder (việc của T1) — dùng --dry-run trước")


if __name__ == "__main__":
    raise SystemExit(main())
