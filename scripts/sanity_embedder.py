# -*- coding: utf-8 -*-
"""Kiểm tra nhanh embedder thật: ảnh trong data/ -> text score + hạng lớp dự đoán. KHÔNG cần dataset.

    python scripts/sanity_embedder.py --device cpu
    python scripts/sanity_embedder.py            # tự chọn GPU nếu có

Mục đích: bắt lỗi plumbing (normalize sai, kênh màu sai, lấy nhầm đầu ra model) trong 1 phút, TRƯỚC khi
đem đi cache 290 video. Đây KHÔNG phải số liệu để báo cáo: 15 ảnh, 3 nhóm, không có ý nghĩa thống kê.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from vad.embed.siglip import SigLIPEmbedder  # noqa: E402
from vad.score.text import text_anomaly_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_prompts(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--prompts", default=str(ROOT / "configs" / "prompts.yaml"))
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32")
    args = p.parse_args(argv)

    files = sorted(Path(args.data_dir).glob("*.jpg"))
    if not files:
        print(f"Không có ảnh .jpg trong {args.data_dir}", file=sys.stderr)
        return 2

    emb = SigLIPEmbedder(device=args.device, dtype=args.dtype)
    emb.load()                       # nạp model TRƯỚC khi bấm giờ, nếu không sẽ tính cả 14 s load vào tốc độ
    prompts = load_prompts(Path(args.prompts))

    t0 = time.time()
    images = [Image.open(f).convert("RGB") for f in files]
    frame_embs = emb.embed_images(images)
    dt = time.time() - t0

    det = prompts["detection"]["ensemble"]
    normal_embs = emb.embed_texts(det["normal"])
    abnormal_embs = emb.embed_texts(det["abnormal"])
    scale = emb.logit_scale
    scores = text_anomaly_scores(frame_embs, normal_embs, abnormal_embs, logit_scale=scale)

    typing = prompts["event_typing"]
    label_names = list(typing["labels"])
    label_embs = emb.embed_texts([typing["template"].format(label=v) for v in typing["labels"].values()])
    sims = frame_embs @ label_embs.T

    print(f"\nmodel={emb.model_id}  dtype={args.dtype}  device={emb.device}")
    print(f"{len(files)} ảnh trong {dt:.2f}s  ->  {len(files) / dt:.2f} ảnh/s")
    print(f"logit_scale thật = {scale:.2f}   dim = {emb.embedding_dim}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"VRAM đỉnh = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    except Exception:  # pragma: no cover
        pass

    print(f"\n{'ảnh':<20} {'text_score':>10}  hạng 1 (cos)      hạng 2")
    print("-" * 72)
    for f, s, row in zip(files, scores, sims):
        order = np.argsort(-row)[:2]
        print(f"{f.name:<20} {s:>10.4f}  {label_names[order[0]]:<16} {label_names[order[1]]}")

    normal = [i for i, f in enumerate(files) if f.name.startswith("duong_pho")]
    abnormal = [i for i, f in enumerate(files) if not f.name.startswith("duong_pho")]
    if normal and abnormal:
        print(f"\ntext_score TB  bình thường = {scores[normal].mean():.4f}")
        print(f"text_score TB  bất thường  = {scores[abnormal].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
