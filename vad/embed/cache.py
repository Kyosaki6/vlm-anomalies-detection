# -*- coding: utf-8 -*-
"""Cache embedding trên đĩa, mỗi video một file .npy — resumable, ghép shard = copy file vào cùng thư mục.

Tên file: <model_slug>__<video_id>__fps<sample_fps>.npy  (float16 trên đĩa để nhẹ).
Ghi nguyên tử (ghi file tạm rồi rename) để máy sập giữa chừng không để lại file hỏng.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "embeddings"


def slug(text: str) -> str:
    """Tên model/video -> phần an toàn cho tên file (thay ký tự lạ bằng '-')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


_slug = slug          # tên cũ: giữ để code ngoài repo (notebook, script cũ) không vỡ


def cache_path(model_id: str, video_id: str, sample_fps: float,
               cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{_slug(model_id)}__{_slug(video_id)}__fps{sample_fps:g}.npy"


def has_cached(model_id: str, video_id: str, sample_fps: float, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    return cache_path(model_id, video_id, sample_fps, cache_dir).is_file()


def save_embeddings(embs: np.ndarray, model_id: str, video_id: str, sample_fps: float,
                    cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    if embs.ndim != 2:
        raise ValueError(f"embs phải là (T, D), nhận shape {embs.shape}")
    path = cache_path(model_id, video_id, sample_fps, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, embs.astype(np.float16))
    os.replace(tmp, path)
    return path


def load_embeddings(model_id: str, video_id: str, sample_fps: float,
                    cache_dir: str | Path = DEFAULT_CACHE_DIR) -> np.ndarray:
    """-> (T, D) float32, đã chuẩn hoá lại L2 (fp16 làm lệch norm một chút)."""
    path = cache_path(model_id, video_id, sample_fps, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có cache: {path}")
    embs = np.load(path).astype(np.float32)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
