# -*- coding: utf-8 -*-
"""SigLIP 2 embedder THẬT (track T1/T2). Tái dùng logic từ search.py gốc, nhưng trả numpy theo `Embedder`.

TODO (Thịnh): nối vào search.load_model / embed_images / embed_texts.
Lưu ý GPU: Pascal (GTX 1060) fp16 rất chậm -> dùng dtype float32 trên đó; kiểm `torch.cuda.get_arch_list()` có sm_61.
Cache trên đĩa theo (model_id, video_id, sample_fps) — xem vad/embed/cache.py.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-384"


class SigLIPEmbedder:
    model_id = DEFAULT_MODEL_ID

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None, dtype: str = "float32"):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

    def embed_images(self, frames: Sequence[Any]) -> np.ndarray:
        raise NotImplementedError("SigLIPEmbedder.embed_images: chưa nối search.embed_images (việc của T1)")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("SigLIPEmbedder.embed_texts: chưa nối search.embed_texts (việc của T1)")
