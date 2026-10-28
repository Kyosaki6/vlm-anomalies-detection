# -*- coding: utf-8 -*-
"""SigLIP 2 embedder THẬT (track T1): ảnh/câu -> vector cùng không gian, đã L2-normalize.

Vì sao phải có file này: mọi con số của dự án (điểm tier-1, typer, AUC) đều là cosine giữa các vector
do đây sinh ra. Trước file này tồn tại, cả pipeline chạy trên `FakeEmbedder` (vector ngẫu nhiên) nên mọi
số đều vô nghĩa.

Ba cái bẫy của transformers 5.x đã đo thật trên GTX 1060 (29/10/2026) — ĐỪNG sửa nếu chưa đọc:
  1. `get_image_features` / `get_text_features` trả **`BaseModelOutputWithPooling`**, KHÔNG phải tensor.
     Lấy `.pooler_output` (shape (N, 768)). Thuộc tính `.image_embeds` / `.text_embeds` KHÔNG tồn tại —
     `getattr(out, "image_embeds", fallback)` rơi âm thầm xuống fallback, rất dễ lấy nhầm thứ.
  2. Text processor CHỈ trả `input_ids`, KHÔNG có `attention_mask`.
  3. Batch lớn có NGƯỠNG SỤP, không phải "càng to càng nhanh": trên GTX 1060 3 GB đo được
     bs=8 -> 13.4, bs=16 -> 14.2, bs=32 -> 14.0 img/s (VRAM 1.7–2.25 GB), nhưng bs=64 rơi xuống
     1.83 img/s vì tràn VRAM. Mặc định ở đây là 8 để chừa VRAM cho desktop/trình duyệt.

Tốc độ thực đo (GTX 1060, fp32, ảnh JPEG thật): ~14 img/s. Nghĩa là 290 video UCF-Crime test lấy mẫu
1 fps (~35k khung) mất khoảng 40 phút trên một card — không phải nhiều giờ.

fp32 trên Pascal: GTX 1060 chạy fp16 rất chậm, nên mặc định float32. `logit_scale` thật của SigLIP 2
base là 112.85 (= exp(4.726)), không phải 100 như code cũ hardcode.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-384"
TEXT_MAX_LENGTH = 64          # SigLIP 2 huấn luyện với padding="max_length", đúng 64 token
CLIP_TEXT_MAX_LENGTH = 77     # OpenAI CLIP: context length 77
DEFAULT_BATCH_SIZE = 8        # GTX 1060: bs 8/16/32 đều ~13-14 img/s, bs=64 sụp còn 1.8 -> chừa VRAM


def _pooled(output: Any, kind: str):
    """Lấy tensor embedding từ đầu ra của transformers.

    transformers 5.x trả `BaseModelOutputWithPooling`; bản cũ hơn trả thẳng tensor. Hàm này chấp nhận cả hai
    và **raise** nếu không tìm được, thay vì trả nhầm một thứ trông giống tensor.
    """
    import torch

    if isinstance(output, torch.Tensor):
        return output
    for attr in ("pooler_output", f"{kind}_embeds"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(
        f"Không lấy được embedding {kind} từ đầu ra {type(output).__name__}. "
        f"Thuộc tính có: {[a for a in dir(output) if not a.startswith('_')][:12]}"
    )


class SigLIPEmbedder:
    """Triển khai `vad.interfaces.Embedder`. Model chỉ nạp ở lần gọi đầu (lazy)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 dtype: str = "float32", batch_size: int = DEFAULT_BATCH_SIZE,
                 model: Any = None, processor: Any = None):
        if dtype not in ("float32", "float16", "float64"):
            raise ValueError(f"dtype không hỗ trợ: {dtype!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size phải >= 1, nhận {batch_size}")

        self.model_id = model_id
        self.dtype = dtype
        self.batch_size = batch_size
        self._device_arg = device
        self._model = model
        self._processor = processor
        self._torch_device: Any = None
        self._dim: int | None = None

    # --- nạp model ----------------------------------------------------------------

    @property
    def device(self) -> Any:
        return self._resolve_device()

    def _resolve_device(self):
        if self._torch_device is not None:
            return self._torch_device
        import torch

        if self._device_arg is None:
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = str(self._device_arg)
            if name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Yêu cầu device={name!r} nhưng CUDA không khả dụng trên máy này")
        dev = torch.device(name)
        if dev.type == "cpu" and self.dtype == "float16":
            raise ValueError("float16 trên CPU không dùng được — dùng 'float32'")
        self._torch_device = dev
        return dev

    def _ensure_loaded(self):
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        dev = self._resolve_device()
        tdt = getattr(torch, self.dtype)
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id, dtype=tdt).to(dev)
        except Exception as e:  # pragma: no cover - cần mạng/cache, test không đi qua đây
            raise RuntimeError(
                f"Không tải được model {self.model_id!r}: {e}. "
                "Kiểm tra cache HF (~/.cache/huggingface) hoặc kết nối mạng."
            ) from e
        if hasattr(model, "eval"):
            model.eval()
        self._model, self._processor = model, processor

    def load(self) -> "SigLIPEmbedder":
        """Nạp model ngay (thay vì đợi lần encode đầu). Dùng để đo tốc độ mà không dính thời gian load."""
        self._ensure_loaded()
        return self

    @property
    def embedding_dim(self) -> int | None:
        """Số chiều embedding; None nếu chưa từng encode gì."""
        return self._dim

    @property
    def logit_scale(self) -> float:
        """logit_scale THẬT của model (=112.85 với siglip2-base). Dùng cho softmax thay vì hardcode 100."""
        self._ensure_loaded()
        raw = getattr(self._model, "logit_scale", None)
        if raw is None:
            return 100.0
        try:
            return float(raw.exp().item())
        except AttributeError:  # pragma: no cover
            return float(raw)

    # --- encode -------------------------------------------------------------------

    def embed_images(self, frames: Sequence[Any]) -> np.ndarray:
        """frames: list PIL.Image hoặc ndarray HxWx3 RGB uint8 -> (N, D) float32, L2-normalize."""
        import torch
        from PIL import Image

        frames = list(frames)
        if not frames:
            raise ValueError("embed_images nhận danh sách rỗng: video/cửa sổ không có khung nào")

        self._ensure_loaded()
        images = [self._to_pil(f, Image) for f in frames]

        chunks = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size]
                inputs = self._processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._resolve_device())
                out = self._model.get_image_features(pixel_values=pixel_values)
                emb = _pooled(out, "image")
                emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
                chunks.append(emb.cpu().numpy())

        result = np.concatenate(chunks, axis=0).astype(np.float32)
        if result.shape[0] != len(images):  # pragma: no cover - model lỗi thì phải biết ngay
            raise RuntimeError(f"Model trả {result.shape[0]} vector cho {len(images)} ảnh")
        self._dim = int(result.shape[1])
        return result

    def embed_texts(self, texts: Sequence[str] | str) -> np.ndarray:
        """texts: list câu (hoặc 1 câu) -> (N, D) float32, L2-normalize, cùng không gian với ảnh."""
        import torch

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            raise ValueError("embed_texts nhận danh sách rỗng")

        self._ensure_loaded()
        inputs = self._processor(text=texts, return_tensors="pt", **self._text_kwargs())
        with torch.inference_mode():
            kwargs = {"input_ids": inputs["input_ids"].to(self._resolve_device())}
            if "attention_mask" in inputs:  # transformers 5.x KHÔNG trả khoá này; bản khác có
                kwargs["attention_mask"] = inputs["attention_mask"].to(self._resolve_device())
            out = self._model.get_text_features(**kwargs)
            emb = torch.nn.functional.normalize(_pooled(out, "text").float(), p=2, dim=-1)
        result = emb.cpu().numpy().astype(np.float32)
        self._dim = int(result.shape[1])
        return result

    # --- helpers ------------------------------------------------------------------

    @property
    def is_siglip(self) -> bool:
        return "siglip" in self.model_id.lower()

    def _text_kwargs(self) -> dict:
        """Tham số tokenizer theo họ model (B3: cùng một class phục vụ SigLIP 2 và OpenAI CLIP B/32).

        SigLIP/SigLIP 2: huấn luyện với padding="max_length" đúng 64 token -> bắt buộc, lệch là embedding text sai.
        CLIP (và model khác): context 77 token, pad tới câu dài nhất trong lô + cắt ở 77. `attention_mask`
        (nếu processor trả) vẫn được chuyển cho model ở `embed_texts`.
        """
        if self.is_siglip:
            return {"padding": "max_length", "max_length": TEXT_MAX_LENGTH, "truncation": True}
        return {"padding": True, "max_length": CLIP_TEXT_MAX_LENGTH, "truncation": True}

    @staticmethod
    def _to_pil(frame: Any, Image) -> Any:
        """ndarray HxWx3 RGB uint8 hoặc PIL.Image -> PIL RGB. Sai định dạng thì raise rõ ràng."""
        if isinstance(frame, Image.Image):
            return frame if frame.mode == "RGB" else frame.convert("RGB")
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Khung hình phải là HxWx3 RGB, nhận shape {arr.shape}. "
                "Nhớ `vad.io.video` đã đổi BGR->RGB; đừng truyền thẳng khung OpenCV."
            )
        if arr.dtype != np.uint8:
            raise ValueError(f"Khung hình phải uint8 0..255, nhận dtype {arr.dtype}")
        return Image.fromarray(arr, mode="RGB")
