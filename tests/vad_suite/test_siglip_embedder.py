# -*- coding: utf-8 -*-
"""Test SigLIPEmbedder — KHÔNG tải model, KHÔNG cần GPU.

Cách làm: tiêm model/processor giả vào constructor. Nhờ vậy test được đúng những chỗ dễ sai
(batching, thứ tự, normalize, lấy .pooler_output, tham số padding) mà không cần mạng hay card đồ hoạ.

Test hợp đồng (isinstance với Protocol) nằm ở test_contracts.py, chạy chung Fake lẫn bản thật.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vad.embed.siglip import DEFAULT_BATCH_SIZE, TEXT_MAX_LENGTH, SigLIPEmbedder, _pooled


class FakeProcessor:
    """Giả lập AutoProcessor: ghi lại tham số gọi, sinh tensor tất định từ nội dung."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, images=None, text=None, **kwargs):
        self.calls.append({"images": images, "text": text, **kwargs})
        if images is not None:
            # Mã hoá chỉ số ảnh qua giá trị điểm ảnh đầu tiên -> kiểm được thứ tự sau batching
            vals = [float(img.getpixel((0, 0))[0]) for img in images]
            return {"pixel_values": torch.tensor(vals).reshape(len(vals), 1)}
        ids = torch.tensor([hash(t) % 1000 for t in text]).reshape(len(text), 1)
        return {"input_ids": ids}                      # transformers 5.x: KHÔNG có attention_mask


class FakeModel:
    """Giả lập SiglipModel: trả BaseModelOutputWithPooling(GIỐNG THẬT), không phải tensor."""

    def __init__(self, logit_scale: float | None = 4.726, return_tensor: bool = False):
        self._ls = None if logit_scale is None else torch.tensor([logit_scale])
        self.return_tensor = return_tensor
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    @property
    def logit_scale(self):
        return self._ls

    def get_image_features(self, pixel_values=None):
        emb = torch.cat([pixel_values, torch.ones_like(pixel_values)], dim=1)
        return emb if self.return_tensor else SimpleNamespace(pooler_output=emb)

    def get_text_features(self, input_ids=None, **kwargs):
        emb = torch.cat([input_ids, torch.ones_like(input_ids)], dim=1)
        return emb if self.return_tensor else SimpleNamespace(pooler_output=emb)


def make(model=None, processor=None, **kwargs):
    kwargs.setdefault("device", "cpu")
    return SigLIPEmbedder(model=model or FakeModel(), processor=processor or FakeProcessor(), **kwargs)


def img(value: int, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (2, 2), (value, 0, 0)) if mode == "RGB" else Image.new(mode, (2, 2), value)


# --- hình dạng / normalize ---------------------------------------------------------

def test_image_shape_dtype_and_l2_normalized():
    out = make().embed_images([img(1), img(2)])
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_text_shape_and_l2_normalized():
    out = make().embed_texts(["a normal scene", "people fighting"])
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_single_string_is_accepted_as_one_text():
    assert make().embed_texts("a normal scene").shape == (1, 2)


def test_embedding_dim_is_none_until_first_encode():
    e = make()
    assert e.embedding_dim is None
    e.embed_texts(["x"])
    assert e.embedding_dim == 2


def test_empty_input_raises_with_a_useful_message():
    e = make()
    with pytest.raises(ValueError, match="rỗng"):
        e.embed_images([])
    with pytest.raises(ValueError, match="rỗng"):
        e.embed_texts([])


# --- batching: chỗ dễ sai nhất ------------------------------------------------------

def test_batching_preserves_order_across_chunks():
    """17 ảnh, batch 8 -> 3 lượt gọi. Thứ tự phải giữ nguyên (lỗi kinh điển: trộn lô)."""
    proc = FakeProcessor()
    e = make(processor=proc, batch_size=8)
    out = e.embed_images([img(i) for i in range(17)])
    assert out.shape[0] == 17
    assert len(proc.calls) == 3                     # 8 + 8 + 1
    assert int(np.argmax(out[:, 0])) == 16          # ảnh cuối = giá trị lớn nhất
    assert int(np.argmin(out[:, 0])) == 0


def test_default_batch_size_is_8():
    assert DEFAULT_BATCH_SIZE == 8                  # đo trên GTX 1060: bs 8/16/32 đều ~14 img/s,
                                                    # bs=64 rơi xuống 1.8 vì tràn VRAM 3 GB


def test_load_is_callable_and_idempotent():
    """`load()` nạp sẵn model; gọi nhiều lần không được nạp lại hay crash."""
    e = make()
    assert e.load() is e
    assert e.load() is e


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        make(batch_size=0)


# --- đầu ra của transformers 5.x ---------------------------------------------------

def test_reads_pooler_output_from_output_object():
    """5.x trả BaseModelOutputWithPooling, KHÔNG phải tensor. Lấy nhầm -> sai âm thầm."""
    out = make(model=FakeModel(return_tensor=False)).embed_images([img(3)])
    assert out.shape == (1, 2)


def test_accepts_plain_tensor_output_too():
    out = make(model=FakeModel(return_tensor=True)).embed_images([img(3)])
    assert out.shape == (1, 2)


def test_pooled_rejects_unrecognised_output():
    with pytest.raises(RuntimeError, match="Không lấy được embedding"):
        _pooled(SimpleNamespace(last_hidden_state=torch.zeros(1, 4)), "image")


def test_pooled_prefers_pooler_output_over_last_hidden_state():
    out = SimpleNamespace(pooler_output=torch.ones(1, 2), last_hidden_state=torch.zeros(1, 2))
    assert torch.equal(_pooled(out, "image"), torch.ones(1, 2))


# --- tham số text: nếu quên padding=max_length thì embedding text lệch âm thầm ---------

def test_text_uses_siglip_padding_contract():
    proc = FakeProcessor()
    make(processor=proc).embed_texts(["a", "b"])
    call = proc.calls[0]
    assert call["padding"] == "max_length"
    assert call["max_length"] == TEXT_MAX_LENGTH == 64
    assert call["truncation"] is True


def test_missing_attention_mask_is_tolerated():
    """transformers 5.x không trả attention_mask; code phải chạy được mà không crash."""
    assert make().embed_texts(["x"]).shape == (1, 2)


def test_attention_mask_is_forwarded_when_present():
    class WithMask(FakeProcessor):
        def __call__(self, images=None, text=None, **kwargs):
            return {"input_ids": torch.ones(len(text), 1, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), 1, dtype=torch.long)}

    assert make(processor=WithMask()).embed_texts(["a", "b"]).shape == (2, 2)


# --- khung hình: chặn đúng lỗi BGR/RGBA/float --------------------------------------

def test_accepts_rgb_uint8_ndarray():
    assert make().embed_images([np.full((4, 4, 3), 7, np.uint8)]).shape == (1, 2)


def test_rejects_grayscale_and_rgba_arrays():
    e = make()
    with pytest.raises(ValueError, match="HxWx3"):
        e.embed_images([np.zeros((4, 4), np.uint8)])
    with pytest.raises(ValueError, match="HxWx3"):
        e.embed_images([np.zeros((4, 4, 4), np.uint8)])


def test_rejects_float_arrays():
    """float 0..1 sẽ bị hiểu thành 0..255 nếu lọt qua -> embedding rác. Chặn ngay."""
    with pytest.raises(ValueError, match="uint8"):
        make().embed_images([np.zeros((4, 4, 3), np.float32)])


def test_converts_non_rgb_pil_to_rgb():
    assert make().embed_images([img(5, mode="L")]).shape == (1, 2)


# --- logit_scale: số thật thay vì hardcode 100 -------------------------------------

def test_logit_scale_comes_from_the_model():
    assert make(model=FakeModel(logit_scale=4.726)).logit_scale == pytest.approx(112.85, abs=0.01)


def test_logit_scale_falls_back_when_model_has_none():
    assert make(model=FakeModel(logit_scale=None)).logit_scale == 100.0


# --- device / dtype ----------------------------------------------------------------

def test_float16_on_cpu_is_rejected():
    with pytest.raises(ValueError, match="float16"):
        _ = make(dtype="float16").device


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        make(dtype="bfloat17")


@pytest.mark.skipif(torch.cuda.is_available(), reason="chỉ có ý nghĩa khi máy KHÔNG có CUDA")
def test_cuda_requested_without_cuda_raises():
    with pytest.raises(RuntimeError, match="CUDA"):
        _ = make(device="cuda").device


def test_model_id_is_exposed_for_cache_keys():
    assert SigLIPEmbedder(model=FakeModel(), processor=FakeProcessor(),
                          device="cpu").model_id == "google/siglip2-base-patch16-384"


def test_custom_model_id_is_kept():
    e = SigLIPEmbedder("google/siglip2-small-patch16-256", model=FakeModel(),
                       processor=FakeProcessor(), device="cpu")
    assert "small" in e.model_id
