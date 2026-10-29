# -*- coding: utf-8 -*-
"""Test VLMVerifier (A2) — dùng mock runner, KHÔNG tải model, KHÔNG cần GPU.

Đường thật (model 4-bit) chỉ chạy tay qua scripts/; test chỉ kiểm hợp đồng:
retry 1 lần, raise khi hỏng, prompt lấy từ prompts.yaml, không trả kết quả bịa.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from vad.verify.vlm import (
    DEFAULT_MODEL_ID,
    VLMVerifier,
    load_verifier_prompt,
    parse_verifier_json,
)

GOOD_JSON = (
    '{"is_anomaly": true, "event_type": "Fighting", '
    '"caption": "two people are fighting", "confidence": 0.85}'
)
FRAMES = [object(), object()]       # PIL ảnh thật không cần cho mock runner
CONTEXT = {"video_id": "Abuse028_x264", "start_s": 12.0, "end_s": 18.0}


class FakeInputs(dict):
    """Dict giống BatchFeature của transformers: có .to(device)."""

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    """Giả lập AutoProcessor: ghi messages, trả input_ids cố định, decode trả GOOD_JSON."""

    def __init__(self):
        self.messages = None
        self.images = None
        self.decoded = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "SYS\n<image><image>\nUSER"

    def __call__(self, text=None, images=None, return_tensors=None):
        assert text == ["SYS\n<image><image>\nUSER"]
        self.images = images
        return FakeInputs(input_ids=torch.tensor([[11, 12, 13]]))

    def batch_decode(self, seqs, skip_special_tokens=True):
        self.decoded = [[int(t) for t in s] for s in seqs]
        return [GOOD_JSON]


class FakeModel:
    """Giả lập model.generate: trả prompt + 2 token sinh, ghi lại kwargs."""

    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[11, 12, 13, 91, 92]])


def make_runner(outputs: list[str | Exception]):
    """Runner ghi lại lời gọi, trả outputs theo thứ tự (lặp phần tử cuối nếu hết)."""
    calls: list[tuple[str, str]] = []

    def run(system: str, user: str, frames) -> str:
        calls.append((system, user))
        out = outputs[min(len(calls) - 1, len(outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return out

    run.calls = calls            # type: ignore[attr-defined]
    return run


def test_verify_returns_result_and_passes_prompt(tmp_path: Path):
    """JSON hợp lệ từ lần 1 -> VerifyResult đúng; runner nhận system+user từ prompts.yaml."""
    runner = make_runner([GOOD_JSON])
    v = VLMVerifier(runner=runner)
    r = v.verify(FRAMES, CONTEXT)
    assert r.is_anomaly is True
    assert r.caption == "two people are fighting"
    assert r.confidence == pytest.approx(0.85)
    assert len(runner.calls) == 1
    system, user = runner.calls[0]
    yaml_system, yaml_user = load_verifier_prompt()
    assert system == yaml_system
    assert user.startswith(yaml_user[:60])          # prompt gốc không bị đổi
    assert "video_id=Abuse028_x264" in user         # context được nối vào


def test_verify_retries_once_on_bad_json():
    """Lần 1 output rác -> retry; lần 2 JSON tốt -> trả kết quả (không raise)."""
    runner = make_runner(["xin chao, khong phai JSON", GOOD_JSON])
    r = VLMVerifier(runner=runner).verify(FRAMES, CONTEXT)
    assert r.is_anomaly is True
    assert len(runner.calls) == 2


def test_verify_raises_after_two_failures():
    """2 lần hỏng -> RuntimeError, KHÔNG trả VerifyResult bịa (spec A2)."""
    runner = make_runner(["hong lan 1", "con hong nua"])
    with pytest.raises(RuntimeError, match="không parse được JSON sau 2 lần"):
        VLMVerifier(runner=runner).verify(FRAMES, CONTEXT)
    assert len(runner.calls) == 2


def test_verify_retry_count_configurable():
    """max_attempts=3 -> 3 lần gọi runner rồi mới raise (không hard-code 2)."""
    runner = make_runner(["hong"])
    with pytest.raises(RuntimeError, match="sau 3 lần"):
        VLMVerifier(runner=runner, max_attempts=3).verify(FRAMES, CONTEXT)
    assert len(runner.calls) == 3


def test_verify_rejects_empty_frames():
    """Đầu vào địch hại: danh sách frame rỗng -> ValueError ngay, không gọi runner."""
    runner = make_runner([GOOD_JSON])
    with pytest.raises(ValueError, match="ít nhất 1 frame"):
        VLMVerifier(runner=runner).verify([], CONTEXT)
    assert runner.calls == []


def test_verify_rejects_non_dict_context():
    """Đầu vào địch hại: context không phải dict -> TypeError rõ ràng, không gọi runner."""
    runner = make_runner([GOOD_JSON])
    with pytest.raises(TypeError, match="context phải là dict"):
        VLMVerifier(runner=runner).verify(FRAMES, ["not", "a", "dict"])  # type: ignore[arg-type]
    assert runner.calls == []


def test_verify_zero_attempts_rejected():
    with pytest.raises(ValueError, match="max_attempts"):
        VLMVerifier(runner=lambda *_: GOOD_JSON, max_attempts=0)


def test_prompt_file_without_verifier_block(tmp_path: Path):
    """prompts.yaml thiếu khối verifier -> ValueError (fail rõ, không im lặng)."""
    bad = tmp_path / "prompts.yaml"
    bad.write_text(yaml.safe_dump({"detection": {"baseline": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu khối 'verifier'"):
        VLMVerifier(runner=lambda *_: GOOD_JSON, prompts_path=bad).verify(FRAMES, CONTEXT)


def test_default_model_id_is_the_a1redo_choice():
    """A1redo đã chốt model; verifier mặc định phải là model đo được (notes/vlm-selection.md)."""
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_quantization_is_validated():
    """Tham số lạ ('8bit') phải raise ngay, không âm thầm rơi xuống nhánh fp16."""
    with pytest.raises(ValueError, match="quantization"):
        VLMVerifier(runner=lambda *_: GOOD_JSON, quantization="8bit")


def test_default_runner_uses_chat_template_and_crops_prompt_tokens():
    """Đường thật (không runner): system là string, inputs chuyển device, decode CHỈ phần sinh.

    Model/processor giả -> không tải transformers/model, nhưng vẫn kiểm được phần dễ sai:
    thứ tự message, tham số generate, và cắt bỏ prompt token trước khi decode.
    """
    proc, mdl = FakeProcessor(), FakeModel()
    v = VLMVerifier(processor=proc, model=mdl)
    r = v.verify(FRAMES, CONTEXT)

    assert r.is_anomaly is True
    yaml_system, _ = load_verifier_prompt()
    assert proc.messages[0] == {"role": "system", "content": yaml_system}
    assert proc.messages[1]["content"][0] == {"type": "image"}
    assert proc.messages[1]["content"][-1]["type"] == "text"
    assert mdl.kwargs["do_sample"] is False
    assert mdl.kwargs["max_new_tokens"] == 128
    assert proc.decoded == [[91, 92]]          # chỉ token sinh, không gồm 3 token prompt


def test_parse_still_rejects_bad_confidence():
    """Hành vi parse_verifier_json giữ nguyên (khớp file cũ): confidence ngoài [0,1] là lỗi."""
    raw = '{"is_anomaly": true, "event_type": "Fighting", "caption": "x", "confidence": 3.0}'
    with pytest.raises(ValueError, match="confidence"):
        parse_verifier_json(raw)
