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


def test_verify_returns_result_and_passes_prompt():
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


def test_retry_is_not_a_noop():
    """Greedy + cùng prompt = y hệt lần 1; retry PHẢI đổi prompt (nudge) hoặc sampling params.

    Đây là test bắt bug 'retry no-op' trong review A2: nếu attempt 2 gửi y hệt attempt 1
    thì retry chỉ nhân đôi thời gian, không tăng xác suất parse được.
    """
    runner = make_runner(["hong lan 1", GOOD_JSON])
    VLMVerifier(runner=runner).verify(FRAMES, CONTEXT)
    assert len(runner.calls) == 2
    assert runner.calls[0] != runner.calls[1], (
        "attempt 2 giống hệt attempt 1 — retry là no-op với greedy decoding"
    )


def test_retry_nudge_accumulates_only_once():
    """max_attempts=3: prompt lần 3 chỉ có 1 nudge (không cộng dồn qua các lần)."""
    runner = make_runner(["hong"])
    with pytest.raises(RuntimeError):
        VLMVerifier(runner=runner, max_attempts=3).verify(FRAMES, CONTEXT)
    p1, p2, p3 = (u for _, u in runner.calls)
    assert p2.count("not valid JSON") == 1
    assert p3.count("not valid JSON") == 1          # không nhân lên


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


def test_quantization_is_validated():
    """Tham số lạ ('8bit') phải raise ngay, không âm thầm rơi xuống nhánh fp16."""
    with pytest.raises(ValueError, match="quantization"):
        VLMVerifier(runner=lambda *_: GOOD_JSON, quantization="8bit")


# ---------- đường _load_model (monkeypatch transformers: KHÔNG tải weight, KHÔNG cần GPU) ----------

class FakeEvalModel:
    """Model giả có .eval() — đúng những gì _load_model gọi sau from_pretrained."""

    def eval(self):
        self.eval_called = True
        return self


def _patch_family(monkeypatch, model_id_cls_name, device_available=True):
    """Patch transformers: AutoProcessor + class gia đình, trả (kwargs_gọi, fake_model)."""
    import transformers
    import torch as _torch

    captured: dict = {}
    fake_model = FakeEvalModel()

    def fake_processor(mid):
        captured["processor_mid"] = mid
        return _FakeProcMarker()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor)
    target_cls = getattr(transformers, model_id_cls_name)

    def fake_from_pretrained(mid, **kwargs):
        captured["model_cls"] = model_id_cls_name
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(target_cls, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(_torch.cuda, "is_available", lambda: device_available)
    return captured, fake_model


class _FakeProcMarker:
    pass


def test_load_model_quantization_none_falls_to_fp32_on_cpu(monkeypatch):
    """quantization='none' + không CUDA -> device_map='cpu', torch_dtype float32, không 4-bit."""
    captured, _ = _patch_family(monkeypatch, "Qwen2_5_VLForConditionalGeneration",
                                device_available=False)
    v = VLMVerifier(quantization="none")
    v._load_model()
    kw = captured["kwargs"]
    assert kw["device_map"] == "cpu"
    assert kw["torch_dtype"] is torch.float32
    assert "quantization_config" not in kw


def test_load_model_honours_explicit_device(monkeypatch):
    """device='cuda:1' phải nối thẳng vào device_map (bug half-wired trong review A2)."""
    captured, _ = _patch_family(monkeypatch, "Qwen2_5_VLForConditionalGeneration",
                                device_available=True)
    v = VLMVerifier(device="cuda:1", quantization="none")
    v._load_model()
    assert captured["kwargs"]["device_map"] == "cuda:1"


def test_load_model_4bit_without_cuda_fails_before_processor_download(monkeypatch):
    """4-bit trên máy không CUDA -> RuntimeError TRƯỚC khi tải processor (network fetch)."""
    import transformers
    import torch as _torch

    def _boom(mid):
        raise AssertionError("processor không được tải khi đã biết 4-bit sẽ fail")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", _boom)
    monkeypatch.setattr(_torch.cuda, "is_available", lambda: False)
    v = VLMVerifier(quantization="4bit")
    with pytest.raises(RuntimeError, match="4bit"):
        v._load_model()


def test_load_model_family_mapping_uses_explicit_class(monkeypatch):
    """Model id không khớp gia đình đã biết -> rơi xuống AutoModelForImageTextToText."""
    import transformers

    captured: dict = {}

    def fake_auto(mid, **kw):
        captured["cls"] = "auto"
        return FakeEvalModel()

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda mid: _FakeProcMarker()
    )
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_auto)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    v = VLMVerifier(model_id="unknown/org-model", quantization="none")
    v._load_model()
    assert captured["cls"] == "auto"


def test_parse_verifier_json_reads_string_false_as_false():
    """VLM thường trả 'is_anomaly': 'false' (chuỗi) — bool('false') == True là bẫy thật."""
    raw = '{"is_anomaly": "false", "event_type": "Normal", "caption": "c", "confidence": 0.9}'
    assert parse_verifier_json(raw).is_anomaly is False


def test_parse_verifier_json_rejects_garbage_is_anomaly():
    """is_anomaly là giá trị không phải bool/true/false -> ValueError, không đoán bừa."""
    raw = '{"is_anomaly": "maybe", "event_type": "Normal", "caption": "c", "confidence": 0.9}'
    with pytest.raises(ValueError, match="is_anomaly"):
        parse_verifier_json(raw)


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
