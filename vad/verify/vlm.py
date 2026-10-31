# -*- coding: utf-8 -*-
"""Tier-2: VLM sinh ngôn ngữ THẬT (track T2, Phước). Bắt buộc theo đề tài đã đăng ký.

Trạng thái:
  - [x] `parse_verifier_json` (hợp đồng với A3/C7; `is_anomaly` nhận cả bool lẫn chuỗi
        "true"/"false" — đầu ra VLM là text, `bool("false")` == True là bẫy thật).
  - [x] A2: `VLMVerifier.verify` thật — prompt từ configs/prompts.yaml (khoá `verifier`),
        retry 1 lần khi JSON hỏng (lần 2 thêm nudge "reply with JSON only" vào prompt —
        greedy + cùng prompt = y hệt lần 1, retry không được là no-op), rồi RuntimeError
        (KHÔNG bao giờ trả kết quả bịa).
  - [x] A1 đã chọn model (notes/vlm-selection.md, đo trên Colab T4):
        Qwen/Qwen2.5-VL-3B-Instruct, 4-bit nf4, 4 khung 320x240 native, greedy, max_new_tokens=128.
        Nạp model lazy (chỉ khi verify thật chạy); test dùng mock runner, không tải model.
  - Còn (A3): ghi mỗi Window ra file riêng để chạy lại được (scripts/run_verify.py).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from vad.types import EventType, VerifyResult

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"

# Kết quả A1redo (notes/vlm-selection.md): T4 median 4.45 s/cửa sổ, peak 2.93 GB, parse 16/16.
# Qwen2-VL-2B bị loại (echo template `true|false`, 2/16 parse); SmolVLM2 crash generate.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# runner(system, user, frames) -> text thô từ model. Test tiêm bản mock; bản thật lazy-load model.
Runner = Callable[[str, str, Sequence[Any]], str]


def parse_verifier_json(raw: str) -> VerifyResult:
    """Parse đầu ra VLM -> VerifyResult. Chịu được text thừa quanh JSON và ```json fence.

    Hỏng/thiếu khoá -> ValueError (người gọi quyết định retry hay đánh dấu lỗi; không đoán bừa).
    """
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Không thấy JSON trong đầu ra VLM: {raw[:80]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON hỏng: {e}") from e
    for key in ("is_anomaly", "event_type", "caption", "confidence"):
        if key not in data:
            raise ValueError(f"Thiếu khoá {key!r} trong JSON VLM")
    conf = float(data["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence ngoài [0, 1]: {conf}")
    return VerifyResult(
        is_anomaly=_as_bool(data["is_anomaly"]),
        event_type=EventType.from_label(str(data["event_type"])),
        caption=str(data["caption"]),
        confidence=conf,
    )


def _as_bool(value: Any) -> bool:
    """Ép `is_anomaly` thành bool AN TOÀN. Đầu ra VLM là text nên `false` thường đến dạng chuỗi.

    bool("false") == True — bẫy khiến mọi cửa sổ "bình thường" bị báo cáo là bất thường.
    Nhận bool; chuỗi "true"/"false" (không phân biệt hoa thường, có space); 0/1 số.
    Giá trị khác -> ValueError (không đoán bừa).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "false"):
            return v == "true"
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"is_anomaly phải là true/false (bool hoặc chuỗi), nhận {value!r}")


def load_verifier_prompt(path: str | Path = DEFAULT_PROMPTS) -> tuple[str, str]:
    """Đọc khối `verifier` từ prompts.yaml -> (system, user). Thiếu khoá -> ValueError."""
    with open(path, encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    block = (prompts or {}).get("verifier")
    if not isinstance(block, dict) or "system" not in block or "user" not in block:
        raise ValueError(f"prompts.yaml thiếu khối 'verifier' (cần 'system' + 'user'): {path}")
    return str(block["system"]), str(block["user"])


def _context_line(context: dict) -> str:
    """Gói ngữ cảnh cửa sổ (video, khoảng giây) thành 1 dòng cho prompt. Bỏ giá trị rỗng."""
    parts = []
    if context.get("video_id"):
        parts.append(f"video_id={context['video_id']}")
    if context.get("start_s") is not None and context.get("end_s") is not None:
        parts.append(f"window={context['start_s']}s..{context['end_s']}s")
    return " | ".join(parts)


class VLMVerifier:
    """Verifier tier-2 thật: frames + prompt `verifier` -> model sinh JSON -> VerifyResult.

    Tham số:
      model_id     HF id (mặc định = model đã đo ở A1; T4 dùng cùng model, code không đổi).
      device       'cuda' | 'cuda:1' | 'cpu' (None = tự chọn cuda nếu có). Nối thẳng vào
                   `device_map` của from_pretrained — không còn ignore như bản cũ.
      quantization '4bit' (nf4, cần CUDA; không có -> RuntimeError TRƯỚC khi tải processor)
                   hoặc 'none' (fp16 trên CUDA, fp32 trên CPU).
      runner       tiêm callable (system, user, frames) -> text. None = lazy-load model thật
                   (yêu cầu requirements-vlm.txt + GPU; test KHÔNG đi đường này).
      max_attempts số lần thử parse (mặc định 2 = 1 retry theo spec A2). Lỗi parse ở lần i
                   sẽ được nudge "reply with JSON only" cho lần i+1 (không phải retry no-op).

    Luôn raise khi không parse được — KHÔNG BAO GIỜ trả kết quả bịa.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        quantization: str = "4bit",
        runner: Runner | None = None,
        prompts_path: str | Path = DEFAULT_PROMPTS,
        max_attempts: int = 2,
        max_new_tokens: int = 128,
        model: Any = None,
        processor: Any = None,
    ):
        if max_attempts < 1:
            raise ValueError(f"max_attempts phải >= 1, nhận {max_attempts}")
        if quantization not in ("4bit", "none"):
            raise ValueError(f"quantization không hỗ trợ: {quantization!r} (chỉ '4bit' | 'none')")
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.prompts_path = Path(prompts_path)
        # Cache prompt 1 lần ở __init__: verify() gọi hàng trăm lần (A5), không đọc lại YAML mỗi cửa sổ.
        self._system, self._user_prompt = load_verifier_prompt(self.prompts_path)
        self.max_attempts = max_attempts
        self.max_new_tokens = max_new_tokens
        self._runner = runner
        self._model = model
        self._processor = processor

    # ---- runner thật (lazy; import nặng chỉ xảy ra ở đây, test không đụng tới) ----

    def _default_runner(self, system: str, user: str, frames: Sequence[Any]) -> str:
        if self._model is None or self._processor is None:
            self._load_model()
        import torch

        content: list[dict] = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": system},      # string, đúng đường đã đo ở A1
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        target = self.device or self._model.device
        inputs = self._processor(
            text=[text], images=list(frames), return_tensors="pt"
        ).to(target)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode([gen], skip_special_tokens=True)[0]

    def _load_model(self) -> None:
        """Nạp VLM lazy, cache lại. quantization='4bit' -> nf4; 'none' -> fp16/fp32.

        `self.device` (ví dụ 'cuda', 'cuda:1', 'cpu') được nối thẳng vào `device_map`;
        mặc định tự chọn cuda nếu có. Check CUDA/`device` TRƯỚC khi tải processor
        (processor là network fetch — fail sớm, không tải gì vô ích).
        """
        import torch
        import transformers

        target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.quantization == "4bit" and not target.startswith("cuda"):
            raise RuntimeError(
                f"quantization='4bit' cần CUDA (device hiện tại: {target!r}); "
                "truyền quantization='none' để chạy fp16/fp32 "
                "(VLMVerifier không tự chạy 4-bit trên CPU)"
            )

        # Class theo gia đình model — Auto* không luôn map đúng VLM (xem scripts/bench_vlm.py).
        by_family = {
            "qwen2-vl": transformers.Qwen2VLForConditionalGeneration,
            "qwen2.5-vl": transformers.Qwen2_5_VLForConditionalGeneration,
            "smolvlm": transformers.SmolVLMForConditionalGeneration,
        }
        cls = next(
            (m for name, m in by_family.items() if name in self.model_id.lower()),
            None,
        )
        if cls is None:
            cls = transformers.AutoModelForImageTextToText
        self._processor = self._processor or transformers.AutoProcessor.from_pretrained(self.model_id)
        if self._model is not None:
            return
        if self.quantization == "4bit":
            self._model = cls.from_pretrained(
                self.model_id,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map=target,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
        else:
            self._model = cls.from_pretrained(
                self.model_id,
                device_map=target,
                torch_dtype=torch.float16 if target.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
            )
        self._model.eval()

    # ---- contract ----

    def verify(self, frames: Sequence[Any], context: dict) -> VerifyResult:
        """frames: vài khung đại diện của Window. context: {'video_id', 'start_s', 'end_s', ...}.

        Parse hỏng ở lần thử cuối -> RuntimeError (retry 1 lần theo spec A2; không bịa kết quả).
        Retry KHÔNG phải no-op: greedy + cùng prompt sẽ cho ra y hệt lần 1, nên từ lần 2
        thêm nudge "chỉ trả JSON" vào user prompt (đổi đầu vào -> đổi xác suất ra output).
        """
        if not frames:
            raise ValueError("verify cần ít nhất 1 frame (danh sách rỗng)")
        if not isinstance(context, dict):
            raise TypeError(f"context phải là dict, nhận {type(context).__name__}")
        user = self._user_prompt
        line = _context_line(context)
        if line:
            user = f"{user}\nContext: {line}"

        run = self._runner or self._default_runner
        last_err: ValueError | None = None
        for attempt in range(1, self.max_attempts + 1):
            prompt = user
            if attempt > 1 and last_err is not None:
                # Nudge cho lần retry: yêu cầu JSON thuần, kèm lỗi lần trước để model tự sửa.
                # Ghép vào bản gốc `user` (không cộng dồn qua các lần) — greedy nên cùng prompt
                # sẽ cho ra y hệt lần 1, retry không đổi đầu vào là no-op.
                prompt = (
                    f"{user}\nYour previous reply was not valid JSON ({last_err}). "
                    "Reply with ONLY the JSON object, no other text."
                )
            raw = run(system=self._system, user=prompt, frames=frames)
            try:
                return parse_verifier_json(raw)
            except ValueError as e:
                last_err = e
                logger.warning("lần %d/%d parse hỏng: %s", attempt, self.max_attempts, e)
        raise RuntimeError(
            f"VLMVerifier không parse được JSON sau {self.max_attempts} lần "
            f"(model={self.model_id}): {last_err}"
        ) from last_err
