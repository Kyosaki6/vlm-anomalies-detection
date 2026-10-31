# -*- coding: utf-8 -*-
"""Event typer zero-shot (track T4, Nam): embedding của cửa sổ -> 1 trong 13 lớp UCF-Crime.

Chạy SAU khi tier-1/tier-2 đã nói "cửa sổ này bất thường". Đây là zero-shot CLASSIFICATION,
đo RIÊNG với phát hiện (plan §7): detection trả lời "có bất thường không", typer trả lời
"bất thường kiểu gì". Không train gì cả, chỉ so cosine giữa embedding khung hình và câu mô tả lớp.

Câu mô tả lớp nằm ở `configs/prompts.yaml` khoá `event_typing` — đây là chỗ DUY NHẤT được phép
nhắc tên lớp. Prompt phát hiện (tier-1) vẫn phải generic, nếu không là rò nhãn.

Chỉ numpy: ăn embedding đã cache -> chạy CPU, test bằng FakeEmbedder, không cần GPU/model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from vad.interfaces import Embedder
from vad.score.text import DEFAULT_LOGIT_SCALE, _l2
from vad.types import EventType

DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"

# Mô tả cho lớp "bình thường": chỉ dùng khi include_normal=True (baseline sanity, không phải 13 lớp UCF)
NORMAL_DESCRIPTION = "a normal scene where nothing unusual is happening"


@dataclass(frozen=True)
class TypingResult:
    """Kết quả phân loại một cửa sổ. `scores` giữ đủ phân phối để soi top-k khi báo cáo.

    LƯU Ý: `confidence` là softmax thô trên cosine, CHƯA calibrate. Khi báo cáo dùng top-1 accuracy,
    đừng trình bày confidence như xác suất đúng.
    """
    event_type: EventType
    confidence: float
    scores: dict[str, float]           # tên lớp (vd 'Fighting') -> xác suất
    top_k: tuple[tuple[EventType, float], ...]


def load_typing_prompts(path: str | Path = DEFAULT_PROMPTS) -> tuple[str, dict[str, str]]:
    """Đọc `event_typing` từ prompts.yaml -> (template, {tên lớp: mô tả})."""
    with open(path, encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    block = (prompts or {}).get("event_typing")
    if not isinstance(block, dict) or "template" not in block or "labels" not in block:
        raise ValueError(f"prompts.yaml thiếu khối 'event_typing' (cần 'template' + 'labels'): {path}")
    labels = block["labels"]
    if not isinstance(labels, dict) or not labels:
        raise ValueError("event_typing.labels phải là dict không rỗng")
    return str(block["template"]), {str(k): str(v) for k, v in labels.items()}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)                     # trừ max để khỏi tràn exp
    e = np.exp(z)
    return e / e.sum()


class ZeroShotEventTyper:
    """Gán loại sự kiện bằng cosine với prompt từng lớp. Không có gradient, không có tham số học.

    Tham số:
      embedder      object thoả `vad.interfaces.Embedder` (thật hoặc FakeEmbedder)
      labels        tuỳ chọn: chỉ dùng một tập con, vd ['Fighting', 'Shoplifting'].
                    Tên phải có trong prompts.yaml, sai -> ValueError (bắt lỗi gõ sai config).
      include_normal  thêm lớp Normal để kiểm tra typer có "biết" khi nào không có gì xảy ra.
      logit_scale   mặc định 100.0 cho khớp tier-1 (`vad/score/text.py`). SigLIP có logit_scale
                    học được; truyền số thật vào đây để so sánh công bằng (xem notes/).
    """

    def __init__(self, embedder: Embedder, prompts_path: str | Path = DEFAULT_PROMPTS,
                 labels: Sequence[str] | None = None, include_normal: bool = False,
                 logit_scale: float = DEFAULT_LOGIT_SCALE):
        template, all_labels = load_typing_prompts(prompts_path)

        if labels is not None:
            missing = [name for name in labels if name not in all_labels]
            if missing:
                raise ValueError(f"labels không có trong prompts.yaml: {missing}")
            picked = {name: all_labels[name] for name in labels}
        else:
            picked = dict(all_labels)

        if include_normal:
            picked["Normal"] = NORMAL_DESCRIPTION

        if not picked:
            raise ValueError("Không có lớp nào để phân loại")

        self.embedder = embedder
        self.logit_scale = float(logit_scale)
        # Tên lớp -> EventType. Gõ sai tên lớp trong yaml thì raise ngay, đừng để thành UNKNOWN âm thầm.
        self._types: dict[str, EventType] = {}
        for name in picked:
            et = EventType.from_label(name)
            if et is EventType.UNKNOWN:
                raise ValueError(f"Tên lớp {name!r} không map được sang EventType (xem vad/types.py)")
            self._types[name] = et

        self._names: list[str] = list(picked)
        texts = [template.format(label=desc) for desc in picked.values()]
        embs = np.asarray(embedder.embed_texts(texts), dtype=np.float64)
        if embs.ndim != 2 or embs.shape[0] != len(texts):
            raise ValueError(f"embedder trả shape lạ {embs.shape}, cần ({len(texts)}, D)")
        self._label_embs = _l2(embs)      # chuẩn hoá 1 lần, dùng lại cho mọi cửa sổ
        self.embedding_dim = int(self._label_embs.shape[1])

    @classmethod
    def from_label_embeddings(cls, names: Sequence[str], embeddings: np.ndarray,
                              prompts_path: str | Path = DEFAULT_PROMPTS,
                              logit_scale: float = DEFAULT_LOGIT_SCALE) -> "ZeroShotEventTyper":
        """Dựng typer từ vector nhãn ĐÃ CACHE — không cần embedder, không nạp text tower.

        Dùng cho đường build kết quả (C7b): 13 câu mô tả lớp được embed một lần rồi lưu npz, các lần
        build sau chỉ đọc lại. `names` vẫn phải có trong prompts.yaml — sai tên là lỗi cấu hình, không
        được âm thầm thành UNKNOWN. Caller chịu trách nhiệm kiểm vector cache khớp ĐÚNG chữ prompt
        hiện tại (xem `scripts/build_results.py`); ở đây chỉ kiểm tên + shape.
        """
        _, all_labels = load_typing_prompts(prompts_path)
        picked = [str(n) for n in names]
        if not picked:
            raise ValueError("names rỗng: không có lớp nào để phân loại")
        missing = [n for n in picked if n not in all_labels]
        if missing:
            raise ValueError(f"names không có trong prompts.yaml: {missing}")

        embs = np.asarray(embeddings, dtype=np.float64)
        if embs.ndim != 2 or embs.shape[0] != len(picked) or embs.shape[1] == 0:
            raise ValueError(f"embeddings shape {embs.shape} không khớp {len(picked)} lớp")

        obj = cls.__new__(cls)               # bỏ qua __init__: không gọi embed_texts
        obj.embedder = None
        obj.logit_scale = float(logit_scale)
        obj._names = picked
        obj._types = {}
        for name in picked:
            et = EventType.from_label(name)
            if et is EventType.UNKNOWN:
                raise ValueError(f"Tên lớp {name!r} không map được sang EventType (xem vad/types.py)")
            obj._types[name] = et
        obj._label_embs = _l2(embs)
        obj.embedding_dim = int(embs.shape[1])
        return obj

    def classify(self, frame_embs: np.ndarray, aggregate: str = "mean", top_k: int = 3) -> TypingResult:
        """frame_embs: (T, D) embedding các khung trong cửa sổ -> TypingResult.

        aggregate:
          'mean' — trung bình embedding rồi chuẩn hoá: "cả cửa sổ đang diễn ra gì" (mặc định).
          'max'  — mỗi lớp lấy cosine cao nhất trong các khung: "có khoảnh khắc nào giống lớp này".
        """
        f = np.asarray(frame_embs, dtype=np.float64)
        if f.size == 0:
            raise ValueError("frame_embs rỗng: cửa sổ không có khung nào")
        if f.ndim == 1:
            f = f[None, :]
        if f.ndim != 2 or f.shape[1] != self.embedding_dim:
            raise ValueError(f"frame_embs shape {f.shape} không khớp D={self.embedding_dim}")

        fn = _l2(f)
        if aggregate == "mean":
            sims = self._label_embs @ _l2(fn.mean(axis=0))
        elif aggregate == "max":
            sims = (fn @ self._label_embs.T).max(axis=0)
        else:
            raise ValueError(f"aggregate phải là 'mean' hoặc 'max', nhận {aggregate!r}")

        probs = _softmax(sims * self.logit_scale)
        order = np.argsort(-probs, kind="stable")[:max(1, top_k)]
        scores = {name: float(probs[i]) for i, name in enumerate(self._names)}
        best = int(order[0])
        return TypingResult(
            event_type=self._types[self._names[best]],
            confidence=float(probs[best]),
            scores=scores,
            top_k=tuple((self._types[self._names[i]], float(probs[i])) for i in order),
        )
