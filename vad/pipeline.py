# -*- coding: utf-8 -*-
"""Tier-1 pipeline (Task TH2): embedding đã cache -> điểm text -> (memory) -> tầng thời gian -> cửa sổ ứng viên.

Một dataclass `PipelineConfig` giữ MỌI công tắc ablation, nên A0..A5 chỉ khác nhau ở config
(`ABLATIONS`). Không dùng GPU, không tải model: nhận mảng embedding + embedding của prompt.

    A0  CLIP B/32, 2 prompt baseline, điểm thô            (tái lập ZS-CLIP ~53)
    A1  SigLIP 2, 2 prompt baseline, điểm thô
    A2  + ensemble prompt
    A3  + chuẩn hoá theo video (robust z-score)
    A4  + top-k pooling + làm mượt thời gian
    A5  + normal memory (fuse)

Thứ tự thời gian (theo vad/temporal/ops.py): robust_normalize -> window_topk_pool -> ema | median_smooth.
Hàm thuần: cùng đầu vào + cùng config = cùng điểm (không random).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from vad.embed.cache import DEFAULT_CACHE_DIR, load_embeddings
from vad.interfaces import Embedder
from vad.results import RunResult, make_metadata
from vad.score.text import DEFAULT_LOGIT_SCALE, memory_anomaly_scores, text_anomaly_scores
from vad.temporal.ops import candidate_windows, ema, median_smooth, robust_normalize, window_topk_pool
from vad.types import Window

DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "configs" / "prompts.yaml"
SIGLIP2_ID = "google/siglip2-base-patch16-384"
CLIP_B32_ID = "openai/clip-vit-base-patch32"
PROMPT_SETS = ("baseline", "ensemble")
SMOOTHING = ("none", "ema", "median")


@dataclass(frozen=True)
class PipelineConfig:
    """Tất cả công tắc ablation. Mặc định = A1 (SigLIP 2, 2 prompt, điểm thô)."""

    model_id: str = SIGLIP2_ID
    sample_fps: float = 1.0
    prompt_set: str = "baseline"             # 'baseline' (1 normal + 1 abnormal) | 'ensemble' (4 + 4)
    logit_scale: float = DEFAULT_LOGIT_SCALE
    per_video_norm: bool = False             # robust z-score theo từng video
    pool_window_s: float = 0.0               # 0 = tắt top-k pooling
    pool_k: int = 3
    smoothing: str = "none"                  # 'none' | 'ema' | 'median'
    ema_alpha: float = 0.3
    median_kernel: int = 5
    use_memory: bool = False                 # fuse điểm memory (few-shot normal reference)
    memory_weight: float = 0.5               # trọng số memory trong fuse, (0, 1]
    threshold: float | None = None           # None = không trích cửa sổ
    min_len_s: float = 1.0
    merge_gap_s: float = 2.0

    def __post_init__(self) -> None:
        if self.prompt_set not in PROMPT_SETS:
            raise ValueError(f"prompt_set phải thuộc {PROMPT_SETS}, nhận {self.prompt_set!r}")
        if self.smoothing not in SMOOTHING:
            raise ValueError(f"smoothing phải thuộc {SMOOTHING}, nhận {self.smoothing!r}")
        if not self.sample_fps > 0:
            raise ValueError(f"sample_fps phải > 0, nhận {self.sample_fps}")
        if self.pool_window_s < 0 or self.pool_k < 1:
            raise ValueError("pool_window_s phải >= 0 và pool_k >= 1")
        if not 0.0 < self.memory_weight <= 1.0:
            raise ValueError(f"memory_weight phải trong (0, 1], nhận {self.memory_weight}")
        if self.use_memory and not self.per_video_norm:
            raise ValueError("use_memory=True cần per_video_norm=True: text (xác suất) và memory (1 - cosine) "
                             "khác thang đo, chỉ fuse được sau khi chuẩn hoá")


def _preset(model_id: str, **kw: Any) -> PipelineConfig:
    return PipelineConfig(model_id=model_id, **kw)


_A2 = dict(prompt_set="ensemble")
_A3 = dict(_A2, per_video_norm=True)
_A4 = dict(_A3, pool_window_s=5.0, pool_k=3, smoothing="ema", ema_alpha=0.3)
_A5 = dict(_A4, use_memory=True, memory_weight=0.5)

ABLATIONS: dict[str, PipelineConfig] = {
    "A0": _preset(CLIP_B32_ID),
    "A1": _preset(SIGLIP2_ID),
    "A2": _preset(SIGLIP2_ID, **_A2),
    "A3": _preset(SIGLIP2_ID, **_A3),
    "A4": _preset(SIGLIP2_ID, **_A4),
    "A5": _preset(SIGLIP2_ID, **_A5),
}


@dataclass(frozen=True)
class Tier1Result:
    video_id: str
    sample_fps: float
    text_scores: np.ndarray                  # điểm text thô (T,)
    memory_scores: np.ndarray | None         # điểm memory thô (T,) hoặc None
    scores: np.ndarray                       # điểm cuối sau tầng thời gian (T,)
    windows: tuple[Window, ...]

    def to_run_result(self, config: PipelineConfig, commit: str = "", **meta: Any) -> RunResult:
        """Đóng gói cho dashboard/tier-2 (chưa có verify/alerts: tier-2 chưa chạy)."""
        return RunResult(video_id=self.video_id, sample_fps=self.sample_fps, anomaly_scores=self.scores,
                         candidate_windows=self.windows,
                         metadata=make_metadata(model=config.model_id, commit=commit, tier="1", **meta))


# --- prompt -----------------------------------------------------------------------

def load_detection_prompts(prompt_set: str, path: str | Path = DEFAULT_PROMPTS) -> tuple[list[str], list[str]]:
    """(normal, abnormal) từ `detection.<prompt_set>` trong prompts.yaml. Chỉ prompt generic, không tên lớp."""
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"prompt_set phải thuộc {PROMPT_SETS}, nhận {prompt_set!r}")
    with open(path, encoding="utf-8-sig") as f:
        section = yaml.safe_load(f)["detection"][prompt_set]
    return list(section["normal"]), list(section["abnormal"])


def embed_detection_prompts(embedder: Embedder, config: PipelineConfig,
                            path: str | Path = DEFAULT_PROMPTS) -> tuple[np.ndarray, np.ndarray]:
    """Embedding (P, D) của prompt normal/abnormal cho `config.prompt_set`."""
    if embedder.model_id != config.model_id:
        raise ValueError(f"embedder là {embedder.model_id!r} nhưng config cần {config.model_id!r}")
    normal, abnormal = load_detection_prompts(config.prompt_set, path)
    return embedder.embed_texts(normal), embedder.embed_texts(abnormal)


# --- tier-1 -----------------------------------------------------------------------

def _validate_frames(frame_embs: np.ndarray) -> np.ndarray:
    f = np.asarray(frame_embs, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] == 0:
        raise ValueError(f"frame_embs phải là (T, D) với T >= 1, nhận shape {f.shape}")
    if not np.isfinite(f).all():
        raise ValueError("frame_embs có NaN/Inf")
    return f


def temporal_stage(scores: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """pool -> smooth (chuẩn hoá nằm ngoài, vì còn phải fuse memory trước khi pool)."""
    out = np.asarray(scores, dtype=np.float64)
    if config.pool_window_s > 0:
        window = max(1, int(round(config.pool_window_s * config.sample_fps)))
        out = window_topk_pool(out, window=window, k=config.pool_k)
    if config.smoothing == "ema":
        out = ema(out, alpha=config.ema_alpha)
    elif config.smoothing == "median":
        out = median_smooth(out, kernel=config.median_kernel)
    return out


def run_tier1(frame_embs: np.ndarray, normal_embs: np.ndarray, abnormal_embs: np.ndarray,
              config: PipelineConfig, video_id: str = "", normal_bank: np.ndarray | None = None) -> Tier1Result:
    """Embedding frame (T, D) + embedding prompt -> điểm cuối + cửa sổ ứng viên."""
    frames = _validate_frames(frame_embs)
    text = text_anomaly_scores(frames, normal_embs, abnormal_embs, logit_scale=config.logit_scale)

    memory: np.ndarray | None = None
    if config.use_memory:
        if normal_bank is None:
            raise ValueError("use_memory=True nhưng không truyền normal_bank (không được lặng lẽ bỏ qua)")
        memory = memory_anomaly_scores(frames, normal_bank)

    fused = text
    if config.per_video_norm:
        fused = robust_normalize(text)
        if memory is not None:
            w = config.memory_weight
            fused = (1.0 - w) * fused + w * robust_normalize(memory)

    final = temporal_stage(fused, config)
    windows: tuple[Window, ...] = ()
    if config.threshold is not None:
        windows = tuple(candidate_windows(final, config.sample_fps, config.threshold, video_id=video_id,
                                          min_len_s=config.min_len_s, merge_gap_s=config.merge_gap_s))
    return Tier1Result(video_id=video_id, sample_fps=config.sample_fps, text_scores=text,
                       memory_scores=memory, scores=final, windows=windows)


def score_cached_video(video_id: str, config: PipelineConfig, normal_embs: np.ndarray, abnormal_embs: np.ndarray,
                       cache_dir: str | Path = DEFAULT_CACHE_DIR,
                       normal_bank: np.ndarray | None = None) -> Tier1Result:
    """Đọc embedding từ cache theo (model, video, fps) của config rồi chạy `run_tier1`."""
    frames = load_embeddings(config.model_id, video_id, config.sample_fps, cache_dir)
    return run_tier1(frames, normal_embs, abnormal_embs, config, video_id=video_id, normal_bank=normal_bank)


def with_threshold(config: PipelineConfig, threshold: float | None) -> PipelineConfig:
    """Bản sao config với ngưỡng mới (ngưỡng calib trên `calib_normal` rồi gắn vào đây)."""
    return replace(config, threshold=threshold)


def build_memory_bank(frame_embs_per_video: Sequence[np.ndarray], max_frames: int | None = None) -> np.ndarray:
    """Gộp embedding các video `calib_normal` thành ngân hàng (N, D). Cắt đều nếu vượt `max_frames`."""
    if not frame_embs_per_video:
        raise ValueError("Cần ít nhất một video bình thường để dựng memory bank")
    bank = np.concatenate([_validate_frames(e) for e in frame_embs_per_video], axis=0)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames phải >= 1")
        if bank.shape[0] > max_frames:
            bank = bank[np.linspace(0, bank.shape[0] - 1, max_frames).round().astype(int)]
    return bank
