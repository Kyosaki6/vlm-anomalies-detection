# -*- coding: utf-8 -*-
"""Test vad/pipeline.py (TH2): công tắc ablation, tính tái lập, đóng gói RunResult. FakeEmbedder, CPU."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vad import pipeline as pl
from vad.embed.cache import save_embeddings
from vad.fakes import FakeEmbedder
from vad.results import RunResult
from vad.temporal.ops import candidate_windows

DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture()
def prompts():
    rng = np.random.default_rng(0)
    return _unit(rng.standard_normal((3, DIM))), _unit(rng.standard_normal((3, DIM)))


@pytest.fixture()
def clip(prompts):
    """60 khung gần centroid 'normal' + đoạn 20..30 gần centroid 'abnormal'."""
    normal, abnormal = prompts
    rng = np.random.default_rng(1)
    c_n, c_a = _unit(normal.mean(0)), _unit(abnormal.mean(0))
    frames = np.stack([_unit(c_n + 0.5 * rng.standard_normal(DIM)) for _ in range(60)])
    for t in range(20, 30):
        frames[t] = _unit(c_a + 0.5 * rng.standard_normal(DIM))
    return frames, c_n


def _cfg(**kw) -> pl.PipelineConfig:
    # logit_scale nhỏ: với vector tổng hợp, scale 100 làm sigmoid bão hoà về 0/1 (MAD = 0, chuẩn hoá suy biến)
    kw.setdefault("logit_scale", 4.0)
    return pl.PipelineConfig(model_id="fake-embedder-v0", **kw)


def test_presets_cover_a0_to_a5_and_use_the_right_model():
    assert list(pl.ABLATIONS) == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert pl.ABLATIONS["A0"].model_id == pl.CLIP_B32_ID
    assert all(c.model_id == pl.SIGLIP2_ID for k, c in pl.ABLATIONS.items() if k != "A0")
    a = pl.ABLATIONS
    assert a["A1"].prompt_set == "baseline" and not a["A1"].per_video_norm
    assert a["A2"].prompt_set == "ensemble"
    assert a["A3"].per_video_norm and a["A3"].pool_window_s == 0 and a["A3"].smoothing == "none"
    assert a["A4"].pool_window_s > 0 and a["A4"].smoothing == "ema" and not a["A4"].use_memory
    assert a["A5"].use_memory


@pytest.mark.parametrize("kw", [dict(prompt_set="x"), dict(smoothing="gauss"), dict(sample_fps=0.0),
                                dict(pool_k=0), dict(pool_window_s=-1.0), dict(memory_weight=0.0),
                                dict(use_memory=True)])
def test_config_rejects_bad_options(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_prompts_in_yaml_are_generic_no_class_names():
    labels = {"abuse", "arrest", "arson", "assault", "burglary", "explosion", "fighting", "accident",
              "robbery", "shooting", "shoplifting", "stealing", "vandal"}
    for ps in pl.PROMPT_SETS:
        normal, abnormal = pl.load_detection_prompts(ps)
        text = " ".join(normal + abnormal).lower()
        assert not any(w in text for w in labels), f"label leak in {ps}"
    assert len(pl.load_detection_prompts("baseline")[1]) == 1
    assert len(pl.load_detection_prompts("ensemble")[1]) == 4
    with pytest.raises(ValueError):
        pl.load_detection_prompts("calib")


def test_embed_detection_prompts_uses_embedder_and_checks_model_id():
    emb = FakeEmbedder(dim=DIM)
    n, a = pl.embed_detection_prompts(emb, _cfg(prompt_set="ensemble"))
    assert n.shape == (4, DIM) and a.shape == (4, DIM)
    with pytest.raises(ValueError, match="config cần"):
        pl.embed_detection_prompts(emb, pl.ABLATIONS["A1"])


def test_raw_text_score_separates_anomaly_segment(prompts, clip):
    frames, _ = clip
    r = pl.run_tier1(frames, *prompts, _cfg())
    assert r.scores.shape == (60,) and r.memory_scores is None and r.windows == ()
    assert r.scores[20:30].min() > r.scores[:20].max()
    np.testing.assert_array_equal(r.scores, r.text_scores)       # A1: không biến đổi gì thêm


def test_same_config_same_scores(prompts, clip):
    frames, bank = clip
    cfg = _cfg(per_video_norm=True, pool_window_s=5.0, smoothing="ema", use_memory=True)
    a = pl.run_tier1(frames, *prompts, cfg, normal_bank=bank[None])
    b = pl.run_tier1(frames.copy(), *prompts, cfg, normal_bank=bank[None])
    np.testing.assert_array_equal(a.scores, b.scores)


def test_every_switch_changes_the_scores(prompts, clip):
    frames, bank = clip
    base = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True)).scores
    pooled = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, pool_window_s=5.0)).scores
    ema_ = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, smoothing="ema")).scores
    med = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, smoothing="median")).scores
    mem = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, use_memory=True),
                       normal_bank=bank[None]).scores
    raw = pl.run_tier1(frames, *prompts, _cfg()).scores
    for other in (pooled, ema_, med, mem, raw):
        assert not np.allclose(base, other)


def test_per_video_norm_is_robust_zscore(prompts, clip):
    frames, _ = clip
    r = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True))
    assert abs(np.median(r.scores)) < 1e-6


def test_non_default_smoothing_kernel_is_honoured(prompts, clip):
    frames, _ = clip
    k3 = pl.run_tier1(frames, *prompts, _cfg(smoothing="median", median_kernel=3)).scores
    k9 = pl.run_tier1(frames, *prompts, _cfg(smoothing="median", median_kernel=9)).scores
    assert not np.allclose(k3, k9)
    with pytest.raises(ValueError):
        pl.run_tier1(frames, *prompts, _cfg(smoothing="median", median_kernel=4))


def test_memory_without_bank_raises_not_silently_skipped(prompts, clip):
    frames, _ = clip
    with pytest.raises(ValueError, match="normal_bank"):
        pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, use_memory=True))


def test_memory_scores_reported_and_high_on_abnormal(prompts, clip):
    frames, bank = clip
    r = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, use_memory=True), normal_bank=bank[None])
    assert r.memory_scores is not None and r.memory_scores[20:30].mean() > r.memory_scores[:20].mean()


def test_candidate_windows_found_when_threshold_set(prompts, clip):
    frames, _ = clip
    cfg = _cfg(per_video_norm=True, threshold=3.0)
    r = pl.run_tier1(frames, *prompts, cfg, video_id="v1")
    assert len(r.windows) == 1
    w = r.windows[0]
    assert w.video_id == "v1" and 19 <= w.start_s <= 21 and 29 <= w.end_s <= 31
    assert r.windows == tuple(candidate_windows(r.scores, 1.0, 3.0, video_id="v1"))
    assert pl.with_threshold(cfg, None).threshold is None and cfg.threshold == 3.0


def test_sample_fps_scales_window_times(prompts, clip):
    frames, _ = clip
    r = pl.run_tier1(frames, *prompts, _cfg(per_video_norm=True, threshold=3.0, sample_fps=2.0, min_len_s=0.5))
    assert 9 <= r.windows[0].start_s <= 11            # mẫu 20 / 2 fps = 10 s


def test_hostile_frames_rejected(prompts):
    for bad in (np.empty((0, DIM)), np.ones(DIM), np.full((3, DIM), np.nan), np.full((3, DIM), np.inf)):
        with pytest.raises(ValueError):
            pl.run_tier1(bad, *prompts, _cfg())


def test_score_cached_video_roundtrip_via_cache(tmp_path, prompts, clip):
    frames, _ = clip
    cfg = _cfg(per_video_norm=True, sample_fps=1.0, threshold=3.0)
    save_embeddings(frames, cfg.model_id, "vidA", cfg.sample_fps, tmp_path)
    r = pl.score_cached_video("vidA", cfg, *prompts, cache_dir=tmp_path)
    assert r.video_id == "vidA" and len(r.scores) == 60 and len(r.windows) == 1
    with pytest.raises(FileNotFoundError):
        pl.score_cached_video("missing", cfg, *prompts, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):                      # cache khoá theo fps: fps khác = không thấy
        pl.score_cached_video("vidA", replace(cfg, sample_fps=2.0), *prompts, cache_dir=tmp_path)


def test_to_run_result_is_loadable_by_dashboard(tmp_path, prompts, clip):
    frames, _ = clip
    cfg = _cfg(per_video_norm=True, threshold=3.0)
    res = pl.run_tier1(frames, *prompts, cfg, video_id="v1").to_run_result(cfg, commit="abc")
    path = res.to_json(root=tmp_path)
    back = RunResult.from_json(path)
    assert back.video_id == "v1" and len(back.anomaly_scores) == 60
    assert back.candidate_windows == res.candidate_windows
    assert back.metadata["tier"] == "1" and back.metadata["model"] == "fake-embedder-v0"


def test_build_memory_bank_concat_and_subsample():
    a, b = np.ones((10, 4)), 2 * np.ones((6, 4))
    assert pl.build_memory_bank([a, b]).shape == (16, 4)
    assert pl.build_memory_bank([a, b], max_frames=5).shape == (5, 4)
    with pytest.raises(ValueError):
        pl.build_memory_bank([])
    with pytest.raises(ValueError):
        pl.build_memory_bank([a], max_frames=0)
