import numpy as np
import pytest

from vad.embed.cache import cache_path, has_cached, load_embeddings, save_embeddings
from vad.eval.metrics import (detection_latency_s, false_alarms_per_hour, frame_level_auc,
                              video_bootstrap_auc_ci)
from vad.score.text import memory_anomaly_scores, prompt_centroid, text_anomaly_scores
from vad.verify.vlm import parse_verifier_json
from vad.types import EventType


# ---------- text / memory scoring ----------
def test_text_scores_separate_clear_cases():
    normal = np.array([[1.0, 0.0]])
    abnormal = np.array([[0.0, 1.0]])
    frames = np.array([[1.0, 0.05], [0.05, 1.0]])
    s = text_anomaly_scores(frames, normal, abnormal)
    assert s[0] < 0.01 and s[1] > 0.99


def test_text_scores_in_unit_interval_and_stable_for_extreme_margin():
    f = np.array([[1e3, 0.0], [0.0, 1e3]])
    s = text_anomaly_scores(f, np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), logit_scale=1e6)
    assert np.isfinite(s).all() and ((0 <= s) & (s <= 1)).all()


def test_prompt_centroid_is_unit_norm():
    c = prompt_centroid(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.linalg.norm(c) == pytest.approx(1.0)


def test_memory_scores_zero_when_frame_in_bank_and_high_when_far():
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    frames = np.array([[1.0, 0.0], [-1.0, 0.0]])
    s = memory_anomaly_scores(frames, bank)
    assert s[0] == pytest.approx(0.0, abs=1e-9) and s[1] > 0.9


def test_memory_scores_rejects_empty_bank():
    with pytest.raises(ValueError):
        memory_anomaly_scores(np.ones((2, 2)), np.empty((0, 2)))


# ---------- eval metrics ----------
def test_frame_level_auc_perfect_and_chance():
    y = [np.array([0, 0, 1, 1], bool)]
    assert frame_level_auc([np.array([0.1, 0.2, 0.8, 0.9])], y) == pytest.approx(1.0)
    assert frame_level_auc([np.array([0.5] * 4)], y) == pytest.approx(0.5)  # toàn điểm trùng = 0.5


def test_frame_level_auc_single_class_is_nan_and_length_mismatch_raises():
    assert np.isnan(frame_level_auc([np.array([0.1, 0.2])], [np.array([0, 0], bool)]))
    with pytest.raises(ValueError, match="lệch"):
        frame_level_auc([np.array([0.1, 0.2, 0.3])], [np.array([0, 1], bool)])
    with pytest.raises(ValueError):
        frame_level_auc([np.array([0.1])], [])


def test_video_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    scores, labels = [], []
    for _ in range(12):
        y = rng.random(40) < 0.3
        scores.append(rng.normal(0, 1, 40) + 1.5 * y)
        labels.append(y)
    est = frame_level_auc(scores, labels)
    lo, hi = video_bootstrap_auc_ci(scores, labels, n_bootstraps=200)
    assert lo <= est <= hi
    assert (lo, hi) == video_bootstrap_auc_ci(scores, labels, n_bootstraps=200)  # seed cố định


def test_video_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        video_bootstrap_auc_ci([], [])
    with pytest.raises(ValueError):
        video_bootstrap_auc_ci([np.zeros(2)], [np.zeros(2, bool)], ci=1.0)


def test_operational_metrics():
    assert false_alarms_per_hour(3, 1800.0) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        false_alarms_per_hour(1, 0)
    assert detection_latency_s(10.0, 14.0) == 4.0
    assert detection_latency_s(10.0, 5.0) == 0.0
    assert detection_latency_s(10.0, None) is None


# ---------- embedding cache ----------
def test_cache_roundtrip_renormalizes_and_is_resumable(tmp_path):
    embs = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    assert not has_cached("google/siglip2-base", "vid1", 1.0, tmp_path)
    save_embeddings(embs, "google/siglip2-base", "vid1", 1.0, tmp_path)
    assert has_cached("google/siglip2-base", "vid1", 1.0, tmp_path)
    back = load_embeddings("google/siglip2-base", "vid1", 1.0, tmp_path)
    assert np.linalg.norm(back, axis=1) == pytest.approx(np.ones(5), abs=1e-5)
    assert (back * embs).sum(1) == pytest.approx(np.ones(5), abs=1e-2)   # fp16 làm tròn
    assert not list(tmp_path.glob("*.tmp"))                               # không để file tạm


def test_cache_key_separates_model_and_fps_and_missing_raises(tmp_path):
    assert cache_path("a/b", "v", 1.0, tmp_path) != cache_path("a/c", "v", 1.0, tmp_path)
    assert cache_path("a/b", "v", 1.0, tmp_path) != cache_path("a/b", "v", 2.0, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_embeddings("a/b", "nope", 1.0, tmp_path)
    with pytest.raises(ValueError):
        save_embeddings(np.zeros(4), "a/b", "v", 1.0, tmp_path)


# ---------- VLM JSON parsing ----------
def test_parse_verifier_json_tolerates_fence_and_chatter():
    raw = 'Sure!\n```json\n{"is_anomaly": true, "event_type": "road accident", "caption": "c", "confidence": 0.8}\n```'
    r = parse_verifier_json(raw)
    assert r.is_anomaly and r.event_type == EventType.ROAD_ACCIDENT and r.confidence == 0.8


@pytest.mark.parametrize("bad", [
    "no json here",
    '{"is_anomaly": true}',
    '{"is_anomaly": true, "event_type": "Fighting", "caption": "c", "confidence": 1.5}',
    '{"is_anomaly": true, "event_type": "Fighting", "caption": "c", "confidence": }',
])
def test_parse_verifier_json_rejects_bad_output(bad):
    with pytest.raises(ValueError):
        parse_verifier_json(bad)
