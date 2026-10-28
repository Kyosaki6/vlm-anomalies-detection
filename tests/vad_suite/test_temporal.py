import numpy as np
import pytest

from vad.temporal.ops import (candidate_windows, ema, expand_to_frames, median_smooth, robust_normalize,
                              window_topk_pool)


def test_robust_normalize_centers_and_resists_outlier():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    z = robust_normalize(x)
    assert np.median(z) == pytest.approx(0.0)
    assert z[-1] > 1e3 or z[-1] > 10  # outlier không làm lệch mốc median


def test_robust_normalize_empty_and_constant():
    assert robust_normalize(np.array([])).size == 0
    assert np.isfinite(robust_normalize(np.full(5, 3.0))).all()  # MAD=0 không chia cho 0


def test_ema_alpha_one_is_identity_and_bounds_checked():
    x = np.array([1.0, 5.0, 2.0])
    assert ema(x, 1.0).tolist() == x.tolist()
    with pytest.raises(ValueError):
        ema(x, 0.0)
    with pytest.raises(ValueError):
        ema(x, 1.5)


def test_median_smooth_removes_single_spike():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    assert median_smooth(x, 3).tolist() == [0.0] * 5
    with pytest.raises(ValueError):
        median_smooth(x, 4)


def test_topk_pool_picks_largest_in_window():
    x = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    out = window_topk_pool(x, window=3, k=1)
    assert out.tolist() == [0.0, 10.0, 10.0, 10.0, 0.0]


def test_functions_do_not_mutate_input():
    x = np.array([3.0, 1.0, 2.0, 8.0, 1.0])
    before = x.copy()
    for fn in (robust_normalize, lambda a: ema(a, 0.5), lambda a: median_smooth(a, 3),
               lambda a: window_topk_pool(a, 3, 2)):
        fn(x)
    assert x.tolist() == before.tolist()


def test_candidate_windows_basic_and_min_len():
    s = np.array([0, 0, 5, 5, 5, 0, 0, 5, 0, 0], dtype=float)
    wins = candidate_windows(s, sample_fps=1.0, threshold=1.0, video_id="v", min_len_s=2.0, merge_gap_s=0.0)
    assert [(w.start_s, w.end_s) for w in wins] == [(2.0, 5.0)]   # đoạn 1 mẫu (idx 7) bị loại do < 2s
    assert wins[0].score == 5.0 and wins[0].video_id == "v"


def test_candidate_windows_merges_close_runs():
    s = np.array([5, 5, 0, 5, 5], dtype=float)
    wins = candidate_windows(s, sample_fps=1.0, threshold=1.0, min_len_s=1.0, merge_gap_s=1.0)
    assert [(w.start_s, w.end_s) for w in wins] == [(0.0, 5.0)]


def test_candidate_windows_none_and_bad_fps():
    assert candidate_windows(np.zeros(5), 1.0, 1.0) == []
    with pytest.raises(ValueError):
        candidate_windows(np.zeros(5), 0.0, 1.0)


def test_expand_to_frames_alignment_30fps_over_1fps():
    scores = np.array([10.0, 20.0, 30.0])
    out = expand_to_frames(scores, sample_fps=1.0, fps=30.0, n_frames=90)
    assert out.shape == (90,)
    assert out[0] == 10.0 and out[29] == 10.0 and out[30] == 20.0 and out[89] == 30.0


def test_expand_to_frames_clips_beyond_last_sample_and_validates():
    out = expand_to_frames(np.array([1.0, 2.0]), 1.0, 30.0, 100)  # video dài hơn số mẫu
    assert out[-1] == 2.0
    assert expand_to_frames(np.array([1.0]), 1.0, 30.0, 0).size == 0
    with pytest.raises(ValueError):
        expand_to_frames(np.array([]), 1.0, 30.0, 10)
