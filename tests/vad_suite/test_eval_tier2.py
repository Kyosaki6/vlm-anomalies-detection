# -*- coding: utf-8 -*-
"""Test scripts/eval_tier2.py (TH7b): cài đúng notes/a6-protocol.md §7. CPU, không tải model, không cần video thật."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import eval_tier2 as et  # noqa: E402
from vad.pipeline import PipelineConfig  # noqa: E402
from vad.results import RunResult, make_metadata  # noqa: E402
from vad.temporal.ops import candidate_windows, expand_to_frames  # noqa: E402
from vad.types import EventType, VerifyResult, Window  # noqa: E402

TAU = 1.0
CFG = replace(PipelineConfig(model_id="m"), sample_fps=1.0)
DUR = 100                                   # giây mỗi video, fps 30
FPS = 30.0

def _scores(spikes: dict[tuple[int, int], float]) -> np.ndarray:
    s = np.full(DUR, 0.1)
    for (a, b), v in spikes.items():
        s[a:b] = v
    return s

def _video(vid: str, spikes: dict[tuple[int, int], float], segments=(), cfg: PipelineConfig = CFG) -> et.VideoInput:
    s = _scores(spikes)
    wins = tuple(candidate_windows(s, cfg.sample_fps, TAU, video_id=vid, min_len_s=cfg.min_len_s,
                                   merge_gap_s=cfg.merge_gap_s))
    return et.VideoInput(vid, FPS, int(DUR * FPS), s, tuple(segments), wins)

@pytest.fixture()
def videos():
    # ANOM: window 10-20 trùng GT 12..18 s (TP), window 50-55 không trùng GT (FP trong video bất thường).
    anom = _video("ANOM", {(10, 20): 2.0, (50, 55): 1.5}, segments=((360, 540),))
    norm1 = _video("NORM1", {(30, 35): 1.8})              # 1 báo động sai
    norm2 = _video("NORM2", {})                           # im lặng
    assert [len(v.windows) for v in (anom, norm1, norm2)] == [2, 1, 0]
    return [anom, norm1, norm2]

def _labels(videos):
    out = []
    for v in videos:
        y = np.zeros(v.n_frames, dtype=bool)
        for a, b in v.segments:
            y[a:b + 1] = True
        out.append(y)
    return out

def _write_verify(root: Path, v: et.VideoInput, index: int, is_anomaly: bool, *, fake=False, model="vlm-x",
                  window: Window | None = None, meta_index: int | None = None, elapsed=4.0) -> Path:
    w = window or v.windows[index]
    res = VerifyResult(is_anomaly=is_anomaly, event_type=EventType.FIGHTING, caption="c", confidence=0.9)
    run = RunResult(video_id=v.video_id, sample_fps=1.0, anomaly_scores=(), candidate_windows=(w,),
                    verify_results=(res,),
                    metadata=make_metadata(model=model, window_index=index if meta_index is None else meta_index,
                                           fake=fake, n_chunks=1, elapsed_s=elapsed,
                                           chunks=[{"start_s": w.start_s, "end_s": w.end_s}],
                                           verifier_config={"prompt_hash": "abc12345"}))
    return run.to_json(et.verify_file(root, v.video_id, index))

def _verdicts(videos, rule):
    """rule(video_id, index) -> bool. Dựng thẳng dict verdict, không qua đĩa."""
    return {(v.video_id, i): et.WindowVerdict(is_anomaly=rule(v.video_id, i))
            for v in videos for i in range(len(v.windows))}

def _write_all(root: Path, videos, rule, **kw):
    for v in videos:
        for i in range(len(v.windows)):
            _write_verify(root, v, i, rule(v.video_id, i), **kw)

def _eval(videos, verdicts, **kw):
    return et.evaluate_a6(videos, verdicts, CFG, TAU, labels=_labels(videos), n_bootstraps=50, **kw)

# --- 1. identity: tất cả được xác nhận = A4c đúng từng bit --------------------------------------

def test_all_confirmed_is_exactly_a4c(videos):
    res = _eval(videos, _verdicts(videos, lambda *_: True))
    assert res["n_confirmed"] == 3 and res["n_rejected"] == 0
    assert res["operating_after"] == res["operating_before"]
    assert res["auc_micro"] == res["auc_micro_a4c"] and res["delta_auc"] == 0.0
    assert res["delta_auc_ci_lo"] == 0.0 and res["delta_auc_ci_hi"] == 0.0
    assert res["lost_true_positive_windows"] == []

def test_all_confirmed_scores_are_bit_identical(videos):
    for v in videos:
        assert np.array_equal(et.demote(v.scores, 1.0, []), v.scores)

def test_before_matches_calibrate_operating_point(videos):
    """Số 'trước' phải bằng `calibrate_threshold.operating_point` (cùng quy tắc chồng lấn GT, cùng FA/hour)."""
    import calibrate_threshold as ct
    from vad.datasets.ucf_crime import Annotation
    rows = [{"video_id": v.video_id, "n_frames": v.n_frames, "fps": v.fps} for v in videos]
    ann = {v.video_id: Annotation(v.video_id, "x", v.segments) for v in videos}
    op = ct.operating_point(rows, ann, [v.scores for v in videos], CFG, TAU)
    mine = _eval(videos, _verdicts(videos, lambda *_: True))["operating_before"]
    for key in ("fa_per_hour", "false_alarm_windows", "normal_hours", "incidents", "incidents_detected",
                "incident_recall", "mean_latency_s", "median_latency_s"):
        assert mine[key] == pytest.approx(op[key]), key
    assert mine["windows"] == op["windows_total"]

# --- 2. tất cả bị bác ------------------------------------------------------------------------------

def test_all_rejected_demotes_every_window_and_nothing_else(videos):
    for v in videos:
        after = et.demote(v.scores, 1.0, list(v.windows))
        inside = np.zeros(after.size, dtype=bool)
        for w in v.windows:
            inside[int(w.start_s):int(w.end_s)] = True
        assert (after[inside] == 0.0).all()
        assert np.array_equal(after[~inside], v.scores[~inside])
        assert candidate_windows(after, 1.0, TAU) == []

def test_all_rejected_kills_alerts_and_recall(videos):
    res = _eval(videos, _verdicts(videos, lambda *_: False))
    a = res["operating_after"]
    assert res["n_confirmed"] == 0 and res["n_rejected"] == 3
    assert a["windows"] == 0 and a["fa_per_hour"] == 0.0 and a["incidents_detected"] == 0
    assert a["window_precision"] is None                           # 0/0, không bịa số
    assert len(res["lost_true_positive_windows"]) == 1

# --- 3. đầu vào xấu phải nổ -----------------------------------------------------------------------------

def test_missing_verify_file_raises(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    et.verify_file(tmp_path, "NORM1", 0).unlink()
    with pytest.raises(et.Tier2EvalError, match="thiếu"):
        et.load_verdicts(tmp_path, videos)

def test_missing_directory_raises(videos, tmp_path):
    with pytest.raises(et.Tier2EvalError):
        et.load_verdicts(tmp_path / "nope", videos)

def test_roundtrip_through_disk_reads_the_verdicts(videos, tmp_path):
    rule = lambda vid, i: not (vid == "ANOM" and i == 1)  # noqa: E731
    _write_all(tmp_path, videos, rule)
    got = et.load_verdicts(tmp_path, videos)
    assert {k: x.is_anomaly for k, x in got.items()} == {("ANOM", 0): True, ("ANOM", 1): False, ("NORM1", 0): True}
    assert all(x.model == "vlm-x" and x.prompt_hash == "abc12345" and x.elapsed_s == 4.0 for x in got.values())

def test_fake_verifier_output_is_refused_unless_allowed(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True, fake=True)
    with pytest.raises(et.Tier2EvalError, match="GIẢ"):
        et.load_verdicts(tmp_path, videos)
    assert all(x.fake for x in et.load_verdicts(tmp_path, videos, allow_fake=True).values())

def test_window_that_differs_from_frozen_window_raises(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    v = videos[0]
    w = v.windows[0]
    _write_verify(tmp_path, v, 0, True, window=Window(w.video_id, w.start_s + 1, w.end_s, w.score))
    with pytest.raises(et.Tier2EvalError, match="khác window"):
        et.load_verdicts(tmp_path, videos)

def test_wrong_window_index_in_metadata_raises(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    _write_verify(tmp_path, videos[0], 1, True, meta_index=0)         # file window_0001 nhưng tự nhận là 0
    with pytest.raises(et.Tier2EvalError, match="window_index"):
        et.load_verdicts(tmp_path, videos)

def test_extra_verify_file_raises(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    _write_verify(tmp_path, videos[1], 5, True, window=videos[1].windows[0])
    with pytest.raises(et.Tier2EvalError, match="thừa"):
        et.load_verdicts(tmp_path, videos)

def test_verify_dir_for_unknown_video_raises(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    (tmp_path / "NORM2").mkdir()                                       # video không có window mà vẫn có thư mục
    with pytest.raises(et.Tier2EvalError, match="ngoài manifest"):
        et.load_verdicts(tmp_path, videos)

def test_mixed_verifiers_in_one_dir_raise(videos, tmp_path):
    _write_all(tmp_path, videos, lambda *_: True)
    _write_verify(tmp_path, videos[1], 0, True, model="other-vlm")
    with pytest.raises(et.Tier2EvalError, match="nhiều verifier"):
        et.load_verdicts(tmp_path, videos)

def test_evaluate_rejects_missing_or_unknown_verdicts(videos):
    ok = _verdicts(videos, lambda *_: True)
    short = dict(ok)
    short.pop(("NORM1", 0))
    with pytest.raises(et.Tier2EvalError, match="thiếu verdict"):
        _eval(videos, short)
    with pytest.raises(et.Tier2EvalError, match="không tồn tại"):
        _eval(videos, {**ok, ("NORM2", 0): et.WindowVerdict(True)})

@pytest.mark.parametrize("tau", [0.0, -1.0, float("nan"), float("inf")])
def test_tau_not_above_demoted_score_is_refused(videos, tau):
    with pytest.raises(et.Tier2EvalError, match="tau"):
        et.evaluate_a6(videos, _verdicts(videos, lambda *_: True), CFG, tau, labels=_labels(videos), n_bootstraps=5)

def test_frozen_windows_not_matching_scores_are_refused(videos):
    v = videos[0]
    bad = replace(v, windows=v.windows[:1])
    with pytest.raises(et.Tier2EvalError, match="đóng băng"):
        et.check_frozen_windows([bad], CFG, TAU)
    et.check_frozen_windows(videos, CFG, TAU)

# --- 4. ca trộn, tính tay -----------------------------------------------------------------------------------

def test_mixed_case_matches_hand_computation(videos):
    # giữ window TP của ANOM, bác window FP của ANOM, bác báo động sai của NORM1.
    res = _eval(videos, _verdicts(videos, lambda vid, i: vid == "ANOM" and i == 0))
    b, a = res["operating_before"], res["operating_after"]
    normal_h = 200 / 3600
    assert (b["windows"], b["true_positive_windows"], b["window_precision"]) == (3, 1, pytest.approx(1 / 3))
    assert (a["windows"], a["true_positive_windows"], a["window_precision"]) == (1, 1, 1.0)
    assert b["fa_per_hour"] == pytest.approx(1 / normal_h) and a["fa_per_hour"] == 0.0
    assert (b["incidents"], b["incidents_detected"], a["incidents_detected"]) == (1, 1, 1)
    assert a["incident_recall"] == b["incident_recall"] == 1.0
    assert res["lost_true_positive_windows"] == []
    assert res["rejected_split"] == {"rejected_in_anomalous_videos": 1, "rejected_in_normal_videos": 1,
                                     "rejected_overlapping_gt": 0, "rejected_not_overlapping_gt": 2}
    assert b["latency_n"] == a["latency_n"] == 1
    assert res["latency_common_incidents"]["n"] == 1

def test_rejecting_the_true_positive_is_counted_and_hurts(videos):
    res = _eval(videos, _verdicts(videos, lambda vid, i: not (vid == "ANOM" and i == 0)))
    assert res["lost_true_positive_windows"] == [{"video_id": "ANOM", "start_s": 10.0, "end_s": 20.0}]
    assert res["operating_after"]["incidents_detected"] == 0 and res["operating_before"]["incidents_detected"] == 1
    assert res["delta_auc"] < 0                                         # mất điểm cao đúng chỗ GT
    assert res["delta_auc_ci_lo"] <= res["delta_auc"] <= res["delta_auc_ci_hi"] + 1e-9
    assert res["delta_auc_ci_lo"] < 0

def test_rejecting_only_false_positives_does_not_lower_auc(videos):
    res = _eval(videos, _verdicts(videos, lambda vid, i: vid == "ANOM" and i == 0))
    assert res["delta_auc"] >= 0
    assert res["auc_micro"] >= res["auc_micro_a4c"]

def test_latency_on_common_incidents_ignores_dropped_incidents():
    """Hai sự cố, sự cố thứ hai bị mất sau tầng 2: latency 'chung' chỉ tính sự cố còn lại ở cả hai phía."""
    s = _scores({(10, 20): 2.0, (60, 70): 2.0})
    wins = tuple(candidate_windows(s, 1.0, TAU, video_id="A", min_len_s=1.0, merge_gap_s=2.0))
    v = et.VideoInput("A", FPS, int(DUR * FPS), s, ((300, 450), (1500, 1800)), wins)   # GT 10..15 s và 50..60 s
    n = et.VideoInput("N", FPS, int(DUR * FPS), _scores({}), (), ())
    res = _eval([v, n], _verdicts([v, n], lambda vid, i: i == 0))
    assert res["operating_before"]["incidents_detected"] == 2 and res["operating_after"]["incidents_detected"] == 1
    assert res["latency_common_incidents"]["n"] == 1

def test_paired_ci_is_nan_free_and_contains_delta_for_mixed_case(videos):
    res = _eval(videos, _verdicts(videos, lambda vid, i: vid == "ANOM" and i == 0))
    assert res["delta_auc_ci_lo"] <= res["delta_auc"] + 1e-9 <= res["delta_auc_ci_hi"] + 2e-9

# --- 5. tính chất: recall không bao giờ tăng ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_recall_never_increases_under_random_verdicts(videos, seed):
    rng = np.random.default_rng(seed)
    verdicts = {(v.video_id, i): et.WindowVerdict(bool(rng.integers(2))) for v in videos for i in range(len(v.windows))}
    res = _eval(videos, verdicts, paired=False)
    assert res["operating_after"]["incidents_detected"] <= res["operating_before"]["incidents_detected"]
    assert res["operating_after"]["windows"] <= res["operating_before"]["windows"]
    assert res["operating_after"]["fa_per_hour"] <= res["operating_before"]["fa_per_hour"]

# --- 6. span hạ điểm khớp span của candidate_windows ------------------------------------------------------------------

@pytest.mark.parametrize("sample_fps", [1.0, 2.0, 0.5])
def test_demote_span_matches_candidate_windows_span(sample_fps):
    n = 40
    s = np.full(n, 0.1)
    s[6:11] = 3.0
    s[n - 4:] = 3.0                                                     # window kết thúc đúng mẫu cuối
    wins = candidate_windows(s, sample_fps, TAU, video_id="V", min_len_s=0.0, merge_gap_s=0.0)
    assert len(wins) == 2
    out = et.demote(s, sample_fps, wins)
    assert (out[s >= TAU] == 0.0).all() and (out[s < TAU] == s[s < TAU]).all()
    assert candidate_windows(out, sample_fps, TAU, min_len_s=0.0, merge_gap_s=0.0) == []

def test_demote_does_not_mutate_input_and_clips_to_bounds():
    s = np.array([0.2, 3.0, 3.0])
    keep = s.copy()
    out = et.demote(s, 1.0, [Window("V", 1.0, 99.0, 3.0)])
    assert np.array_equal(s, keep) and out.tolist() == [0.2, 0.0, 0.0]

def test_demote_at_non_default_sample_fps_end_to_end():
    cfg = replace(CFG, sample_fps=2.0)
    s = np.full(200, 0.1)
    s[40:60] = 2.0                                                      # 20 mẫu @2 fps = 10 s
    w = tuple(candidate_windows(s, 2.0, TAU, video_id="V", min_len_s=1.0, merge_gap_s=2.0))
    assert (w[0].start_s, w[0].end_s) == (20.0, 30.0)
    v = et.VideoInput("V", FPS, int(DUR * FPS), s, ((600, 900),), w)
    n = et.VideoInput("N", FPS, int(DUR * FPS), np.full(200, 0.1), (), ())
    res = et.evaluate_a6([v, n], {("V", 0): et.WindowVerdict(False)}, cfg, TAU, labels=_labels([v, n]), n_bootstraps=5)
    assert res["n_rejected"] == 1 and res["operating_after"]["windows"] == 0
    frames = expand_to_frames(et.demote(s, 2.0, list(w)), 2.0, FPS, v.n_frames)
    assert frames[600:900].max() == 0.0

def test_no_normal_video_raises(videos):
    with pytest.raises(et.Tier2EvalError, match="bình thường"):
        _eval(videos[:1], _verdicts(videos[:1], lambda *_: True))

def test_verify_file_path_matches_run_verify_output_path(tmp_path):
    import run_verify as rv
    for vid, idx in (("Arrest001_x264", 0), ("Normal_Videos_935_x264", 54)):
        assert et.verify_file(tmp_path, vid, idx) == rv.output_path(tmp_path, vid, idx)

@pytest.mark.parametrize("bad", ["", "..", "../x", "a/b", "a\\b"])
def test_verify_file_rejects_path_like_video_ids(tmp_path, bad):
    with pytest.raises(et.Tier2EvalError):
        et.verify_file(tmp_path, bad, 0)

# --- chi phí + giao thức --------------------------------------------------------------------------------------------------------

def test_cost_summary_from_verdict_metadata():
    vs = {("A", 0): et.WindowVerdict(True, elapsed_s=4.0, n_chunks=1, chunk_seconds=10.0),
          ("A", 1): et.WindowVerdict(False, elapsed_s=12.0, n_chunks=3, chunk_seconds=90.0)}
    c = et.cost_summary(vs, n_videos=2, peak_vram_gb=2.93)
    assert c["chunk_calls"] == 4 and c["windows_chunked"] == 1 and c["pct_windows_chunked"] == 50.0
    assert c["video_seconds_shown"] == 100.0 and c["windows_per_video"] == 1.0
    assert c["s_per_window_median"] == 8.0 and c["s_per_window_total"] == 16.0 and c["peak_vram_gb"] == 2.93

def test_protocol_fingerprint_requires_the_note(tmp_path):
    with pytest.raises(et.Tier2EvalError, match="giao thức"):
        et.protocol_fingerprint(tmp_path / "missing.md")
    f = et.protocol_fingerprint()
    assert f["path"].endswith("a6-protocol.md") and len(f["sha256"]) == 64
