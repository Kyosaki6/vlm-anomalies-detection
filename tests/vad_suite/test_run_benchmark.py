# -*- coding: utf-8 -*-
"""Test scripts/run_benchmark.py (TH3) + AUC nhanh trong vad/eval/metrics.py. FakeEmbedder, CPU, không tải model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_benchmark as rb  # noqa: E402
from vad import pipeline as pl  # noqa: E402
from vad.datasets.ucf_crime import load_annotations  # noqa: E402
from vad.embed.cache import save_embeddings  # noqa: E402
from vad.eval.metrics import (fast_frame_level_auc, fast_video_bootstrap_auc_ci, frame_level_auc,  # noqa: E402
                              video_bootstrap_auc_ci)
from vad.fakes import FakeEmbedder  # noqa: E402

DIM = 8
CLIP = pl.CLIP_B32_ID
SIGLIP = pl.SIGLIP2_ID


# --- AUC nhanh == AUC chậm ----------------------------------------------------------------

def _random_videos(rng, n_videos=12, tied=False):
    s_pv, y_pv = [], []
    for _ in range(n_videos):
        n = int(rng.integers(20, 200))
        s = rng.random(n)
        if tied:
            s = np.round(s, 1)
        y = rng.random(n) < 0.3
        s_pv.append(s)
        y_pv.append(y)
    return s_pv, y_pv


@pytest.mark.parametrize("tied", [False, True])
def test_fast_auc_matches_slow(tied):
    s, y = _random_videos(np.random.default_rng(0), tied=tied)
    assert fast_frame_level_auc(s, y) == pytest.approx(frame_level_auc(s, y), abs=1e-12)


def test_fast_auc_constant_scores_is_half_and_single_class_is_nan():
    assert fast_frame_level_auc([np.zeros(10)], [np.arange(10) % 2 == 0]) == 0.5
    assert np.isnan(fast_frame_level_auc([np.arange(5.0)], [np.zeros(5, bool)]))


def test_fast_auc_rejects_nan_and_length_mismatch():
    with pytest.raises(ValueError, match="NaN"):
        fast_frame_level_auc([np.array([0.1, np.nan])], [np.array([True, False])])
    with pytest.raises(ValueError, match="lệch"):
        fast_frame_level_auc([np.arange(3.0)], [np.zeros(4, bool)])
    with pytest.raises(ValueError):
        fast_frame_level_auc([], [])


@pytest.mark.parametrize("tied", [False, True])
def test_fast_bootstrap_matches_slow_same_seed(tied):
    s, y = _random_videos(np.random.default_rng(1), n_videos=10, tied=tied)
    slow = video_bootstrap_auc_ci(s, y, n_bootstraps=40, seed=7)
    fast = fast_video_bootstrap_auc_ci(s, y, n_bootstraps=40, seed=7)
    assert fast == pytest.approx(slow, abs=1e-12)


# --- fixture benchmark nhỏ ----------------------------------------------------------------

@pytest.fixture()
def world(tmp_path):
    """3 video: V1 30 fps bất thường (giây 3..5), V2 25 fps (như Normal_Videos586) bình thường, V3 30 fps bình thường.

    Embedding được dựng để frame bất thường nằm gần prompt abnormal, frame thường gần prompt normal.
    """
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    normal_dir, abnormal_dir = unit(rng.standard_normal(DIM)), unit(rng.standard_normal(DIM))
    prompts = (np.stack([normal_dir]).astype(np.float32), np.stack([abnormal_dir]).astype(np.float32))

    specs = {  # video_id: (n_frames, fps, anomalous seconds)
        "V1": (300, 30.0, range(3, 6)),
        "Normal_Videos586_x264": (250, 25.0, ()),
        "V3": (240, 30.0, ()),
    }
    cache = tmp_path / "emb"
    rows = []
    for vid, (n_frames, fps, anom) in specs.items():
        n = int(np.ceil(n_frames / fps))
        embs = np.stack([unit(abnormal_dir if t in anom else normal_dir) + 0.05 * rng.standard_normal(DIM)
                         for t in range(n)])
        for model in (CLIP, SIGLIP):
            save_embeddings(embs.astype(np.float32), model, vid, 1.0, cache)
        rows.append({"video_id": vid, "n_frames": n_frames, "fps": fps})

    ann_file = tmp_path / "ann.txt"
    ann_file.write_text("V1.mp4  Abuse  90  179  -1  -1\n"
                        "Normal_Videos586_x264.mp4  Normal  -1  -1  -1  -1\n"
                        "V3.mp4  Normal  -1  -1  -1  -1\n", encoding="utf-8")
    return {"rows": rows, "cache": cache, "ann": load_annotations(ann_file), "prompts": prompts, "tmp": tmp_path}


def _run(world, config, prompts=None, **kw):
    return rb.run_ablation("A0", config, world["rows"], world["ann"], world["cache"],
                           prompts or world["prompts"], n_bootstraps=20, commit="testcommit", **kw)


def test_perfect_scorer_gives_auc_one(world):
    res = _run(world, pl.ABLATIONS["A0"])
    assert res["auc_micro"] == pytest.approx(1.0)
    assert (res["n_videos_anomalous"], res["n_videos_normal"]) == (1, 2)
    assert res["n_frames"] == 300 + 250 + 240
    assert res["config"]["model_id"] == CLIP and res["commit"] == "testcommit"


def test_inverted_scorer_gives_auc_zero(world):
    normal, abnormal = world["prompts"]
    res = _run(world, pl.ABLATIONS["A0"], prompts=(abnormal, normal))
    assert res["auc_micro"] == pytest.approx(0.0)


def test_constant_scorer_gives_auc_half(world):
    same = world["prompts"][0]
    res = _run(world, pl.ABLATIONS["A0"], prompts=(same, same))
    assert res["auc_micro"] == pytest.approx(0.5)


def test_non_default_ablation_a3_runs_with_per_video_norm(world):
    """A3 bật ensemble + per_video_norm: nhánh khác mặc định, hai prompt mỗi bên."""
    normal, abnormal = world["prompts"]
    two = (np.concatenate([normal, normal]), np.concatenate([abnormal, abnormal]))
    res = rb.run_ablation("A3", pl.ABLATIONS["A3"], world["rows"], world["ann"], world["cache"], two,
                          n_bootstraps=10, commit="x")
    assert res["config"]["per_video_norm"] is True
    assert res["auc_micro"] > 0.9


def test_25fps_video_gets_full_length_scores_and_labels(world):
    config = pl.ABLATIONS["A0"]
    s, y = rb.score_all(world["rows"], world["ann"], config, *world["prompts"], world["cache"])
    assert [len(a) for a in s] == [300, 250, 240] == [len(a) for a in y]
    assert y[0].sum() == 90 and y[1].sum() == 0


def test_label_shift_changes_alignment_but_keeps_length(world):
    config = pl.ABLATIONS["A0"]
    _, y0 = rb.score_all(world["rows"], world["ann"], config, *world["prompts"], world["cache"])
    _, y5 = rb.score_all(world["rows"], world["ann"], config, *world["prompts"], world["cache"], label_shift=5)
    assert len(y5[0]) == len(y0[0]) and y5[0].sum() == y0[0].sum()
    assert not np.array_equal(y0[0], y5[0])


def test_shift_labels_edges():
    y = np.array([0, 1, 1, 0, 0], dtype=bool)
    assert shift_eq(rb.shift_labels(y, 1), [0, 0, 1, 1, 0])
    assert shift_eq(rb.shift_labels(y, -1), [1, 1, 0, 0, 0])
    assert shift_eq(rb.shift_labels(y, 99), [0, 0, 0, 0, 0])
    assert shift_eq(rb.shift_labels(y, 0), list(y))


def shift_eq(a, b):
    return list(map(bool, a)) == list(map(bool, b))


# --- fail loud ----------------------------------------------------------------------------

def test_missing_cache_lists_every_missing_id(world):
    rows = world["rows"] + [{"video_id": "GHOST1", "n_frames": 100, "fps": 30.0},
                            {"video_id": "GHOST2", "n_frames": 100, "fps": 30.0}]
    with pytest.raises(rb.BenchmarkInputError, match="GHOST1.*GHOST2"):
        rb.check_inputs(rows, world["ann"], pl.ABLATIONS["A0"], world["cache"])


def test_video_missing_from_annotations_is_error_not_silent_normal(world):
    ann = {k: v for k, v in world["ann"].items() if k != "V1"}
    with pytest.raises(rb.BenchmarkInputError, match="V1"):
        rb.check_inputs(world["rows"], ann, pl.ABLATIONS["A0"], world["cache"])


def test_length_mismatch_between_cache_and_manifest_is_error(world):
    rows = [dict(r) for r in world["rows"]]
    rows[0]["n_frames"] = 3000
    with pytest.raises(rb.BenchmarkInputError, match="V1"):
        rb.score_all(rows, world["ann"], pl.ABLATIONS["A0"], *world["prompts"], world["cache"])


def test_length_within_tolerance_is_accepted(world):
    rows = [dict(r) for r in world["rows"]]
    rows[0]["n_frames"] = 300 + 30
    s, _ = rb.score_all(rows, world["ann"], pl.ABLATIONS["A0"], *world["prompts"], world["cache"])
    assert len(s[0]) == 330


def test_nan_embedding_is_error(world):
    bad = np.full((10, DIM), np.nan, dtype=np.float32)
    save_embeddings(bad, CLIP, "V3", 1.0, world["cache"])
    with pytest.raises(ValueError):
        rb.score_all(world["rows"], world["ann"], pl.ABLATIONS["A0"], *world["prompts"], world["cache"])


def test_manifest_without_fps_or_frames_is_error(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("video_id,path,dataset,split\nA,a.mp4,ucf_crime,test\n", encoding="utf-8")
    with pytest.raises(rb.BenchmarkInputError, match="n_frames/fps"):
        rb.read_benchmark_manifest(p)


def test_duplicate_video_ids_in_manifest_is_error(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("video_id,path,dataset,split,n_frames,fps\nA,a,u,test,10,30\nA,b,u,test,10,30\n", encoding="utf-8")
    with pytest.raises(rb.BenchmarkInputError, match="trùng"):
        rb.read_benchmark_manifest(p)


def test_select_rows_limit_is_evenly_spread_and_limit_zero_rejected():
    rows = [{"video_id": str(i)} for i in range(10)]
    picked = [r["video_id"] for r in rb.select_rows(rows, 4)]
    assert picked[0] == "0" and picked[-1] == "9" and len(picked) == 4
    assert rb.select_rows(rows, None) is rows and rb.select_rows(rows, 50) is rows
    with pytest.raises(rb.BenchmarkInputError):
        rb.select_rows(rows, 0)


# --- prompt cache -------------------------------------------------------------------------

class _CountingEmbedder(FakeEmbedder):
    def __init__(self, model_id):
        super().__init__(dim=DIM)
        self.model_id = model_id
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return super().embed_texts(texts)


def test_prompt_cache_second_call_does_not_build_embedder(tmp_path, capsys):
    made = []

    def factory():
        e = _CountingEmbedder(CLIP)
        made.append(e)
        return e

    cfg = pl.ABLATIONS["A0"]
    n1, a1 = rb.load_or_embed_prompts(cfg, cache_dir=tmp_path, embedder_factory=factory)
    assert len(made) == 1 and made[0].calls == 2

    def boom():
        raise AssertionError("embedder được tạo dù cache đã có")

    n2, a2 = rb.load_or_embed_prompts(cfg, cache_dir=tmp_path, embedder_factory=boom)
    assert "prompt cache hit" in capsys.readouterr().out
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_array_equal(a1, a2)


def test_prompt_cache_is_per_model_and_per_prompt_set(tmp_path):
    f = lambda m: (lambda: _CountingEmbedder(m))  # noqa: E731
    rb.load_or_embed_prompts(pl.ABLATIONS["A0"], cache_dir=tmp_path, embedder_factory=f(CLIP))
    rb.load_or_embed_prompts(pl.ABLATIONS["A1"], cache_dir=tmp_path, embedder_factory=f(SIGLIP))
    rb.load_or_embed_prompts(pl.ABLATIONS["A2"], cache_dir=tmp_path, embedder_factory=f(SIGLIP))
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_prompt_cache_stale_when_prompt_text_changes(tmp_path):
    cfg = pl.ABLATIONS["A0"]
    f = lambda: _CountingEmbedder(CLIP)  # noqa: E731
    rb.load_or_embed_prompts(cfg, cache_dir=tmp_path, embedder_factory=f)
    edited = tmp_path / "prompts.yaml"
    edited.write_text("detection:\n  baseline:\n    normal: ['a different normal']\n    abnormal: ['a different abnormal']\n",
                      encoding="utf-8")
    made = []
    rb.load_or_embed_prompts(cfg, prompts_path=edited, cache_dir=tmp_path,
                             embedder_factory=lambda: made.append(1) or _CountingEmbedder(CLIP))
    assert made, "cache cũ phải bị bỏ khi chữ prompt đổi"


def test_prompt_cache_rejects_wrong_embedder_model(tmp_path):
    with pytest.raises(rb.BenchmarkInputError, match="config cần"):
        rb.load_or_embed_prompts(pl.ABLATIONS["A0"], cache_dir=tmp_path,
                                 embedder_factory=lambda: _CountingEmbedder(SIGLIP))


# --- đầu ra -------------------------------------------------------------------------------

def test_result_json_round_trip_and_no_leftover_tmp(world):
    res = _run(world, pl.ABLATIONS["A0"])
    out = rb.write_json_atomic(rb.result_file(world["tmp"] / "out", "A0"), res)
    assert out.name == "ablation_A0.json" and not list(out.parent.glob("*.tmp"))
    back = json.loads(out.read_text(encoding="utf-8"))
    for key in ("ablation", "config", "n_videos", "n_videos_anomalous", "n_videos_normal", "n_frames", "auc_micro",
                "ci_lo", "ci_hi", "seed", "n_bootstraps", "commit", "timestamp", "per_video"):
        assert key in back, key
    assert back["auc_micro"] == pytest.approx(res["auc_micro"])
    assert [v["video_id"] for v in back["per_video"]] == ["V1", "Normal_Videos586_x264", "V3"]


def test_result_file_tag_keeps_diagnostics_apart(tmp_path):
    assert rb.result_file(tmp_path, "A0").name == "ablation_A0.json"
    assert rb.result_file(tmp_path, "A0", "shift16").name == "ablation_A0_shift16.json"


def test_cli_end_to_end_with_prompt_cache_and_limit(world, tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.csv"
    manifest.write_text("video_id,path,dataset,split,n_frames,fps\n" + "".join(
        f"{r['video_id']},x.mp4,ucf_crime,test,{r['n_frames']},{r['fps']:g}\n" for r in world["rows"]), encoding="utf-8")
    ann = tmp_path / "ann2.txt"
    ann.write_text("V1.mp4 Abuse 90 179 -1 -1\nNormal_Videos586_x264.mp4 Normal -1 -1 -1 -1\nV3.mp4 Normal -1 -1 -1 -1\n",
                   encoding="utf-8")
    pcache = tmp_path / "pc"
    pcache.mkdir()
    n, a = world["prompts"]
    np.savez(rb.prompt_cache_path(CLIP, "baseline", pcache), normal=n, abnormal=a,
             normal_texts=np.array(pl.load_detection_prompts("baseline")[0]),
             abnormal_texts=np.array(pl.load_detection_prompts("baseline")[1]), model_id=np.array(CLIP))
    out_dir = tmp_path / "res"
    argv = ["--manifest", str(manifest), "--annotations", str(ann), "--ablation", "A0", "--cache-dir", str(world["cache"]),
            "--prompt-cache-dir", str(pcache), "--out-dir", str(out_dir), "--n-bootstraps", "10"]
    assert rb.main(argv) == 0
    full = json.loads((out_dir / "ablation_A0.json").read_text(encoding="utf-8"))
    assert full["auc_micro"] == pytest.approx(1.0) and full["note"] == ""
    assert rb.main(argv + ["--limit", "2", "--out-dir", str(tmp_path / "lim")]) == 0
    lim = json.loads((tmp_path / "lim" / "ablation_A0.json").read_text(encoding="utf-8"))
    assert lim["n_videos"] == 2 and "chạy thử" in lim["note"]


def test_cli_diagnostic_run_requires_tag(world, tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("video_id,path,dataset,split,n_frames,fps\nV1,x,u,test,300,30\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rb.main(["--manifest", str(manifest), "--ablation", "A0", "--label-shift", "16"])


def test_cli_unknown_ablation_and_missing_ablation_rejected(tmp_path):
    with pytest.raises(SystemExit):
        rb.main(["--manifest", "m.csv", "--ablation", "A9"])
    with pytest.raises(SystemExit):
        rb.main(["--manifest", "m.csv"])

# --- A3c: chuẩn hoá bằng thống kê calib_normal -------------------------------------------------

def test_calib_norm_ablation_runs_and_records_stats(world):
    normal, abnormal = world["prompts"]
    two = (np.concatenate([normal, normal]), np.concatenate([abnormal, abnormal]))
    cfg = pl.ABLATIONS["A3c"]
    calib_rows = [r for r in world["rows"] if r["video_id"] != "V1"]            # chỉ video bình thường làm calib
    from vad.embed.cache import load_embeddings
    stats = pl.compute_calib_stats([load_embeddings(cfg.model_id, r["video_id"], 1.0, world["cache"])
                                    for r in calib_rows], *two, cfg)
    res = rb.run_ablation("A3c", cfg, world["rows"], world["ann"], world["cache"], two, n_bootstraps=10,
                          commit="x", calib_stats=stats)
    assert res["config"]["norm_ref"] == "calib"
    assert res["calib_stats"]["text_median"] == pytest.approx(stats.text_median)
    assert res["auc_micro"] > 0.9
    rb.write_json_atomic(world["tmp"] / "o.json", res)                          # allow_nan=False: không NaN lọt vào JSON

def test_calib_norm_ablation_without_stats_fails_loud(world):
    normal, abnormal = world["prompts"]
    two = (np.concatenate([normal, normal]), np.concatenate([abnormal, abnormal]))
    with pytest.raises(ValueError, match="calib_stats"):
        rb.run_ablation("A3c", pl.ABLATIONS["A3c"], world["rows"], world["ann"], world["cache"], two,
                        n_bootstraps=10, commit="x")

def test_build_calib_stats_lists_missing_cache(world):
    man = world["tmp"] / "calib.csv"
    man.write_text("video_id,path,dataset,split,n_frames,fps\nGHOST,x,ucf_crime,calib_normal,100,30\n", encoding="utf-8")
    with pytest.raises(rb.BenchmarkInputError, match="GHOST"):
        rb.build_calib_stats(man, pl.ABLATIONS["A3c"], world["cache"], world["prompts"], 1000)
