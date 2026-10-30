# -*- coding: utf-8 -*-
"""Test scripts/export_windows.py (TH6): cửa sổ cho tầng 2, ngưỡng lấy từ file calib, fail loud. CPU, không tải model."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import export_windows as ew  # noqa: E402
import run_benchmark as rb  # noqa: E402
from vad import pipeline as pl  # noqa: E402
from vad.embed.cache import save_embeddings  # noqa: E402
from vad.results import RunResult  # noqa: E402

DIM = 8
MODEL = "m"

def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(0)
    nd, ad = _unit(rng.standard_normal(DIM)), _unit(rng.standard_normal(DIM))
    prompts = (nd[None].astype(np.float32), ad[None].astype(np.float32))
    cache = tmp_path / "emb"
    specs = {"HIT1": range(10, 16), "HIT2": range(3, 6), "QUIET": ()}
    rows = []
    for vid, anom in specs.items():
        embs = np.stack([_unit(ad if t in anom else nd) + 0.01 * rng.standard_normal(DIM) for t in range(40)])
        save_embeddings(embs.astype(np.float32), MODEL, vid, 1.0, cache)
        rows.append({"video_id": vid, "n_frames": 1200, "fps": 30.0})
    cfg = pl.PipelineConfig(model_id=MODEL, sample_fps=1.0, logit_scale=4.0)
    return {"rows": rows, "cache": cache, "prompts": prompts, "cfg": cfg, "tmp": tmp_path}

def _export(w, thr=0.6, out=None, stats=None):
    return ew.export_windows(w["rows"], w["cfg"], w["prompts"], w["cache"], out or w["tmp"] / "res", thr, stats,
                             "TEST", commit="c0")

def test_only_videos_with_windows_are_written_and_index_matches(world):
    idx = _export(world)
    out = world["tmp"] / "res"
    assert idx["n_videos_total"] == 3 and idx["n_videos_with_windows"] == 2 and idx["n_windows"] == 2
    assert (out / "windows" / "HIT1.json").is_file() and (out / "windows" / "HIT2.json").is_file()
    assert not (out / "windows" / "QUIET.json").exists()
    on_disk = json.loads((out / "windows" / ew.INDEX_NAME).read_text(encoding="utf-8"))
    assert on_disk["n_windows"] == len(on_disk["windows"]) == 2
    assert {e["video_id"] for e in on_disk["windows"]} == {"HIT1", "HIT2"}

def test_window_files_roundtrip_through_results_schema(world):
    _export(world)
    run = RunResult.load_windows("HIT1", world["tmp"] / "res")
    assert len(run.candidate_windows) == 1 and run.verify_results == ()
    w = run.candidate_windows[0]
    assert 9 <= w.start_s <= 11 and 15 <= w.end_s <= 17
    assert run.metadata["ablation"] == "TEST" and run.metadata["threshold"] == 0.6 and run.metadata["fps"] == 30.0

def test_export_is_deterministic(world):
    a = _export(world, out=world["tmp"] / "a")
    b = _export(world, out=world["tmp"] / "b")
    assert a["windows"] == b["windows"]

def test_threshold_actually_controls_window_count(world):
    """Điểm thô nằm trong [0, 1]: ngưỡng 0 -> cả 3 video (QUIET cũng thành 1 cửa sổ), 0.6 -> 2, 1.01 -> 0."""
    counts = [_export(world, thr=t, out=world["tmp"] / f"t{i}")["n_windows"] for i, t in enumerate((0.0, 0.6, 1.01))]
    assert counts == [3, 2, 0]

def test_missing_cache_lists_every_missing_id(world):
    rows = world["rows"] + [{"video_id": "G1", "n_frames": 30, "fps": 30.0}, {"video_id": "G2", "n_frames": 30, "fps": 30.0}]
    with pytest.raises(rb.BenchmarkInputError, match="G1.*G2"):
        ew.export_windows(rows, world["cfg"], world["prompts"], world["cache"], world["tmp"] / "x", 0.5, None, "T")

# --- load_threshold_file -------------------------------------------------------------------

def _thr_file(tmp_path, ablation="A4c", config=None, thr=1.17, stats=None):
    cfg = config if config is not None else pl.ABLATIONS[ablation]
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"ablation": ablation, "config": asdict(cfg), "calib_stats": stats,
                             "calibration": {"threshold": thr}}), encoding="utf-8")
    return p

def test_load_threshold_file_ok_with_calib_stats(tmp_path):
    stats = {"text_median": 0.6, "text_mad": 0.17, "memory_median": None, "memory_mad": None}
    thr, cs = ew.load_threshold_file(_thr_file(tmp_path, stats=stats), "A4c", pl.ABLATIONS["A4c"])
    assert thr == 1.17 and cs == pl.CalibStats(0.6, 0.17)

def test_stale_threshold_file_with_changed_config_is_rejected(tmp_path):
    p = _thr_file(tmp_path)
    changed = replace(pl.ABLATIONS["A4c"], ema_alpha=0.9)          # cấu hình đổi sau khi calib
    with pytest.raises(ew.ExportError, match="calibrate_threshold"):
        ew.load_threshold_file(p, "A4c", changed)

def test_threshold_file_for_other_ablation_is_rejected(tmp_path):
    with pytest.raises(ew.ExportError, match="A3c"):
        ew.load_threshold_file(_thr_file(tmp_path, ablation="A3c"), "A4c", pl.ABLATIONS["A4c"])

@pytest.mark.parametrize("bad", [None, "x", float("inf")])
def test_hostile_threshold_value_is_rejected(tmp_path, bad):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"ablation": "A4c", "config": asdict(pl.ABLATIONS["A4c"]),
                             "calibration": {"threshold": bad}}).replace("Infinity", "Infinity"), encoding="utf-8")
    with pytest.raises(ew.ExportError):
        ew.load_threshold_file(p, "A4c", pl.ABLATIONS["A4c"])
