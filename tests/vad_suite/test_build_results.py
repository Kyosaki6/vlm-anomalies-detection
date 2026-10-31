# -*- coding: utf-8 -*-
"""Test scripts/build_results.py (C7): cache giả + prompt giả, CPU, không tải model, không có video.

Điều quan trọng nhất được canh ở đây: file sinh ra phải (a) dashboard đọc được, (b) tự khai là giả khi
ngưỡng chưa calibrate hoặc tier-2 là FakeVerifier, (c) KHÔNG bỏ qua video im lặng khi thiếu cache/lệch độ dài.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import build_results as br  # noqa: E402
import run_benchmark as rb  # noqa: E402
from app import data as dash  # noqa: E402
from vad import pipeline as pl  # noqa: E402
from vad.embed.cache import save_embeddings  # noqa: E402
from vad.fakes import FakeVerifier  # noqa: E402
from vad.graph.policy import GraphPolicy  # noqa: E402
from vad.results import RunResult  # noqa: E402
from vad.types import EventType, VerifyResult, Window  # noqa: E402

DIM = 8
SIGLIP = pl.SIGLIP2_ID
CLIP = pl.CLIP_B32_ID
ANOMALY_SEC = (10, 20)          # cửa sổ bất thường trong video 40 giây


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@pytest.fixture()
def world(tmp_path):
    """Hai video (một bất thường, một bình thường) + embedding giả tách sạch theo hướng prompt.

    Cache 1 fps: 40 mẫu = 40 giây. Manifest khai n_frames theo fps gốc nên độ dài vẫn khớp.
    """
    rng = np.random.default_rng(0)
    normal_dir = _unit(rng.normal(size=DIM))
    raw = rng.normal(size=DIM)
    abn_dir = _unit(raw - (raw @ normal_dir) * normal_dir)      # trực giao với normal_dir

    def frames(seconds: int, anomaly: tuple[int, int] | None) -> np.ndarray:
        out = []
        for t in range(seconds):
            center = abn_dir if anomaly and anomaly[0] <= t < anomaly[1] else normal_dir
            out.append(_unit(center + 0.05 * rng.normal(size=DIM)))
        return np.stack(out)

    cache = tmp_path / "cache"
    videos = {
        # (số giây cache, fps gốc, đoạn bất thường)
        "Abuse028_x264": (40, 30.0, ANOMALY_SEC),
        "Normal_Videos586_x264": (30, 25.0, None),          # 25 fps, giống video thật của UCF
    }
    for model in (SIGLIP, CLIP):
        for vid, (seconds, _, anomaly) in videos.items():
            save_embeddings(frames(seconds, anomaly), model, vid, 1.0, cache)

    manifest = tmp_path / "mini.csv"
    lines = ["video_id,path,dataset,split,n_frames,fps"]
    for vid, (seconds, fps, _) in videos.items():
        lines.append(f"{vid},{vid}.mp4,ucf_crime,test,{int(seconds * fps)},{fps:g}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "tmp": tmp_path, "cache": cache, "manifest": manifest,
        "normal_embs": normal_dir[None, :].astype(np.float32),
        "abnormal_embs": abn_dir[None, :].astype(np.float32),
    }


def _config(ablation: str = "A1", threshold: float | None = 0.5):
    return dataclasses.replace(pl.ABLATIONS[ablation], threshold=threshold)


def _rows(world, limit: int | None = None):
    return br.select_rows(br.read_benchmark_manifest(world["manifest"]), limit)


def _build(world, rows=None, limit: int | None = None, **kw):
    rows = rows if rows is not None else _rows(world, limit)
    kw.setdefault("config", _config())
    return br.build_from_manifest(rows, kw.pop("config"), world["normal_embs"], world["abnormal_embs"],
                                  cache_dir=world["cache"], out_dir=world["tmp"] / "results", **kw)


def _write_prompt_cache(config, cache_dir: Path, normal_embs: np.ndarray, abnormal_embs: np.ndarray) -> None:
    """Giả lập `cache/prompts/*.npz` để CLI không phải nạp model thật."""
    normal_txt, abnormal_txt = pl.load_detection_prompts(config.prompt_set)
    path = rb.prompt_cache_path(config.model_id, config.prompt_set, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, normal=normal_embs, abnormal=abnormal_embs, normal_texts=np.array(normal_txt),
             abnormal_texts=np.array(abnormal_txt), model_id=np.array(config.model_id))


# --- file sinh ra phải dashboard đọc được ------------------------------------------------

def test_build_writes_file_dashboard_can_load(world):
    res = _build(world, limit=1)[0]

    loaded = dash.load_run(res.video_id, world["tmp"] / "results")     # đúng đường dashboard đi
    assert loaded.video_id == res.video_id
    assert len(loaded.anomaly_scores) == 40 and loaded.sample_fps == 1.0
    assert len(loaded.candidate_windows) == 1
    w = loaded.candidate_windows[0]
    assert (w.start_s, w.end_s) == pytest.approx(ANOMALY_SEC)
    assert loaded.verify_results == () and loaded.alerts == ()          # chưa có tier-2 -> không bịa
    assert loaded.metadata["tier"] == "1"
    assert dash.is_placeholder(loaded)                                  # ngưỡng giả định -> có banner
    assert "NGƯỠNG GIẢ THUYẾT" in dash.run_note(loaded)


def test_windows_file_is_written_for_tier2_reruns(world):
    _build(world, limit=1)
    win = RunResult.load_windows("Abuse028_x264", world["tmp"] / "results")
    assert len(win.candidate_windows) == 1


# --- tier-2: fake tự khai là fake; verify-dir ghép đúng cặp -------------------------------

def test_fake_verifier_produces_alerts_and_declares_itself_fake(world):
    res = _build(world, limit=1, verifier=FakeVerifier(EventType.FIGHTING, 0.9))[0]

    assert len(res.verify_results) == len(res.candidate_windows) == 1
    assert len(res.alerts) == 1
    assert res.alerts[0].event_type is EventType.FIGHTING
    assert res.alerts[0].severity == "medium"                    # Fighting = medium trong policy.yaml
    assert res.alerts[0].path[0] == "camera:Abuse028_x264"       # một video = một camera
    assert res.metadata["tier"] == "1+2"
    assert "TIER-2 GIẢ" in res.metadata["note"]


def test_zone_and_night_hour_raise_severity(world):
    res = _build(world, limit=1, verifier=FakeVerifier(EventType.FIGHTING, 0.9),
                 zone="school_gate", hour=23)[0]
    alert = res.alerts[0]
    assert alert.severity == "critical"                          # medium +1 (zone) +1 (ban đêm)
    assert "zone:school_gate(+1)" in alert.path
    assert "time_band:night(+1)" in alert.path


def test_verify_dir_roundtrip_and_pair_mismatch(world):
    _build(world, limit=1)                                        # tạo cửa sổ trước
    vdir = world["tmp"] / "tier2"
    vdir.mkdir()
    good = {"schema_version": 1, "video_id": "Abuse028_x264", "sample_fps": 1.0,
            "candidate_windows": [{"video_id": "Abuse028_x264", "start_s": 10.0, "end_s": 20.0, "score": 0.9}],
            "verify_results": [{"is_anomaly": True, "event_type": "RoadAccidents",
                                "caption": "a car crash", "confidence": 0.8}]}
    (vdir / "Abuse028_x264.json").write_text(json.dumps(good), encoding="utf-8")

    res = _build(world, limit=1, verify_dir=vdir)[0]
    assert res.alerts[0].event_type is EventType.ROAD_ACCIDENT
    assert res.metadata["tier2"] == "json"

    # file tự nhất quán (2 cửa sổ + 2 kết quả) nhưng dựng bằng ngưỡng KHÁC -> ghép bừa = sai
    other = {**good, "candidate_windows": good["candidate_windows"] * 2,
             "verify_results": good["verify_results"] * 2}
    (vdir / "Abuse028_x264.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(br.BuildResultsError, match="lệch cặp"):
        _build(world, limit=1, verify_dir=vdir)

    # file tự mâu thuẫn (2 kết quả, 1 cửa sổ) -> lỗi ngay ở hợp đồng
    (vdir / "Abuse028_x264.json").write_text(json.dumps({**good, "verify_results": good["verify_results"] * 2}),
                                             encoding="utf-8")
    with pytest.raises(br.BuildResultsError, match="không đọc được tier-2"):
        _build(world, limit=1, verify_dir=vdir)


def test_verify_dir_same_count_shifted_windows_raises(world):
    """Số lượng khớp nhưng cửa sổ đã dịch (+100 s) -> phải lỗi, không được gán nhãn cho cửa sổ khác.

    Đây là lỗ mà review PR #20 bắt được: kiểm `len(verify) == len(windows)` thôi thì file tier-2 của
    ngưỡng/ablation khác (hoặc cửa sổ đã sửa tay) vẫn lọt, và caption VLM của cửa sổ cũ dán sang cửa sổ mới.
    """
    _build(world, limit=1)                                        # tạo results/windows/<id>.json (1 cửa sổ)
    saved = json.loads((world["tmp"] / "results" / "windows" / "Abuse028_x264.json").read_text(encoding="utf-8"))
    assert len(saved["candidate_windows"]) == 1

    shifted = json.loads(json.dumps(saved))
    for w in shifted["candidate_windows"]:
        w["start_s"] += 100.0
        w["end_s"] += 100.0
    shifted["verify_results"] = [{"is_anomaly": True, "event_type": "RoadAccidents",
                                  "caption": "a car crash", "confidence": 0.8}]
    vdir = world["tmp"] / "tier2_shifted"
    vdir.mkdir()
    (vdir / "Abuse028_x264.json").write_text(json.dumps(shifted), encoding="utf-8")

    with pytest.raises(br.BuildResultsError, match="cửa sổ"):
        _build(world, limit=1, verify_dir=vdir)


def test_verify_dir_empty_verify_with_windows_raises(world):
    """File windows của lần chạy tier-1-only (có cửa sổ, 0 kết quả) không được nhận làm tier-2.

    Trước đây script chạy "thành công" và ghi metadata `tier2="json"` dù không có kết quả tier-2 nào —
    metadata nói dối. Phải lỗi, để dashboard không hiện file không cảnh báo mà không rõ lý do.
    """
    _build(world, limit=1)                                        # windows/<id>.json: 1 cửa sổ, 0 verify
    vdir = world["tmp"] / "tier2_empty"
    vdir.mkdir()
    (vdir / "Abuse028_x264.json").write_text(
        (world["tmp"] / "results" / "windows" / "Abuse028_x264.json").read_text(encoding="utf-8"),
        encoding="utf-8")

    with pytest.raises(br.BuildResultsError, match="không có kết quả nào"):
        _build(world, limit=1, verify_dir=vdir)


def test_verify_dir_missing_file_is_an_error(world):
    with pytest.raises(br.BuildResultsError, match="Chưa có file tier-2"):
        _build(world, limit=1, verify_dir=world["tmp"] / "nope")


# --- không bỏ qua video im lặng ---------------------------------------------------------

def test_missing_cache_fails_and_lists_every_video(world):
    (world["cache"] / f"google-siglip2-base-patch16-384__Normal_Videos586_x264__fps1.npy").unlink()
    with pytest.raises(br.BuildResultsError, match="Normal_Videos586_x264"):
        _build(world)
    assert not (world["tmp"] / "results").exists() or not list((world["tmp"] / "results").glob("*.json"))


def test_length_mismatch_between_cache_and_manifest_fails(world):
    manifest = world["tmp"] / "wrong.csv"
    manifest.write_text("video_id,path,dataset,split,n_frames,fps\nAbuse028_x264,a.mp4,ucf_crime,test,300,30\n",
                        encoding="utf-8")
    rows = br.read_benchmark_manifest(manifest)                   # 300 frame @30fps = 10 s, cache có 40 mẫu
    with pytest.raises(br.BuildResultsError, match="cần ~10"):
        _build(world, rows=rows)


def test_manifest_without_fps_is_rejected(world):
    manifest = world["tmp"] / "nofps.csv"
    manifest.write_text("video_id,path,dataset,split\nAbuse028_x264,a.mp4,ucf_crime,test\n", encoding="utf-8")
    with pytest.raises(Exception):
        br.read_benchmark_manifest(manifest)


def test_hostile_video_id_never_escapes_results_dir(world):
    evil = "../escape"
    rows = [{"video_id": evil, "n_frames": 1200, "fps": 30.0}]
    save_embeddings(np.tile(world["normal_embs"], (40, 1)), SIGLIP, evil, 1.0, world["cache"])
    with pytest.raises(ValueError, match="video_id"):
        _build(world, rows=rows)
    assert not (world["tmp"] / "escape.json").exists()


# --- ngưỡng: nguồn phải rõ, thiếu threshold thì không có cửa sổ ---------------------------

def test_threshold_source_must_be_justified(world):
    with pytest.raises(br.BuildResultsError, match="chứng cứ"):
        br.threshold_note(0.5, "calib_normal")
    with pytest.raises(br.BuildResultsError, match="threshold_source"):
        br.threshold_note(0.5, "made_up")
    assert br.threshold_note(0.5, "calib_normal", "manifests/ucf_calib_normal.csv") == ""


def test_calibrated_threshold_keeps_provenance_without_fake_banner(world):
    """Ngưỡng thật -> `note` rỗng (dashboard không dán nhãn giả), nhưng xuất xứ vẫn nằm trong metadata."""
    res = _build(world, limit=1, threshold_source="calib_normal",
                 calib_manifest="manifests/ucf_calib_normal.csv")[0]
    assert res.metadata["note"] == ""
    assert res.metadata["threshold_source"] == "calib_normal"
    assert res.metadata["calib_manifest"] == "manifests/ucf_calib_normal.csv"
    assert not dash.is_placeholder(res)


def test_without_threshold_no_windows_and_no_note(world):
    res = _build(world, limit=1, config=_config(threshold=None))[0]
    assert res.candidate_windows == () and res.alerts == ()
    assert res.metadata["note"] == ""
    assert len(res.anomaly_scores) == 40                        # đường điểm vẫn đủ để dashboard vẽ


# --- ablation khác mặc định ------------------------------------------------------------

@pytest.mark.parametrize("ablation,threshold", [("A0", 0.5), ("A3", 1.0), ("A4", 1.0)])
def test_non_default_ablation_still_builds(world, ablation, threshold):
    res = _build(world, limit=2, config=_config(ablation, threshold))
    assert len(res) == 2
    assert res[0].metadata["ablation"] == ablation
    assert res[0].metadata["model"] == pl.ABLATIONS[ablation].model_id
    assert len(res[0].candidate_windows) >= 1


def test_normal_video_produces_no_window_or_alert(world):
    res = [r for r in _build(world, limit=2) if r.video_id.startswith("Normal")][0]
    assert res.candidate_windows == () and res.alerts == ()
    assert len(res.anomaly_scores) == 30                          # 30 giây @ 25 fps = 750 frame


# --- CLI ---------------------------------------------------------------------------------

def test_cli_dry_run_needs_no_model_and_writes_nothing(world):
    rc = br.main(["--manifest", str(world["manifest"]), "--ablation", "A1", "--threshold", "0.5",
                  "--cache-dir", str(world["cache"]), "--out-dir", str(world["tmp"] / "out"),
                  "--prompt-cache-dir", str(world["tmp"] / "no_prompts"), "--dry-run"])
    assert rc == 0
    assert not (world["tmp"] / "out").exists()


def test_cli_dry_run_reports_missing_cache(world):
    (world["cache"] / "google-siglip2-base-patch16-384__Abuse028_x264__fps1.npy").unlink()
    rc = br.main(["--manifest", str(world["manifest"]), "--cache-dir", str(world["cache"]),
                  "--prompt-cache-dir", str(world["tmp"] / "no_prompts"), "--dry-run"])
    assert rc == 1


def test_cli_end_to_end_with_cached_prompts(world, capsys):
    config = _config()
    prompts_cache = world["tmp"] / "prompts"
    _write_prompt_cache(config, prompts_cache, world["normal_embs"], world["abnormal_embs"])
    out = world["tmp"] / "out"

    rc = br.main(["--manifest", str(world["manifest"]), "--ablation", "A1", "--threshold", "0.5",
                  "--verifier", "fake", "--limit", "1", "--cache-dir", str(world["cache"]),
                  "--prompt-cache-dir", str(prompts_cache), "--out-dir", str(out)])
    assert rc == 0
    loaded = dash.load_run("Abuse028_x264", out)
    assert len(loaded.alerts) == 1
    assert dash.is_placeholder(loaded)
    assert "CẢNH BÁO" in capsys.readouterr().out                  # nói rõ có file placeholder


@pytest.mark.parametrize("extra", [
    ["--verifier", "fake", "--verify-dir", "x"],
    ["--verifier", "fake"],                                       # tier-2 mà không có --threshold
    ["--threshold", "0.5", "--threshold-source", "calib_normal"],  # thiếu --calib-manifest
    ["--threshold", "nan"],
    ["--ablation", "A9"],
])
def test_cli_rejects_inconsistent_flags(world, extra):
    with pytest.raises(SystemExit):
        br.main(["--manifest", str(world["manifest"]), *extra])


def test_summary_line_flags_placeholder(world):
    res = _build(world, limit=1)[0]
    assert "PLACEHOLDER" in br.summary_line(res)
    clean = RunResult(video_id="v", sample_fps=1.0, anomaly_scores=(0.1, 0.2),
                      candidate_windows=(Window("v", 0.0, 1.0, 0.2),), metadata={})
    assert "[ok]" in br.summary_line(clean)


def test_graph_policy_used_is_the_real_one(world):
    """Không truyền policy -> dùng configs/policy.yaml thật (đừng để test chạy trên policy giả)."""
    res = _build(world, limit=1, verifier=FakeVerifier(EventType.SHOOTING, 0.9))[0]
    assert res.alerts[0].severity == "critical"                   # Shooting = critical
    assert "police" in res.alerts[0].responders


def test_verify_windows_calls_with_replay_context(world):
    """`verify_windows` truyền ngữ cảnh cửa sổ (video_id, start, end) — FakeVerifier không cần frames."""
    calls = []

    class Spy:
        def verify(self, frames, context):
            calls.append(context)
            return VerifyResult(True, EventType.FIGHTING, "c", 0.9)

    out = br.verify_windows([Window("v", 1.0, 2.0, 0.9)], Spy(), "v")
    assert out[0].is_anomaly and calls == [{"video_id": "v", "start_s": 1.0, "end_s": 2.0, "tier": "2"}]


def test_ablation_a4c_with_calib_stats_builds(world):
    from vad.pipeline import CalibStats
    st = CalibStats(text_median=0.5, text_mad=0.1)
    res = _build(world, limit=2, config=_config("A4c", threshold=1.0), calib_stats=st)
    assert len(res) == 2
    assert res[0].metadata["ablation"] == "A4c"
    assert np.isfinite(res[0].anomaly_scores).all()
