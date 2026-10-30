# -*- coding: utf-8 -*-
"""Test Task A3: scripts/run_verify.py — FakeVerifier, video tự sinh, không GPU/model."""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_verify as rv  # noqa: E402
from vad.fakes import FakeVerifier  # noqa: E402
from vad.results import RunResult, make_metadata  # noqa: E402
from vad.types import EventType, VerifyResult, Window  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class CountingVerifier(FakeVerifier):
    """FakeVerifier + ghi lại số frame/context mỗi lần gọi."""

    def __init__(self):
        super().__init__(event_type=EventType.FIGHTING, confidence=0.88)
        self.calls: list[tuple[int, dict, list[tuple[int, ...]]]] = []

    def verify(self, frames, context):
        self.calls.append((len(frames), dict(context), [tuple(f.shape) for f in frames]))
        return super().verify(frames, context)


class BrightnessVerifier(CountingVerifier):
    """Ghi thêm độ sáng trung bình để kiểm tra đúng frame được cắt."""

    def __init__(self):
        super().__init__()
        self.means: list[list[float]] = []

    def verify(self, frames, context):
        self.means.append([float(np.asarray(f).mean()) for f in frames])
        return super().verify(frames, context)


@pytest.fixture()
def synthetic_video(tmp_path):
    """90 khung 30 fps = 3 giây; độ sáng tăng theo giây để kiểm tra đúng frame."""
    path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for i in range(90):
        writer.write(np.full((48, 64, 3), 40 + 60 * (i // 30), dtype=np.uint8))
    writer.release()
    return path


def write_manifest(tmp_path, video_id, video_path):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "path", "dataset", "split"])
        writer.writerow([video_id, Path(video_path).as_posix(), "ucf_crime", "test"])
    return path


def write_windows(tmp_path, video_id="v1", windows=None, verify_results=()):
    windows = windows or (
        Window(video_id, 0.2, 1.2, 0.9),
        Window(video_id, 1.5, 2.7, 0.8),
    )
    root = tmp_path / "results"
    run = RunResult(
        video_id=video_id,
        sample_fps=1.0,
        anomaly_scores=(),
        candidate_windows=tuple(windows),
        verify_results=tuple(verify_results),
        metadata=make_metadata(model="test"),
    )
    run.save_windows(root=root)
    return root / "windows" / f"{video_id}.json", tuple(windows)


# --- hàm thuần --------------------------------------------------------------------

def test_pick_frame_times_evenly_spaced_and_inside_window():
    times = rv.pick_frame_times(1.0, 3.0, 4)
    assert times == pytest.approx([1.25, 1.75, 2.25, 2.75])
    assert all(1.0 <= t <= 3.0 for t in times)

    with pytest.raises(ValueError, match="k phải"):
        rv.pick_frame_times(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="không hợp lệ"):
        rv.pick_frame_times(-1.0, 1.0, 4)
    with pytest.raises(ValueError, match="không hợp lệ"):
        rv.pick_frame_times(1.0, 1.0, 4)


def test_frame_indices_reject_too_short_window_and_out_of_range():
    with pytest.raises(ValueError, match="quá ngắn"):
        rv.frame_indices_for_times([0.01, 0.02, 0.03, 0.04], 30.0, 90)
    with pytest.raises(ValueError, match="ngoài video"):
        rv.frame_indices_for_times([3.1], 30.0, 90)
    with pytest.raises(ValueError, match="n_frames"):
        rv.frame_indices_for_times([0.1], 30.0, 0)


def test_read_frames_at_returns_rgb_frames_in_order(synthetic_video):
    frames = rv.read_frames_at(synthetic_video, [0, 30, 60])
    assert set(frames) == {0, 30, 60}
    assert all(f.shape == (48, 64, 3) for f in frames.values())
    means = [float(frames[i].mean()) for i in (0, 30, 60)]
    assert means == sorted(means) and means[2] - means[0] > 100


def test_output_path_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="video_id"):
        rv.output_path(tmp_path, "../x", 0)
    with pytest.raises(ValueError, match="window_index"):
        rv.output_path(tmp_path, "v1", -1)


# --- chạy end-to-end với FakeVerifier ---------------------------------------------

def test_one_json_per_window_with_k_frames(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, windows = write_windows(tmp_path)
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier, k=4)

    assert (summary.examined, summary.verified, summary.skipped) == (2, 2, 0)
    files = sorted((out / "v1").glob("window_*.json"))
    assert len(files) == 2
    for path, window in zip(files, windows):
        loaded = RunResult.from_json(path)
        assert loaded.video_id == "v1"
        assert loaded.candidate_windows == (window,)
        assert len(loaded.verify_results) == 1
        assert loaded.metadata["k"] == 4
        assert loaded.metadata["fake"] is True
        times = loaded.metadata["frame_times_s"]
        assert len(times) == 4 and len(set(times)) == 4
        assert all(window.start_s <= t <= window.end_s for t in times)
    assert len(verifier.calls) == 2
    assert all(n == 4 for n, _, _ in verifier.calls)
    assert all(all(shape == (48, 64, 3) for shape in shapes) for _, _, shapes in verifier.calls)


def test_frames_match_evenly_spaced_times(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    verifier = BrightnessVerifier()

    rv.run_verify(manifest, windows_file.parent, out, verifier=verifier, k=4)

    expected = [[40, 40, 40, 100], [100, 100, 160, 160]]
    for got, want in zip(verifier.means, expected):
        assert got == pytest.approx(want, abs=25)


def test_resume_skips_completed_and_recomputes_only_missing(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)
    assert len(verifier.calls) == 2

    second = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)
    assert (second.examined, second.verified, second.skipped) == (2, 0, 2)
    assert len(verifier.calls) == 2

    files = sorted((out / "v1").glob("window_*.json"))
    files[1].unlink()
    third = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)
    assert (third.examined, third.verified, third.skipped) == (2, 1, 1)
    assert len(verifier.calls) == 3
    assert verifier.calls[-1][1]["window_index"] == 1


def test_corrupt_json_and_tmp_are_not_treated_as_done(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    (out / "v1").mkdir(parents=True)
    (out / "v1" / "window_0000.json").write_text("{not json", encoding="utf-8")
    (out / "v1" / "window_0001.json.tmp").write_text("partial", encoding="utf-8")
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)

    assert (summary.verified, summary.skipped) == (2, 0)
    assert len(verifier.calls) == 2
    assert RunResult.from_json(out / "v1" / "window_0000.json").metadata["window_index"] == 0
    assert RunResult.from_json(out / "v1" / "window_0001.json").metadata["window_index"] == 1


def test_malformed_output_type_is_not_treated_as_done(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    (out / "v1").mkdir(parents=True)
    (out / "v1" / "window_0000.json").write_text(
        '{"schema_version": 1, "video_id": "v1", "sample_fps": 1.0, "candidate_windows": null}',
        encoding="utf-8",
    )
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)

    assert (summary.verified, summary.skipped) == (2, 0)
    assert len(verifier.calls) == 2


def test_verifier_error_writes_no_final_json(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"

    class Boom(FakeVerifier):
        def verify(self, frames, context):
            raise RuntimeError("VLM chết")

    with pytest.raises(RuntimeError, match="VLM chết"):
        rv.run_verify(manifest, windows_file.parent, out, verifier=Boom())
    assert not list(out.rglob("window_*.json"))


def test_fingerprint_change_invalidates_completed_files(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    rv.run_verify(manifest, windows_file.parent, out, verifier=verifier, k=4)
    assert len(verifier.calls) == 2

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier, k=2)
    assert (summary.verified, summary.skipped) == (2, 0)
    assert len(verifier.calls) == 4
    loaded = RunResult.from_json(out / "v1" / "window_0000.json")
    assert loaded.metadata["k"] == 2 and len(loaded.metadata["frame_times_s"]) == 2


def test_source_verify_results_are_ignored(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    old = (
        VerifyResult(True, EventType.FIGHTING, "[old] should be ignored", 0.1),
        VerifyResult(True, EventType.FIGHTING, "[old] should be ignored", 0.1),
    )
    windows_file, _ = write_windows(tmp_path, verify_results=old)
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier)

    assert summary.verified == 2
    assert len(verifier.calls) == 2
    loaded = RunResult.from_json(out / "v1" / "window_0000.json")
    assert loaded.verify_results[0].caption == "[fake] two people are fighting near a car"


def test_dry_run_does_not_verify_or_write(synthetic_video, tmp_path, capsys):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=None, dry_run=True)

    assert (summary.examined, summary.verified, summary.skipped) == (2, 0, 0)
    assert not out.exists()
    assert "[dry-run] v1: 2 window" in capsys.readouterr().out


def test_missing_video_raises_and_writes_nothing(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", tmp_path / "missing.mp4")
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"

    with pytest.raises(FileNotFoundError):
        rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier())
    assert not out.exists()


def test_manifest_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(
        "video_id,path,dataset,split\n"
        "v1,/x/a.mp4,ucf_crime,test\n"
        "v1,/x/b.mp4,ucf_crime,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trùng"):
        rv.load_manifest(path)


def test_cli_fake_verifier_smoke(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify_fake"

    rc = rv.main([
        "--manifest", str(manifest),
        "--windows-dir", str(windows_file.parent),
        "--out-dir", str(out),
        "--fake",
        "--limit", "1",
    ])

    assert rc == 0
    assert len(list((out / "v1").glob("window_*.json"))) == 1


def test_fake_verifier_rejects_default_verify_dir(synthetic_video, tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    monkeypatch.setattr(rv, "DEFAULT_VERIFY_DIR", tmp_path / "results" / "verify")

    with pytest.raises(ValueError, match="results/verify"):
        rv.run_verify(manifest, windows_file.parent, rv.DEFAULT_VERIFY_DIR,
                      verifier=FakeVerifier())


def test_sample_run_windows_are_processed_with_fake_verifier(tmp_path):
    """Dùng đúng results/windows/sample_run.json; video tổng hợp dài 110 giây để chứa các window."""
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for _ in range(550):
        writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
    writer.release()

    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    shutil.copy(ROOT / "results" / "windows" / "sample_run.json", windows_dir / "sample_run.json")
    manifest = write_manifest(tmp_path, "sample_run", video)
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_dir, out, verifier=verifier, k=4,
                            video_ids=["sample_run"])

    assert (summary.examined, summary.verified) == (3, 3)
    assert len(verifier.calls) == 3
    assert len(list((out / "sample_run").glob("window_*.json"))) == 3
