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


# --- sửa theo review PR #27 -------------------------------------------------------

def test_index_json_in_windows_dir_does_not_break_run(synthetic_video, tmp_path):
    """Blocker review: `results/windows/_index.json` không phải RunResult."""
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    (windows_file.parent / "_index.json").write_text(
        '{"ablation": "A4c", "n_windows": 2}', encoding="utf-8")
    out = tmp_path / "verify"

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier())
    dry = rv.run_verify(manifest, windows_file.parent, out, verifier=FakeVerifier(), dry_run=True)

    assert (summary.verified, summary.failed) == (2, 0)
    assert (dry.examined, dry.skipped) == (2, 2)
    assert rv.main(["--manifest", str(manifest), "--windows-dir", str(windows_file.parent),
                    "--out-dir", str(out), "--dry-run"]) == 0


def test_video_id_filter_opens_only_requested_file(synthetic_video, tmp_path):
    """`--video-id X` không được mở file windows của video khác (kể cả file hỏng)."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,path,dataset,split\n"
        f"v1,{synthetic_video.as_posix()},ucf_crime,test\n"
        "v2,/nope/v2.mp4,ucf_crime,test\n", encoding="utf-8")
    windows_file, _ = write_windows(tmp_path, "v1")
    (windows_file.parent / "v2.json").write_text("{broken", encoding="utf-8")

    summary = rv.run_verify(manifest, windows_file.parent, tmp_path / "verify",
                            verifier=CountingVerifier(), video_ids=["v1"])

    assert (summary.examined, summary.verified) == (2, 2)
    with pytest.raises(ValueError, match="manifest"):
        rv.run_verify(manifest, windows_file.parent, tmp_path / "verify2",
                      verifier=CountingVerifier(), video_ids=["ghost"])


def test_video_without_windows_file_is_skipped(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_dir = tmp_path / "empty_windows"
    windows_dir.mkdir()

    summary = rv.run_verify(manifest, windows_dir, tmp_path / "verify",
                            verifier=CountingVerifier())

    assert (summary.examined, summary.verified) == (0, 0)


def test_windows_file_for_rejects_path_video_id(tmp_path):
    with pytest.raises(ValueError, match="video_id"):
        rv.windows_file_for(tmp_path, "../x")


class FakeQuantVerifier(CountingVerifier):
    """Có đủ thuộc tính giống VLMVerifier thật để kiểm fingerprint."""

    model_id = "org/model"

    def __init__(self, quantization="4bit", max_attempts=2, system="s", user="u"):
        super().__init__()
        self.quantization = quantization
        self.max_attempts = max_attempts
        self.max_new_tokens = 128
        self._system = system
        self._user_prompt = user


@pytest.mark.parametrize("change", [
    {"quantization": "none"},
    {"max_attempts": 3},
    {"user": "prompt đã sửa"},
])
def test_verifier_config_change_invalidates_fingerprint(change):
    """Review #2: đổi quantization / max_attempts / prompt thì resume phải làm lại."""
    w = Window("v1", 0.2, 1.2, 0.9)
    base = FakeQuantVerifier()
    # FakeVerifier luôn được coi là 'fake' -> dùng lớp không kế thừa FakeVerifier cho nhánh thật.
    class Real:
        model_id = "org/model"
        def __init__(self, **kw):
            d = dict(quantization="4bit", max_attempts=2, system="s", user="u")
            d.update(kw)
            self.quantization = d["quantization"]
            self.max_attempts = d["max_attempts"]
            self.max_new_tokens = 128
            self._system = d["system"]
            self._user_prompt = d["user"]
    a, b = Real(), Real(**change)
    assert rv.make_fingerprint(w, 4, rv.verifier_name(a), rv.verifier_config(a)) != \
        rv.make_fingerprint(w, 4, rv.verifier_name(b), rv.verifier_config(b))
    assert rv.verifier_config(base) == {}  # FakeVerifier không có cấu hình


def test_prompt_hash_matches_bench_vlm_formula():
    import hashlib

    class Real:
        model_id = "m"
        quantization = "4bit"
        max_attempts = 2
        max_new_tokens = 128
        _system = "SYS"
        _user_prompt = "USER"

    expected = hashlib.sha1(b"SYSUSER").hexdigest()[:8]
    assert rv.verifier_config(Real())["prompt_hash"] == expected


def test_config_change_triggers_reverify_end_to_end(synthetic_video, tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    monkeypatch.setattr(rv, "verifier_name", lambda v: "org/model")
    monkeypatch.setattr(rv, "verifier_config",
                        lambda v: {"quantization": v.quantization})
    v4, v16 = CountingVerifier(), CountingVerifier()
    v4.quantization, v16.quantization = "4bit", "none"
    # tránh nhánh `--fake` chặn: chỉ cần out-dir khác results/verify (đã đúng)
    rv.run_verify(manifest, windows_file.parent, out, verifier=v4)
    again = rv.run_verify(manifest, windows_file.parent, out, verifier=v4)
    changed = rv.run_verify(manifest, windows_file.parent, out, verifier=v16)

    assert (again.verified, again.skipped) == (0, 2)
    assert (changed.verified, changed.skipped) == (2, 0)


def test_resume_does_not_need_video_when_everything_done(synthetic_video, tmp_path):
    """Review #2: máy chỉ có cache verify (không có video) vẫn resume/dry-run được."""
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"
    rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier())
    synthetic_video.unlink()

    again = rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier())
    dry = rv.run_verify(manifest, windows_file.parent, out, verifier=FakeVerifier(), dry_run=True)

    assert (again.verified, again.skipped, again.failed) == (0, 2, 0)
    assert (dry.skipped, dry.failed) == (2, 0)


def test_dry_run_does_not_need_video(tmp_path):
    manifest = write_manifest(tmp_path, "v1", tmp_path / "missing.mp4")
    windows_file, _ = write_windows(tmp_path)

    summary = rv.run_verify(manifest, windows_file.parent, tmp_path / "verify",
                            verifier=None, dry_run=True)

    assert summary.examined == 2


class FlakyVerifier(CountingVerifier):
    """Window index 0 luôn lỗi."""

    def verify(self, frames, context):
        if context["window_index"] == 0:
            raise RuntimeError("window 0 hỏng")
        return super().verify(frames, context)


def test_continue_on_error_records_failure_and_keeps_going(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    out = tmp_path / "verify"

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=FlakyVerifier(),
                            continue_on_error=True)

    assert (summary.verified, summary.failed) == (1, 1)
    assert "window 0 hỏng" in summary.errors[0]
    assert [p.name for p in (out / "v1").glob("window_*.json")] == ["window_0001.json"]


def test_continue_on_error_skips_window_too_short_for_k_frames(synthetic_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(
        tmp_path, windows=(Window("v1", 0.5, 0.52, 0.9), Window("v1", 1.5, 2.7, 0.8)))
    out = tmp_path / "verify"
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier,
                            continue_on_error=True)

    assert (summary.verified, summary.failed) == (1, 1)
    assert len(verifier.calls) == 1
    with pytest.raises(ValueError, match="quá ngắn"):
        rv.run_verify(manifest, windows_file.parent, tmp_path / "v2", verifier=CountingVerifier())


def test_continue_on_error_survives_missing_video_and_cli_exit_code(synthetic_video, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,path,dataset,split\n"
        "bad,/nope/bad.mp4,ucf_crime,test\n"
        f"v1,{synthetic_video.as_posix()},ucf_crime,test\n", encoding="utf-8")
    windows_file, _ = write_windows(tmp_path, "v1")
    write_windows(tmp_path, "bad")
    out = tmp_path / "verify"

    rc_strict = rv.main(["--manifest", str(manifest), "--windows-dir", str(windows_file.parent),
                         "--out-dir", str(out / "strict"), "--fake"])
    rc_soft = rv.main(["--manifest", str(manifest), "--windows-dir", str(windows_file.parent),
                       "--out-dir", str(out / "soft"), "--fake", "--continue-on-error"])

    assert rc_strict == 1
    assert rc_soft == 2
    assert len(list((out / "soft" / "v1").glob("window_*.json"))) == 2


def test_metadata_commit_is_recorded(synthetic_video, tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)
    monkeypatch.setattr(rv, "git_commit", lambda: "abc1234")
    out = tmp_path / "verify"

    rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier())

    assert RunResult.from_json(out / "v1" / "window_0000.json").metadata["commit"] == "abc1234"


def test_git_commit_falls_back_to_unknown(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(rv.subprocess, "run", boom)
    assert rv.git_commit() == "unknown"


def test_cli_dry_run_with_verifier_model_builds_lazy_verifier(synthetic_video, tmp_path, monkeypatch, capsys):
    """`--dry-run --verifier-model X` dựng VLMVerifier (lazy, không tải model) để fingerprint đúng."""
    import vad.verify.vlm as vlm

    built = []

    class Lazy:
        quantization, max_attempts, max_new_tokens = "4bit", 2, 128
        _system, _user_prompt = "s", "u"

        def __init__(self, model_id):
            self.model_id = model_id
            built.append(model_id)

        def verify(self, frames, context):  # pragma: no cover - dry-run never calls it
            raise AssertionError("dry-run must not verify")

    monkeypatch.setattr(vlm, "VLMVerifier", Lazy)
    manifest = write_manifest(tmp_path, "v1", synthetic_video)
    windows_file, _ = write_windows(tmp_path)

    rc = rv.main(["--manifest", str(manifest), "--windows-dir", str(windows_file.parent),
                  "--out-dir", str(tmp_path / "verify"), "--dry-run",
                  "--verifier-model", "org/m"])

    assert rc == 0 and built == ["org/m"]
    assert "[dry-run] v1: 2 window" in capsys.readouterr().out

# --- A3b: chunking (D-1), timing, shard -------------------------------------------

from vad.datasets.sharding import parse_shard_spec  # noqa: E402


@pytest.fixture()
def long_video(tmp_path):
    """70 giây ở 2 fps (140 khung); độ sáng = 10 * (giây // 10) để thấy chunk nào lấy khung nào."""
    path = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for i in range(140):
        writer.write(np.full((48, 64, 3), 20 + 30 * (i // 20), dtype=np.uint8))
    writer.release()
    return path


class ScriptedVerifier(FakeVerifier):
    """Trả lần lượt các VerifyResult định sẵn (mỗi chunk một cái)."""

    def __init__(self, results):
        super().__init__()
        self._queue = list(results)
        self.contexts: list[dict] = []

    def verify(self, frames, context):
        self.contexts.append(dict(context))
        return self._queue.pop(0)


def vr(is_anomaly, confidence, caption="c", event_type=EventType.FIGHTING):
    return VerifyResult(is_anomaly=is_anomaly, event_type=event_type, caption=caption,
                        confidence=confidence)


@pytest.mark.parametrize("start,end,expected_n", [
    (10.0, 40.0, 1),      # đúng 30.0 s -> không chia
    (0.0, 30.0, 1),
    (0.0, 31.0, 2),       # 31 s -> 2 chunk 15.5 s
    (5.0, 65.0, 2),       # 60 s -> 2 chunk 30 s
    (0.0, 60.5, 3),
    (0.0, 490.0, 17),     # window dài nhất thật: ceil(490/30)
    (0.0, 1.0, 1),
])
def test_split_into_chunks_boundaries(start, end, expected_n):
    chunks = rv.split_into_chunks(start, end, 30.0)
    assert len(chunks) == expected_n
    assert chunks[0][0] == start and chunks[-1][1] == end        # phủ kín, không trôi biên
    assert all(b - a <= 30.0 + 1e-9 for a, b in chunks)          # không chunk nào vượt mức
    assert all(a2 == b1 for (_, b1), (a2, _) in zip(chunks, chunks[1:]))  # liền nhau, không chồng


def test_split_into_chunks_honours_non_default_max():
    chunks = rv.split_into_chunks(0.0, 25.0, 10.0)
    assert len(chunks) == 3 and chunks[0] == pytest.approx((0.0, 25.0 / 3))


@pytest.mark.parametrize("bad", [(5.0, 5.0, 30.0), (-1.0, 3.0, 30.0), (0.0, 3.0, 0.0),
                                 (0.0, float("nan"), 30.0), (0.0, 3.0, float("inf"))])
def test_split_into_chunks_rejects_hostile_input(bad):
    with pytest.raises(ValueError):
        rv.split_into_chunks(*bad)


def test_merge_is_or_and_takes_strongest_anomalous_chunk():
    merged = rv.merge_chunk_results([
        vr(False, 0.99, "calm street"),
        vr(True, 0.60, "weak fight", EventType.ROBBERY),
        vr(True, 0.80, "strong fight", EventType.FIGHTING),
    ])
    assert merged.is_anomaly is True
    assert (merged.caption, merged.event_type, merged.confidence) == (
        "strong fight", EventType.FIGHTING, 0.80)   # KHÔNG lấy chunk bình thường 0.99


def test_merge_all_normal_takes_most_confident_and_ties_keep_first():
    merged = rv.merge_chunk_results([vr(False, 0.7, "a"), vr(False, 0.9, "b"), vr(False, 0.9, "c")])
    assert merged.is_anomaly is False and merged.caption == "b"
    with pytest.raises(ValueError):
        rv.merge_chunk_results([])


def test_long_window_is_chunked_with_one_json_and_or_verdict(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))  # 64 s
    out = tmp_path / "verify"
    verifier = ScriptedVerifier([vr(False, 0.9, "n1"), vr(True, 0.7, "hit", EventType.ROBBERY),
                                 vr(False, 0.8, "n3")])

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=verifier, k=4)

    assert (summary.verified, summary.chunk_calls, summary.chunked) == (1, 3, 1)
    files = list((out / "v1").glob("window_*.json"))
    assert len(files) == 1                                          # vẫn MỘT json mỗi window
    loaded = RunResult.from_json(files[0])
    r = loaded.verify_results[0]
    assert (r.is_anomaly, r.caption, r.event_type) == (True, "hit", EventType.ROBBERY)
    meta = loaded.metadata
    assert meta["n_chunks"] == 3 and len(meta["chunks"]) == 3
    assert meta["max_chunk_s"] == 30.0
    assert [c["is_anomaly"] for c in meta["chunks"]] == [False, True, False]
    assert len(meta["frame_times_s"]) == 12
    assert meta["elapsed_s"] >= 0.0
    # mỗi chunk gọi VLM với đúng biên chunk, không phải biên cả window
    starts = [c["start_s"] for c in verifier.contexts]
    assert starts == pytest.approx([2.0, 2.0 + 64 / 3, 2.0 + 128 / 3])
    assert all(c["window_start_s"] == 2.0 and c["window_end_s"] == 66.0 for c in verifier.contexts)


def test_chunk_frames_come_from_each_chunk(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))
    verifier = BrightnessVerifier()

    rv.run_verify(manifest, windows_file.parent, tmp_path / "verify", verifier=verifier, k=4)

    assert len(verifier.means) == 3
    firsts = [m[0] for m in verifier.means]
    assert firsts == sorted(firsts) and len(set(firsts)) == 3       # chunk sau sáng hơn chunk trước


def test_short_window_stays_one_chunk_and_not_counted_as_chunked(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 0.0, 30.0, 0.9),))
    verifier = CountingVerifier()

    summary = rv.run_verify(manifest, windows_file.parent, tmp_path / "verify",
                            verifier=verifier, k=4)

    assert (summary.chunk_calls, summary.chunked) == (1, 0)
    assert len(verifier.calls) == 1


def test_resume_skips_chunked_window_and_rerun_makes_no_vlm_calls(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))
    out = tmp_path / "verify"
    first = CountingVerifier()
    rv.run_verify(manifest, windows_file.parent, out, verifier=first, k=4)

    second = CountingVerifier()
    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=second, k=4)

    assert (summary.verified, summary.skipped, summary.chunk_calls) == (0, 1, 0)
    assert second.calls == []


def test_max_chunk_s_change_invalidates_resume(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))
    out = tmp_path / "verify"
    rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier(), k=4)

    summary = rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier(),
                            k=4, max_chunk_s=60.0)

    assert (summary.verified, summary.skipped, summary.chunk_calls) == (1, 0, 2)


def test_failed_chunk_writes_no_json(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))
    out = tmp_path / "verify"

    class Boom(FakeVerifier):
        n = 0

        def verify(self, frames, context):
            Boom.n += 1
            if Boom.n == 2:
                raise RuntimeError("vlm boom")
            return super().verify(frames, context)

    with pytest.raises(RuntimeError, match="vlm boom"):
        rv.run_verify(manifest, windows_file.parent, out, verifier=Boom(), k=4)
    assert list(out.rglob("window_*.json")) == []                   # không ghi kết quả một nửa


def test_timing_summary_reports_median_p90_and_percent_chunked(long_video, tmp_path):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(
        Window("v1", 0.0, 10.0, 0.9), Window("v1", 12.0, 66.0, 0.8)))
    summary = rv.run_verify(manifest, windows_file.parent, tmp_path / "verify",
                            verifier=CountingVerifier(), k=4)

    assert (summary.chunk_calls, summary.chunked, len(summary.elapsed_s)) == (3, 1, 2)
    report = summary.timing_report()
    assert "median" in report and "p90" in report
    assert "1/2 = 50.0%" in report and "chunk-call: 3" in report
    assert "chưa có" in rv.VerifySummary().timing_report()


def test_dry_run_counts_chunk_calls(long_video, tmp_path, capsys):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(
        Window("v1", 0.0, 10.0, 0.9), Window("v1", 2.0, 66.0, 0.8)))

    summary = rv.run_verify(manifest, windows_file.parent, tmp_path / "verify", verifier=None,
                            dry_run=True)

    assert summary.chunk_calls == 4 and "4 chunk-call" in capsys.readouterr().out
    assert not (tmp_path / "verify").exists()


def test_shard_union_equals_full_list_and_is_disjoint(long_video, tmp_path):
    n = 12
    windows = tuple(Window("v1", float(i * 5), float(i * 5 + 4), 0.9) for i in range(n))
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=windows)

    full_out = tmp_path / "full"
    rv.run_verify(manifest, windows_file.parent, full_out, verifier=CountingVerifier(), k=2)
    full = {p.name for p in (full_out / "v1").glob("window_*.json")}
    assert len(full) == n

    seen: list[str] = []
    for i in range(1, 4):
        out = tmp_path / f"shard{i}"
        idx, total = parse_shard_spec(f"{i}/3")
        summary = rv.run_verify(manifest, windows_file.parent, out, verifier=CountingVerifier(),
                                k=2, shard=(idx, total))
        names = [p.name for p in (out / "v1").glob("window_*.json")] if (out / "v1").exists() else []
        assert summary.examined == len(names)                       # ngoài shard không bị đếm
        seen += names
    assert sorted(seen) == sorted(full)                             # hợp = đủ, không trùng


def test_shard_none_and_invalid(long_video, tmp_path):
    assert rv.in_shard("v1", 0, None) is True
    with pytest.raises(ValueError):
        rv.in_shard("v1", 0, (2, 2))
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path)
    with pytest.raises(ValueError, match="shard"):
        rv.run_verify(manifest, windows_file.parent, tmp_path / "o", verifier=CountingVerifier(),
                      shard=(3, 2))


def test_cli_shard_and_max_chunk_flags(long_video, tmp_path, capsys):
    manifest = write_manifest(tmp_path, "v1", long_video)
    windows_file, _ = write_windows(tmp_path, windows=(Window("v1", 2.0, 66.0, 0.9),))
    base = ["--manifest", str(manifest), "--windows-dir", str(windows_file.parent), "--fake"]

    assert rv.main(base + ["--out-dir", str(tmp_path / "a"), "--shard", "1/1",
                           "--max-chunk-s", "20"]) == 0
    out = capsys.readouterr().out
    assert "chunk-call: 4" in out and "median" in out                # ceil(64/20) = 4
    with pytest.raises(SystemExit):
        rv.main(base + ["--out-dir", str(tmp_path / "b"), "--shard", "3/2"])
    with pytest.raises(SystemExit):
        rv.main(base + ["--out-dir", str(tmp_path / "c"), "--max-chunk-s", "0"])
