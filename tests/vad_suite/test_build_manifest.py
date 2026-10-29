# -*- coding: utf-8 -*-
"""Test `scripts/build_manifest.py` (B1) — video tự sinh bằng OpenCV, không cần dataset thật."""
import csv

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import build_manifest as bm  # noqa: E402
from precompute_embeddings import plan_work, read_manifest  # noqa: E402


def make_video(path: Path, n_frames=30, fps=30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not w.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for i in range(n_frames):
        w.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    w.release()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """Giả lập gốc repo = tmp_path để kiểm tra đường dẫn tương đối."""
    monkeypatch.setattr(bm, "REPO_ROOT", tmp_path.resolve())
    return tmp_path


def test_rows_use_relative_posix_path_and_real_probe(repo):
    make_video(repo / "datasets/ucf_crime/test/Abuse/Abuse028_x264.mp4", n_frames=60, fps=30.0)
    rows = bm.build_rows(bm.find_videos(repo / "datasets/ucf_crime/test"), "ucf_crime", "test")
    assert rows == [{"video_id": "Abuse028_x264", "path": "datasets/ucf_crime/test/Abuse/Abuse028_x264.mp4",
                     "dataset": "ucf_crime", "split": "test", "n_frames": 60, "fps": "30"}]


def test_path_outside_repo_is_rejected(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "x.mp4"
    make_video(outside)
    with pytest.raises(ValueError, match="ngoài repo"):
        bm.build_rows({"x": outside}, "ucf_crime", "test")
    rows = bm.build_rows({"x": outside}, "ucf_crime", "test", allow_absolute=True)
    assert Path(rows[0]["path"]).is_absolute()


def test_duplicate_stem_in_two_folders_raises(repo):
    make_video(repo / "v/a/Same.mp4")
    make_video(repo / "v/b/Same.mp4")
    with pytest.raises(ValueError, match="trùng"):
        bm.find_videos(repo / "v")


def test_corrupt_video_listed_not_silently_skipped(repo):
    make_video(repo / "v/ok.mp4")
    (repo / "v").joinpath("broken.mp4").write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="broken"):
        bm.build_rows(bm.find_videos(repo / "v"), "ucf_crime", "test")


def test_missing_root_raises(repo):
    with pytest.raises(FileNotFoundError):
        bm.find_videos(repo / "nope")


def test_read_id_list_accepts_folder_prefix_bom_and_blank_lines(tmp_path):
    f = tmp_path / "list.txt"
    f.write_bytes("\ufeffAbuse/Abuse028_x264.mp4\n\nNormal_Videos_003_x264.mp4\r\nTesting_Normal_Videos_Anomaly/Normal_Videos_900_x264.mp4\n".encode("utf-8"))
    assert bm.read_id_list(f) == ["Abuse028_x264", "Normal_Videos_003_x264", "Normal_Videos_900_x264"]


def test_read_id_list_duplicate_raises(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("a.mp4\nb/a.mp4\n")
    with pytest.raises(ValueError):
        bm.read_id_list(f)


def test_pick_limit_is_deterministic_order_independent_and_seed_sensitive():
    ids = [f"Normal_{i:03d}" for i in range(100)]
    a = bm.pick_limit(ids, 10, seed=0)
    assert a == bm.pick_limit(list(reversed(ids)), 10, seed=0)      # không phụ thuộc thứ tự quét
    assert len(set(a)) == 10 and set(a) <= set(ids)
    assert a != bm.pick_limit(ids, 10, seed=1)                      # seed khác -> tập khác
    with pytest.raises(ValueError):
        bm.pick_limit(ids, 101, seed=0)


def test_cli_end_to_end_then_precompute_plan_reads_it(repo, monkeypatch):
    for name in ("Abuse028_x264", "Arson011_x264", "Normal_Videos_003_x264"):
        make_video(repo / f"datasets/ucf_crime/test/{name}.mp4")
    (repo / "list.txt").write_text("Abuse/Abuse028_x264.mp4\nArson/Arson011_x264.mp4\nNormal_Videos_003_x264.mp4\n")
    (repo / "ann.txt").write_text("Abuse028_x264.mp4  Abuse  5  20  -1  -1\n")
    out = repo / "manifests/ucf_test.csv"
    rc = bm.main(["--root", str(repo / "datasets/ucf_crime/test"), "--split", "test", "--list", str(repo / "list.txt"),
                  "--annotation", str(repo / "ann.txt"), "--expect", "3", "--out", str(out)])
    assert rc == 0
    with open(out, newline="", encoding="utf-8") as f:
        assert [r["video_id"] for r in csv.DictReader(f)] == ["Abuse028_x264", "Arson011_x264", "Normal_Videos_003_x264"]
    refs = read_manifest(out)                                       # chính loader của precompute đọc được
    todo = []
    for i in range(1, 4):
        t, _ = plan_work(refs, i - 1, 3, "m", 1.0, repo / "cache")
        todo += t
    assert sorted(r.video_id for r in todo) == sorted(r.video_id for r in refs)


def test_cli_expect_mismatch_writes_nothing(repo):
    make_video(repo / "v/a.mp4")
    out = repo / "m.csv"
    with pytest.raises(SystemExit, match="Mong đợi 290"):
        bm.main(["--root", str(repo / "v"), "--split", "test", "--expect", "290", "--out", str(out)])
    assert not out.exists()


def test_cli_missing_listed_video_and_annotation_orphan_are_errors(repo):
    make_video(repo / "v/a.mp4")
    (repo / "list.txt").write_text("a.mp4\nghost.mp4\n")
    with pytest.raises(SystemExit, match="Thiếu 1 video"):
        bm.main(["--root", str(repo / "v"), "--split", "test", "--list", str(repo / "list.txt"), "--out", str(repo / "m.csv")])
    (repo / "ann.txt").write_text("a.mp4 Abuse 1 2 -1 -1\nghost.mp4 Abuse 1 2 -1 -1\n")
    with pytest.raises(SystemExit, match="annotation"):
        bm.main(["--root", str(repo / "v"), "--split", "test", "--annotation", str(repo / "ann.txt"), "--out", str(repo / "m.csv")])


def test_cli_limit_uses_non_default_seed(repo):
    for i in range(12):
        make_video(repo / f"n/Normal_{i:02d}.mp4", n_frames=3)
    outs = []
    for seed in ("0", "7"):
        out = repo / f"m{seed}.csv"
        bm.main(["--root", str(repo / "n"), "--split", "calib_normal", "--limit", "4", "--seed", seed, "--expect", "4", "--out", str(out)])
        outs.append([r.video_id for r in read_manifest(out)])
    assert len(outs[0]) == len(outs[1]) == 4 and outs[0] != outs[1]
