# -*- coding: utf-8 -*-
"""Test logic ánh xạ split -> đường dẫn repo HF của `scripts/download_ucf_crime.py` (không chạm mạng)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import download_ucf_crime as d  # noqa: E402


def make_repo(n_test_anom=140, n_test_norm=150, n_train_norm=100):
    """Repo HF giả, tái hiện đúng bẫy thật: train-video trùng tên trong `test/`, và thư mục event-recognition."""
    files, test, train = set(), [], []
    for i in range(n_test_anom):
        name = f"Abuse{i:03d}_x264.mp4"
        files.add(f"Anomaly-Videos-Part-{i % 4 + 1}/Abuse/{name}")
        test.append(f"Abuse/{name}")
    for i in range(n_test_norm):
        name = f"Normal_Videos_{i:03d}_x264.mp4"
        files.add(f"Testing_Normal_Videos_Anomaly/{name}")
        files.add(f"Normal_Videos_for_Event_Recognition/{name}")      # TRÙNG tên: không được chọn nhầm
        test.append(f"Testing_Normal_Videos_Anomaly/{name}")
    for i in range(n_train_norm):
        name = f"Normal_Videos{i + 500:03d}_x264.mp4"
        files.add(f"Training-Normal-Videos-Part-{i % 2 + 1}/{name}")
        train.append(f"Training_Normal_Videos_Anomaly/{name}")
    files.add("test/Abuse/Abuse000_x264.mp4")                          # bản sao thật trong repo, KHÔNG thuộc Anomaly-Videos
    return files, test, train


def test_resolves_to_unique_correct_folder_not_basename_lookup():
    files, test, train = make_repo()
    test_paths, calib = d.plan_downloads(test, train, files, calib_count=10)
    assert len(test_paths) == 290
    assert not any(p.startswith(("test/", "Normal_Videos_for_Event_Recognition/")) for p in test_paths)
    assert all(p.startswith(d.TEST_NORMAL_DIR) or p.startswith(d.ANOMALY_PREFIX) for p in test_paths)
    assert all(p.startswith("Training-Normal-Videos-Part-") for p in calib) and len(calib) == 10


def test_missing_entry_raises():
    files, test, train = make_repo()
    files.discard("Anomaly-Videos-Part-1/Abuse/Abuse000_x264.mp4")
    with pytest.raises(ValueError, match="0 khớp"):
        d.plan_downloads(test, train, files, calib_count=5)


def test_ambiguous_entry_raises():
    files, test, train = make_repo()
    files.add("Anomaly-Videos-Part-2/Abuse/Abuse000_x264.mp4")        # cùng tên ở 2 Part
    with pytest.raises(ValueError, match="2 khớp"):
        d.plan_downloads(test, train, files, calib_count=5)


def test_wrong_test_count_raises():
    files, test, train = make_repo()
    with pytest.raises(ValueError, match="290"):
        d.plan_downloads(test[:-1], train, files, calib_count=5)


def test_calib_selection_deterministic_and_seed_sensitive():
    files, test, train = make_repo()
    a = d.plan_downloads(test, train, files, 10, seed=0)[1]
    assert a == d.plan_downloads(test, list(reversed(train)), files, 10, seed=0)[1]   # không phụ thuộc thứ tự file
    assert a != d.plan_downloads(test, train, files, 10, seed=3)[1]


def test_size_cap_excludes_huge_videos_before_picking():
    files, test, train = make_repo()
    sizes = {f: 10_000_000 for f in files}
    huge = {"Training-Normal-Videos-Part-1/Normal_Videos500_x264.mp4", "Training-Normal-Videos-Part-2/Normal_Videos501_x264.mp4"}
    for h in huge:
        sizes[h] = 5_000_000_000
    # 100 train-normal, 2 khổng lồ; lấy 98 = mọi video nhỏ -> phải KHÔNG có video khổng lồ
    _, calib = d.plan_downloads(test, train, files, 98, sizes=sizes, max_calib_mb=100)
    assert huge.isdisjoint(calib) and len(calib) == 98
    with pytest.raises(ValueError):                                   # 99 > số video nhỏ còn lại
        d.plan_downloads(test, train, files, 99, sizes=sizes, max_calib_mb=100)
    with pytest.raises(ValueError, match="sizes"):
        d.plan_downloads(test, train, files, 5, max_calib_mb=100)


def test_parse_split_entries_handles_bom_crlf_backslash_and_dupes():
    assert d.parse_split_entries("\ufeffAbuse\\A.mp4\r\n\r\nArson/B.mp4\r\n") == ["Abuse/A.mp4", "Arson/B.mp4"]
    with pytest.raises(ValueError):
        d.parse_split_entries("a/x.mp4\na/x.mp4\n")


def test_run_downloads_reports_failures_and_continues(tmp_path):
    calls = []

    def fake(repo_id, repo_type, filename, local_dir):
        calls.append(filename)
        if filename == "bad.mp4":
            raise OSError("boom")
        return str(tmp_path / filename)

    failed = d.run_downloads(["a.mp4", "bad.mp4", "c.mp4"], tmp_path, threads=2, hf_hub_download=fake)
    assert failed == ["bad.mp4"] and sorted(calls) == ["a.mp4", "bad.mp4", "c.mp4"]
