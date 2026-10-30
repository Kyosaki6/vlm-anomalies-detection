# -*- coding: utf-8 -*-
"""Test hàm thuần của `scripts/make_bench_windows.py` (A1redo §3.1) — không mạng, không video thật.

Chọn cửa sổ phải XÁC ĐỊNH (seed 0, cùng máy khác máy) và khung phải nằm trong đoạn GT.
"""
from __future__ import annotations

import pytest

from scripts.make_bench_windows import choose_windows, evenly_spaced
from vad.datasets.ucf_crime import Annotation


def test_evenly_spaced_dong_mut_bat_buoc() -> None:
    assert evenly_spaced(100, 200, 4) == [100, 133, 167, 200]
    assert evenly_spaced(10, 11, 2) == [10, 11]
    assert evenly_spaced(5, 5, 1) == [5]
    # k=1: midpoint; mọi chỉ số thuộc [start, end]
    assert evenly_spaced(0, 9, 1) == [4]


def test_evenly_spaced_tu_choi_input_hong() -> None:
    with pytest.raises(ValueError):
        evenly_spaced(10, 5, 4)
    with pytest.raises(ValueError):
        evenly_spaced(0, 10, 0)


def _rows(prefix: str, n: int, cls: str | None = None) -> list[dict]:
    return [
        {
            "video_id": f"{prefix}{i:03d}_x264",
            "path": f"datasets/ucf_crime/test/{cls or 'Normal'}/{prefix}{i:03d}_x264.mp4",
            "split": "test",
            "n_frames": 3000,
            "fps": 30.0,
        }
        for i in range(n)
    ]


def _fake_data() -> tuple[list[dict], list[dict], dict[str, Annotation]]:
    test_rows = _rows("Fighting", 3, "Fighting") + _rows("Arson", 3, "Arson")
    calib_rows = [
        {
            "video_id": f"Normal_Videos{i:03d}_x264",
            "path": f"datasets/ucf_crime/train_normal/Training-Normal-Videos-Part-1/"
                    f"Normal_Videos{i:03d}_x264.mp4",
            "split": "calib_normal",
            "n_frames": 2500,
            "fps": 30.0,
        }
        for i in range(6)
    ]
    annotations = {
        r["video_id"]: Annotation(video_id=r["video_id"], label="Fighting",
                                  segments=((100, 400),))
        for r in test_rows[:3]
    }
    annotations.update({
        r["video_id"]: Annotation(video_id=r["video_id"], label="Arson",
                                  segments=((500, 900),))
        for r in test_rows[3:]
    })
    return test_rows, calib_rows, annotations


def test_choose_windows_xac_dinh_va_nam_trong_gt() -> None:
    test_rows, calib_rows, annotations = _fake_data()
    a = choose_windows(test_rows, calib_rows, annotations, seed=0, k=4, n_anomaly=2, n_normal=3)
    b = choose_windows(test_rows, calib_rows, annotations, seed=0, k=4, n_anomaly=2, n_normal=3)
    assert a == b, "cùng seed phải ra đúng một tập cửa sổ"

    anomalies = [w for w in a if w["label"] == "anomaly"]
    normals = [w for w in a if w["label"] == "normal"]
    assert len(anomalies) == 2 and len(normals) == 3
    assert {w["gt_event"] for w in anomalies} == {"Fighting", "Arson"}  # mỗi lớp một cửa sổ

    for w in anomalies:
        seg = w["segment"]
        assert w["frame_indices"] == evenly_spaced(seg[0], seg[1], 4)
        assert seg[0] <= w["frame_indices"][0] <= w["frame_indices"][-1] <= seg[1]
    for w in normals:
        seg = w["segment"]
        assert 0 <= seg[0] <= w["frame_indices"][-1] <= seg[1] < w["n_frames"]
        # span normal 4-8 giây tại 30 fps -> 120..240 khung (kẹp trên bởi n_frames)
        assert 118 <= seg[1] - seg[0] <= 241


def test_choose_windows_khong_co_gt_du_dai_thi_loi() -> None:
    test_rows, calib_rows, _ = _fake_data()
    empty_ann: dict[str, Annotation] = {}
    with pytest.raises(ValueError):
        choose_windows(test_rows, calib_rows, empty_ann, seed=0, n_anomaly=2, n_normal=1)
