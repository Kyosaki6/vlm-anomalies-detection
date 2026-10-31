# -*- coding: utf-8 -*-
"""A6a — rút mẫu 20 cửa sổ kiểm tay (scripts/draw_a6_sample.py, Phước).

Kiểm: phân tầng 10/10, tính chồng lấn GT theo quy ước [start, end] đóng,
không có cột caption (chọn mẫu TRƯỚC khi đọc kết quả), chạy lại cùng seed = y hệt,
thiếu file tier-2 thì TỪ CHỐI (không rút mẫu méo).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.draw_a6_sample import draw_sample, gt_overlap, load_verdicts, main  # noqa: E402

FPS = 30.0


def _write_windows(dirpath: Path, spec: dict[str, list[tuple[float, float]]]) -> None:
    """spec: {video_id: [(start_s, end_s), ...]}. metadata.fps/n_frames là bắt buộc."""
    index = {"windows": []}
    for video_id, spans in spec.items():
        rows = []
        for i, (start, end) in enumerate(spans):
            index["windows"].append({
                "video_id": video_id, "window_index": i,
                "start_s": start, "end_s": end, "score": 1.2, "duration_s": end - start,
            })
            rows.append({
                "video_id": video_id,
                "start_s": start, "end_s": end, "score": 1.2,
                "candidate_windows": [{"video_id": video_id, "start_s": start,
                                       "end_s": end, "score": 1.2}],
            })
        (dirpath / f"{video_id}.json").write_text(json.dumps({
            "metadata": {"fps": FPS, "n_frames": 18000},
            "candidate_windows": rows,
        }), encoding="utf-8")
    (dirpath / "_index.json").write_text(json.dumps(index), encoding="utf-8")


def _write_verify(dirpath: Path, verdicts: dict[tuple[str, int], bool]) -> None:
    for (video_id, w_index), is_anomaly in verdicts.items():
        target = dirpath / video_id
        target.mkdir(parents=True, exist_ok=True)
        (target / f"window_{w_index:04d}.json").write_text(json.dumps({
            "video_id": video_id,
            "metadata": {"window_index": w_index, "k": 4, "elapsed_s": 4.5},
            "verify_results": [{"is_anomaly": is_anomaly, "event_type": "Arrest",
                                "caption": "SECRET: must not enter the sample CSV",
                                "confidence": 0.95}],
        }), encoding="utf-8")


def _write_annotation(path: Path, line: str = "Anomalous01_x264  Arrest  300  600  -1  -1") -> None:
    path.write_text(line + "\n", encoding="utf-8")


@pytest.fixture()
def scene(tmp_path: Path) -> dict[str, Path]:
    """2 video: 1 có annotation (đoạn frame 300..600 = giây 10..20), 1 bình thường.
    Mỗi video 4 cửa sổ; 4 confirmed + 4 rejected xen kẽ."""
    windows_dir = tmp_path / "windows"
    verify_dir = tmp_path / "verify"
    windows_dir.mkdir()
    _write_windows(windows_dir, {
        "Anomalous01_x264": [(9.0, 21.0),      # chồng GT (300..600)
                             (60.0, 70.0),      # KHÔNG chồng GT
                             (100.0, 110.0),
                             (120.0, 130.0)],
        "Normal01_x264": [(0.0, 10.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0)],
    })
    _write_verify(verify_dir, {
        ("Anomalous01_x264", 0): True, ("Anomalous01_x264", 1): False,
        ("Anomalous01_x264", 2): True, ("Anomalous01_x264", 3): False,
        ("Normal01_x264", 0): True, ("Normal01_x264", 1): False,
        ("Normal01_x264", 2): True, ("Normal01_x264", 3): False,
    })
    annotation = tmp_path / "annotation.txt"
    _write_annotation(annotation)
    return {"windows": windows_dir, "verify": verify_dir, "annotation": annotation}


def test_sample_is_balanced_and_stratified(scene, tmp_path):
    out = tmp_path / "sample.csv"
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--seed", "7", "--per-verdict", "4"])
    assert rc == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert sum(r["verdict"] == "confirmed" for r in rows) == 4
    assert sum(r["verdict"] == "rejected" for r in rows) == 4
    # phân tầng: mỗi phía đều có video anomalous và normal (pool đủ nên allocate chia đều)
    for verdict in ("confirmed", "rejected"):
        classes = {r["video_class"] for r in rows if r["verdict"] == verdict}
        assert classes == {"anomalous", "normal"}, (verdict, classes)


def test_sample_has_no_caption_and_no_leak(scene, tmp_path):
    out = tmp_path / "sample.csv"
    main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
          "--annotation", str(scene["annotation"]), "--out", str(out),
          "--seed", "7", "--per-verdict", "4"])
    text = out.read_text(encoding="utf-8")
    assert "caption" not in text.lower()
    assert "SECRET" not in text  # không được mang mô tả vào mẫu trước khi chấm tay


def test_same_seed_same_sample_different_seed_may_differ(scene, tmp_path):
    args = ["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
            "--annotation", str(scene["annotation"]), "--per-verdict", "4"]
    a, b, c = (tmp_path / n for n in ("a.csv", "b.csv", "c.csv"))
    assert main(args + ["--out", str(a), "--seed", "7"]) == 0
    assert main(args + ["--out", str(b), "--seed", "7"]) == 0
    assert main(args + ["--out", str(c), "--seed", "8"]) == 0
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    # seed khác: vẫn đúng cấu trúc (8 dòng, 4/4) dù có thể trùng lựa chọn
    with open(c, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8


def test_gt_overlap_uses_inclusive_frames(scene):
    windows = {"Anomalous01_x264": (9.0, 21.0)}  # frame 300 vs GT 300..600
    annotations = {"Anomalous01_x264": type("A", (), {"segments": ((300, 600),)})()}
    # giây 9.5..10.5 -> frame 285..315, chạm GT bắt đầu ở 300 (cận đóng)
    has_event, overlap = gt_overlap({"video_id": "Anomalous01_x264",
                                     "start_s": 9.5, "end_s": 10.5, "fps": FPS}, annotations)
    assert has_event and overlap
    # khung vừa DỪNG trước frame 300 (0..299) -> không chồng, chặn cận đóng hở
    has_event, overlap = gt_overlap({"video_id": "Anomalous01_x264",
                                     "start_s": 9.0, "end_s": 9.9, "fps": FPS}, annotations)
    assert has_event and not overlap
    has_event, overlap = gt_overlap({"video_id": "Anomalous01_x264",
                                     "start_s": 60.0, "end_s": 70.0, "fps": FPS}, annotations)
    assert has_event and not overlap      # cùng video nhưng ngoài đoạn GT
    has_event, overlap = gt_overlap({"video_id": "Normal01_x264",
                                     "start_s": 0.0, "end_s": 10.0, "fps": FPS}, annotations)
    assert not has_event and not overlap  # video bình thường -> không annotation


def test_missing_verify_file_refuses_to_draw(scene, tmp_path):
    # xoá 1 file tier-2 -> thiếu -> PHẢI từ chối khi chưa --allow-partial
    victim = scene["verify"] / "Normal01_x264" / "window_0003.json"
    victim.unlink()
    out = tmp_path / "sample.csv"
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--seed", "7", "--per-verdict", "4"])
    assert rc == 1 and not out.exists()
    # chạy thử thì vẫn được — nhưng với số mẫu mà phần thiếu vẫn đáp ứng (3/3)
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--seed", "7", "--per-verdict", "3", "--allow-partial"])
    assert rc == 0 and out.exists()   # chạy thử thì vẫn được
    # còn thiếu mà vẫn đòi đủ 4/4 -> từ chối, không bao giờ ghi mẫu méo
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--seed", "7", "--per-verdict", "4", "--allow-partial"])
    assert rc == 1


def test_missing_or_wrong_verify_result_raises(scene, tmp_path):
    bad = scene["verify"] / "Normal01_x264" / "window_0001.json"
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["verify_results"] = []
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_verdicts(scene["verify"])


def test_unbalanced_verdicts_refuse(scene, tmp_path):
    # toàn bộ verdict = rejected -> không đủ 4 confirmed -> từ chối, không rút lệch
    _write_verify(scene["verify"], {
        ("Anomalous01_x264", i): False for i in range(4)
    } | {("Normal01_x264", i): False for i in range(4)})
    out = tmp_path / "sample.csv"
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--seed", "7", "--per-verdict", "4"])
    assert rc == 1 and not out.exists()
