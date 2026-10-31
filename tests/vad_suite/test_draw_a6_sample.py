# -*- coding: utf-8 -*-
"""A6a — rút mẫu 20 cửa sổ kiểm tay (scripts/draw_a6_sample.py, Phước).

Kiểm: phân tầng 10/10, tính chồng lấn GT theo quy ước [start, end] đóng,
CSV blind không có caption/verdict/gt_overlap/score (đáp án ở file key riêng),
chạy lại cùng seed = y hệt, thiếu file tier-2 thì TỪ CHỐI (so tập, không so số lượng).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.draw_a6_sample import (  # noqa: E402
    BLIND_FIELDS, allocate, gt_overlap, load_verdicts, main,
)

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


def _run(scene, tmp_path: Path, *extra: str, name: str = "sample"):
    """Chạy main với out/key trong tmp; trả (rc, blind_path, key_path)."""
    out = tmp_path / f"{name}.csv"
    key = tmp_path / f"{name}-key.csv"
    extra = ("--hash-out", str(tmp_path / f"{name}.sha256"), *extra)
    rc = main(["--windows-dir", str(scene["windows"]), "--verify-dir", str(scene["verify"]),
               "--annotation", str(scene["annotation"]), "--out", str(out),
               "--key-out", str(key), *extra])
    return rc, out, key


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sample_is_balanced_and_stratified(scene, tmp_path):
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4")
    assert rc == 0
    rows = _rows(key)
    assert len(rows) == 8
    assert sum(r["verdict"] == "confirmed" for r in rows) == 4
    assert sum(r["verdict"] == "rejected" for r in rows) == 4
    for verdict in ("confirmed", "rejected"):
        classes = {r["video_class"] for r in rows if r["verdict"] == verdict}
        assert classes == {"anomalous", "normal"}, (verdict, classes)
    blind = _rows(out)
    assert len(blind) == 8
    assert list(blind[0]) == BLIND_FIELDS
    assert all(r["human_label"] == "" for r in blind)


def test_sample_has_no_caption_and_no_leak(scene, tmp_path):
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "caption" not in text.lower()
    assert "SECRET" not in text  # không được mang mô tả vào mẫu trước khi chấm tay
    for column in ("verdict", "video_class", "gt_overlap", "score"):
        assert column not in text, f"CSV blind còn cột đáp án: {column}"
    assert "human_label" in text
    # source_file trỏ tới file tier-2 có caption/is_anomaly -> không được nằm trong file blind
    assert "source_file" not in text and "verify/" not in text
    key_text = key.read_text(encoding="utf-8")
    assert "verdict" in key_text and "gt_overlap" in key_text


def test_draw_with_pool_larger_than_n_has_exact_stratum_counts(scene, tmp_path):
    """Pool mỗi verdict = 4 (> n=3): allocate chia theo tỉ lệ, không lấy hết pool."""
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "3")
    assert rc == 0
    rows = _rows(key)
    assert len(rows) == 6
    for verdict in ("confirmed", "rejected"):
        subset = [r for r in rows if r["verdict"] == verdict]
        assert len(subset) == 3
        assert sum(r["video_class"] == "anomalous" for r in subset) == 2
        assert sum(r["video_class"] == "normal" for r in subset) == 1


def test_allocate_is_proportional_and_keeps_min_one():
    assert allocate({"anomalous": 10, "normal": 2}, 6) == {"anomalous": 5, "normal": 1}
    assert allocate({"anomalous": 20, "normal": 1}, 4) == {"anomalous": 3, "normal": 1}
    assert allocate({"anomalous": 4, "normal": 4}, 4) == {"anomalous": 2, "normal": 2}
    assert allocate({"anomalous": 3}, 5) == {"anomalous": 3}


def test_same_seed_same_sample_different_seed_may_differ(scene, tmp_path):
    rc, a, a_key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4", name="a")
    assert rc == 0
    rc, b, b_key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4", name="b")
    assert rc == 0
    rc, c, _ = _run(scene, tmp_path, "--seed", "8", "--per-verdict", "4", name="c")
    assert rc == 0
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    assert a_key.read_text(encoding="utf-8") == b_key.read_text(encoding="utf-8")
    # seed khác: vẫn đúng cấu trúc (8 dòng, 4/4) dù có thể trùng lựa chọn
    assert len(_rows(c)) == 8


def test_blind_row_order_is_not_grouped_by_verdict(scene, tmp_path):
    """Thứ tự dòng không được lộ verdict: không có kiểu "nửa đầu confirmed, nửa sau rejected"."""
    for seed in ("7", "8", "9"):
        rc, out, key = _run(scene, tmp_path, "--seed", seed, "--per-verdict", "4", name=f"o{seed}")
        assert rc == 0
        verdicts = [r["verdict"] for r in _rows(key)]
        assert verdicts != sorted(verdicts) and verdicts != sorted(verdicts, reverse=True), verdicts
        # blind và key cùng thứ tự dòng
        blind_keys = [(r["video_id"], r["window_index"]) for r in _rows(out)]
        key_keys = [(r["video_id"], r["window_index"]) for r in _rows(key)]
        assert blind_keys == key_keys


def test_gt_overlap_rounds_both_edges_with_fractional_fps():
    annotations = {"V": type("A", (), {"segments": ((300, 600),)})()}
    # 29.97 fps: 10.01 s * 29.97 ~= 299.99 -> round = 300, phải chạm cận trái 300
    _, overlap = gt_overlap("V", {"start_s": 0.0, "end_s": 10.01, "fps": 29.97}, annotations)
    assert overlap


def test_key_hash_matches_key_file(scene, tmp_path):
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4")
    assert rc == 0
    recorded = (tmp_path / "sample.sha256").read_text(encoding="utf-8").split()[0]
    assert recorded == hashlib.sha256(key.read_bytes()).hexdigest()
    # hash không được lộ nội dung đáp án
    assert "confirmed" not in (tmp_path / "sample.sha256").read_text(encoding="utf-8")


def test_default_key_path_is_gitignored():
    from scripts.draw_a6_sample import DEFAULT_KEY_OUT, ROOT
    rel = DEFAULT_KEY_OUT.relative_to(ROOT).as_posix()
    assert rel.startswith("notes/private/")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/notes/private/" in ignore


def test_gt_overlap_uses_inclusive_frames():
    annotations = {"Anomalous01_x264": type("A", (), {"segments": ((300, 600),)})()}
    # giây 9.5..10.5 -> frame 285..315, chạm GT bắt đầu ở 300 (cận đóng)
    has_event, overlap = gt_overlap("Anomalous01_x264", {"start_s": 9.5, "end_s": 10.5, "fps": FPS}, annotations)
    assert has_event and overlap
    # khung vừa DỪNG trước frame 300 (0..299) -> không chồng, chặn cận đóng hở
    has_event, overlap = gt_overlap("Anomalous01_x264", {"start_s": 9.0, "end_s": 9.9, "fps": FPS}, annotations)
    assert has_event and not overlap
    has_event, overlap = gt_overlap("Anomalous01_x264", {"start_s": 60.0, "end_s": 70.0, "fps": FPS}, annotations)
    assert has_event and not overlap      # cùng video nhưng ngoài đoạn GT
    has_event, overlap = gt_overlap("Normal01_x264", {"start_s": 0.0, "end_s": 10.0, "fps": FPS}, annotations)
    assert not has_event and not overlap  # video bình thường -> không annotation


def test_missing_verify_file_refuses_to_draw(scene, tmp_path):
    # xoá 1 file tier-2 -> thiếu -> PHẢI từ chối khi chưa --allow-partial
    victim = scene["verify"] / "Normal01_x264" / "window_0003.json"
    victim.unlink()
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4")
    assert rc == 1 and not out.exists() and not key.exists()
    # chạy thử thì vẫn được — nhưng với số mẫu mà phần thiếu vẫn đáp ứng (3/3)
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "3", "--allow-partial")
    assert rc == 0 and out.exists() and key.exists()
    # còn thiếu mà vẫn đòi đủ 4/4 -> từ chối, không bao giờ ghi mẫu méo
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4", "--allow-partial")
    assert rc == 1


def test_missing_window_hidden_by_extra_verdict_file(scene, tmp_path):
    """Số file vẫn bằng số window nhưng thiếu 1 window thật + thừa 1 file stale.

    Guard cũ (`len(verdicts) < len(windows)`) không bắt được; guard mới so TẬP nên phải chặn.
    """
    (scene["verify"] / "Normal01_x264" / "window_0003.json").unlink()
    stale = scene["verify"] / "Normal01_x264" / "window_0099.json"
    stale.write_text(json.dumps({
        "video_id": "Normal01_x264",
        "metadata": {"window_index": 99, "k": 4, "elapsed_s": 1.0},
        "verify_results": [{"is_anomaly": False, "event_type": "Normal",
                            "caption": "stale", "confidence": 0.9}],
    }), encoding="utf-8")
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "3")
    assert rc == 1 and not out.exists() and not key.exists()
    # --allow-partial: chạy được, file stale bị bỏ qua (không lọt vào mẫu)
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "3", "--allow-partial")
    assert rc == 0
    assert "window_0099" not in key.read_text(encoding="utf-8")


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
    rc, out, key = _run(scene, tmp_path, "--seed", "7", "--per-verdict", "4")
    assert rc == 1 and not out.exists() and not key.exists()
