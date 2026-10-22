# -*- coding: utf-8 -*-
"""
tests/test_main.py — unit test cho main.py pipeline (model-free).

Mọi test đều mock hàm search và anomaly_score để chạy tức thì (< 1s),
không cần tải model SigLIP 2 (1.5 GB).
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

import graph
import main
from main import PipelineResult, NORMAL_ACTION

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_IMAGE = str(DATA_DIR / "duong_pho_1.jpg")


def test_pipeline_result_dataclass_fields():
    res = PipelineResult(
        image_path="test.jpg",
        is_abnormal=False,
        anomaly_score=0.1,
        threshold=0.3,
        normal_similarity=0.9,
        best_file="duong_pho_1.jpg",
        score=95,
        event_id=None,
        event_name=None,
        action=NORMAL_ACTION,
        reference_event_id="6",
        reference_event_name="Phố đêm có hàng quán",
    )
    assert res.image_path == "test.jpg"
    assert res.is_abnormal is False
    assert res.action == NORMAL_ACTION
    assert res.reference_event_id == "6"

    with pytest.raises((FrozenInstanceError, AttributeError)):
        res.is_abnormal = True  # type: ignore


def test_run_normal_verdict_consistent_action(monkeypatch):
    monkeypatch.setattr(main, "search", lambda p: ("duong_pho_1.jpg", 95))
    monkeypatch.setattr(main, "anomaly_score", lambda p: (0.85, 0.15))

    res = main.run(SAMPLE_IMAGE, threshold=0.30, raise_on_error=True, verbose=False)
    assert res is not None
    assert res.is_abnormal is False
    assert res.anomaly_score == 0.15
    assert res.action == NORMAL_ACTION
    assert "Không cần xử lí" in res.action
    assert res.event_id is None
    assert res.reference_event_id is not None
    assert res.reference_event_name is not None


def test_run_abnormal_verdict_matches_graph_action(monkeypatch):
    monkeypatch.setattr(main, "search", lambda p: ("danh_nhau_1.jpg", 98))
    monkeypatch.setattr(main, "anomaly_score", lambda p: (0.60, 0.40))

    res = main.run(SAMPLE_IMAGE, threshold=0.30, raise_on_error=True, verbose=False)
    assert res is not None
    assert res.is_abnormal is True
    assert res.anomaly_score == 0.40
    assert res.event_id == "1"
    assert res.event_name == graph.get_event("1")
    assert res.action == graph.get_action("1")
    assert "Không cần xử lí" not in res.action
    assert res.reference_event_id is None
    assert res.reference_event_name is None


def test_run_threshold_boundary_equal(monkeypatch):
    monkeypatch.setattr(main, "search", lambda p: ("danh_nhau_2.jpg", 90))
    monkeypatch.setattr(main, "anomaly_score", lambda p: (0.70, 0.30))

    # score == threshold -> abnormal
    res = main.run(SAMPLE_IMAGE, threshold=0.30, raise_on_error=True, verbose=False)
    assert res is not None
    assert res.is_abnormal is True
    assert res.action == graph.get_action("2")


def test_run_threshold_boundary_below(monkeypatch):
    monkeypatch.setattr(main, "search", lambda p: ("tai_nan_xe_1.jpg", 90))
    monkeypatch.setattr(main, "anomaly_score", lambda p: (0.71, 0.29))

    # score < threshold -> normal
    res = main.run(SAMPLE_IMAGE, threshold=0.30, raise_on_error=True, verbose=False)
    assert res is not None
    assert res.is_abnormal is False
    assert res.action == NORMAL_ACTION
    assert res.reference_event_id is not None


def test_run_file_not_found_raise():
    with pytest.raises(FileNotFoundError):
        main.run("/nonexistent/dummy_path_12345.jpg", raise_on_error=True, verbose=False)


def test_run_file_not_found_no_raise(capsys):
    res = main.run("/nonexistent/dummy_path_12345.jpg", raise_on_error=False, verbose=False)
    assert res is None
    captured = capsys.readouterr()
    assert "[main.py] Lỗi: không thấy ảnh" in captured.out


def test_run_search_exception_handling(monkeypatch, capsys):
    def mock_fail(p):
        raise RuntimeError("search failed")

    monkeypatch.setattr(main, "search", mock_fail)

    with pytest.raises(RuntimeError):
        main.run(SAMPLE_IMAGE, raise_on_error=True, verbose=False)

    res = main.run(SAMPLE_IMAGE, raise_on_error=False, verbose=False)
    assert res is None
    captured = capsys.readouterr()
    assert "Lỗi khi tìm ảnh" in captured.out


def test_run_anomaly_score_exception_handling(monkeypatch, capsys):
    monkeypatch.setattr(main, "search", lambda p: ("duong_pho_1.jpg", 95))

    def mock_fail(p):
        raise RuntimeError("anomaly failed")

    monkeypatch.setattr(main, "anomaly_score", mock_fail)

    with pytest.raises(RuntimeError):
        main.run(SAMPLE_IMAGE, raise_on_error=True, verbose=False)

    res = main.run(SAMPLE_IMAGE, raise_on_error=False, verbose=False)
    assert res is None
    captured = capsys.readouterr()
    assert "Lỗi khi tính điểm bất thường" in captured.out


def test_run_unmapped_event_file(monkeypatch, capsys):
    monkeypatch.setattr(main, "search", lambda p: ("unmapped_image_xyz.jpg", 95))
    monkeypatch.setattr(main, "anomaly_score", lambda p: (0.5, 0.5))

    with pytest.raises(ValueError) as exc:
        main.run(SAMPLE_IMAGE, raise_on_error=True, verbose=False)
    assert "unmapped_image_xyz.jpg" in str(exc.value)

    res = main.run(SAMPLE_IMAGE, raise_on_error=False, verbose=False)
    assert res is None
    captured = capsys.readouterr()
    assert "Cảnh báo: ảnh khớp" in captured.out


def test_load_file_to_id():
    mapping = main.load_file_to_id(raise_on_error=True)
    assert mapping is not None
    assert len(mapping) == 15
    assert "danh_nhau_1.jpg" in mapping
    assert mapping["danh_nhau_1.jpg"] == "1"

    with pytest.raises(FileNotFoundError):
        main.load_file_to_id("/nonexistent_path/events.json", raise_on_error=True)

    assert main.load_file_to_id("/nonexistent_path/events.json", raise_on_error=False) is None
