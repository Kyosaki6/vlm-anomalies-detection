# -*- coding: utf-8 -*-
"""Test C3 — `scripts/eval_typer.py`. Không GPU, không model, không video thật.

Điểm dễ sai khi chấm là chuyển frame 30 fps -> khung đã lấy mẫu và cách xử lý đoạn quá ngắn.
Đó là phần được test kỹ nhất.
"""
import json

import numpy as np
import pytest

from scripts.eval_typer import (
    LENIENT_GROUPS,
    SOURCE_FPS,
    build_cases,
    evaluate_cases,
    main,
    segment_sample_range,
    summarise,
)
from vad.datasets.ucf_crime import Annotation, load_annotations
from vad.embed.cache import save_embeddings
from vad.score.typing import TypingResult
from vad.types import EventType


# --- chuyển frame 30 fps -> khung đã lấy mẫu ---------------------------------------

def test_one_fps_takes_one_frame_per_30_source_frames():
    # đoạn frame 0..29 = 1 giây đầu -> đúng 1 khung ở 1 fps
    assert segment_sample_range(0, 29, 1.0, 100) == (0, 0)
    # frame 30..59 = giây thứ 2
    assert segment_sample_range(30, 59, 1.0, 100) == (1, 1)


def test_two_fps_doubles_the_indices():
    assert segment_sample_range(0, 59, 2.0, 100) == (0, 3)   # 0,1,2,3 (floor(59*2/30)=3)


def test_long_segment_spans_many_samples():
    i0, i1 = segment_sample_range(0, 3000, 1.0, 100)          # 100 giây
    assert (i0, i1) == (0, 99)                                # bị cắt theo cache


def test_segment_past_the_end_is_clamped():
    # frame 90 = 3.0 s -> khung 3; frame 200 = 6.67 s -> khung 6 (vẫn < 9 nên không bị cắt thêm)
    assert segment_sample_range(90, 200, 1.0, 10) == (3, 6)
    # và khi end vượt hẳn cache thì cắt về khung cuối
    assert segment_sample_range(90, 900, 1.0, 10) == (3, 9)


def test_segment_entirely_past_end_clamps_to_last_frame():
    i0, i1 = segment_sample_range(300, 400, 1.0, 10)
    assert i0 == i1 == 9            # vẫn chấm 1 khung, không bỏ ca


def test_very_short_segment_falls_back_to_its_middle_frame():
    # 5 frame @30fps = 0.17 s; ở 1 fps cả start và end đều floor về cùng chỉ số
    i0, i1 = segment_sample_range(100, 105, 1.0, 50)
    assert i0 == i1 == int(np.floor(102.5 / SOURCE_FPS))


@pytest.mark.parametrize("bad", [(0, 10, 0.0, 10), (-1, 10, 1.0, 10), (10, 5, 1.0, 10)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        segment_sample_range(*bad)


def test_empty_cache_returns_none():
    assert segment_sample_range(0, 10, 1.0, 0) is None


# --- chọn ca để chấm ---------------------------------------------------------------

def _anns():
    return {
        "Fighting001": Annotation("Fighting001", "Fighting", ((0, 300),)),
        "Abuse002": Annotation("Abuse002", "Abuse", ((0, 150), (600, 750))),
        "Normal001": Annotation("Normal001", "Normal", ()),          # không có đoạn bất thường
        "NotCached": Annotation("NotCached", "Arrest", ((0, 60),)),
    }


def test_build_cases_skips_normal_videos_and_missing_cache():
    cases = build_cases(_anns(), 1.0, {"Fighting001": 20, "Abuse002": 40, "Normal001": 60})
    vids = [a.video_id for a, _ in cases]
    assert "Normal001" not in vids          # typer không có lớp Normal -> không chấm
    assert "NotCached" not in vids          # chưa embed -> bỏ
    assert vids.count("Abuse002") == 2      # hai đoạn = hai ca


def test_build_cases_keeps_both_segments_of_one_video():
    cases = build_cases(_anns(), 1.0, {"Abuse002": 40})
    ranges = [(s, e, i0, i1) for _, (s, e, i0, i1) in cases]
    assert ranges == [(0, 150, 0, 5), (600, 750, 20, 25)]


# --- chạy typer trên từng ca (typer giả) -------------------------------------------

class StubTyper:
    """Typer giả: trả về đúng nhãn được chỉ định, để test phần chấm không phụ thuộc model."""

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = []

    def classify(self, frame_embs, aggregate="mean", top_k=3):
        self.calls.append((frame_embs.shape, aggregate))
        return TypingResult(self.predictions.pop(0), 0.9, {}, ())


def test_evaluate_cases_records_predictions_and_window_sizes():
    cases = build_cases(_anns(), 1.0, {"Fighting001": 20, "Abuse002": 40})
    embs = {"Fighting001": np.ones((20, 4), np.float32),
            "Abuse002": np.ones((40, 4), np.float32)}
    typer = StubTyper([EventType.FIGHTING, EventType.ABUSE, EventType.FIGHTING])

    results = evaluate_cases(cases, typer, embs.get)

    assert len(results) == 3
    assert results[0].correct and results[1].correct
    assert not results[2].correct                # đoán Fighting cho đoạn Abuse
    assert typer.calls[0] == ((11, 4), "mean")   # Fighting001: frame 0..300 -> 11 khung (0..10)
    assert results[0].start_s == 0.0 and results[0].end_s == 10.0


def test_evaluate_cases_skips_missing_embeddings():
    cases = build_cases(_anns(), 1.0, {"Fighting001": 20})
    results = evaluate_cases(cases, StubTyper([EventType.FIGHTING]), lambda v: None)
    assert results == []


def test_evaluate_cases_passes_aggregate_through():
    cases = build_cases(_anns(), 1.0, {"Fighting001": 20})[:1]
    typer = StubTyper([EventType.FIGHTING])
    evaluate_cases(cases, typer, lambda v: np.ones((20, 4), np.float32), aggregate="max")
    assert typer.calls[0][1] == "max"


# --- tổng hợp số liệu --------------------------------------------------------------

def _res(gt, pred, vid="v"):
    from scripts.eval_typer import CaseResult
    return CaseResult(vid, gt, pred, 0.9, 5, 0.0, 1.0)


def test_summarise_computes_strict_and_lenient():
    results = [
        _res(EventType.FIGHTING, EventType.FIGHTING),   # đúng
        _res(EventType.ABUSE, EventType.FIGHTING),      # sai strict, đúng lenient
        _res(EventType.ARSON, EventType.SHOOTING),      # sai cả hai
        _res(EventType.ARSON, EventType.ARSON),         # đúng
    ]
    s = summarise(results, n_classes=13)
    assert s["n_cases"] == 4 and s["n_videos"] == 1
    assert s["strict_correct"] == 2 and s["strict_accuracy"] == 0.5
    assert s["lenient_correct"] == 3 and s["lenient_accuracy"] == 0.75
    assert s["chance_level"] == pytest.approx(1 / 13)
    assert s["per_class"]["Arson"] == {"total": 2, "correct": 1}


def test_summarise_on_empty_returns_none_not_zero():
    s = summarise([], n_classes=13)
    assert s["n_cases"] == 0
    assert s["strict_accuracy"] is None and s["lenient_accuracy"] is None


def test_lenient_groups_are_symmetric():
    for gt, group in LENIENT_GROUPS.items():
        assert gt in group, f"{gt} phải nằm trong nhóm của chính nó"
        for other in group:
            assert gt in LENIENT_GROUPS[other], f"nhóm lệch: {gt} <-> {other}"


# --- câu kết luận (đơn vị 0..1, không phải %) --------------------------------------

def test_verdict_uses_the_same_units_as_accuracy():
    """Lỗi thật đã từng xảy ra: so accuracy 0..1 với chance đã nhân 100 -> luôn 'kém'."""
    from scripts.eval_typer import verdict_text
    s = {"n_cases": 200, "strict_accuracy": 1.0, "chance_level": 1 / 13}
    assert "TỐT HƠN" in verdict_text(s)


def test_verdict_refuses_to_conclude_on_a_tiny_sample():
    from scripts.eval_typer import verdict_text
    s = {"n_cases": 3, "strict_accuracy": 1.0, "chance_level": 1 / 13}
    assert "quá ít" in verdict_text(s)


def test_verdict_is_honest_when_at_or_below_chance():
    from scripts.eval_typer import verdict_text
    s = {"n_cases": 200, "strict_accuracy": 0.05, "chance_level": 1 / 13}
    assert "KHÔNG hơn" in verdict_text(s)


def test_verdict_middle_band_is_not_oversold():
    from scripts.eval_typer import verdict_text
    s = {"n_cases": 200, "strict_accuracy": 0.10, "chance_level": 1 / 13}
    assert "còn yếu" in verdict_text(s)


# --- CLI --------------------------------------------------------------------------

ANNOTATION_TEXT = """\
Fighting001_x264.mp4  Fighting  0  300  -1  -1
Abuse002_x264.mp4  Abuse  0  150  600  750
"""


def _write_annotation(tmp_path):
    p = tmp_path / "ann.txt"
    p.write_text(ANNOTATION_TEXT, encoding="utf-8")
    return p


def test_load_annotations_roundtrip(tmp_path):
    anns = load_annotations(_write_annotation(tmp_path))
    assert set(anns) == {"Fighting001_x264", "Abuse002_x264"}
    assert anns["Abuse002_x264"].event_type is EventType.ABUSE


def test_cli_dry_run_counts_segments_without_a_model(tmp_path, capsys):
    ann = _write_annotation(tmp_path)
    rc = main(["--annotation", str(ann), "--cache-dir", str(tmp_path / "cache"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sẽ chấm 0 đoạn" in out               # chưa có cache
    assert "Không nạp model" in out


def test_cli_dry_run_with_cached_videos(tmp_path, capsys):
    ann = _write_annotation(tmp_path)
    cache = tmp_path / "cache"
    save_embeddings(np.ones((20, 8), np.float32), "m", "Fighting001_x264", 1.0, cache)
    save_embeddings(np.ones((40, 8), np.float32), "m", "Abuse002_x264", 1.0, cache)

    rc = main(["--annotation", str(ann), "--cache-dir", str(cache), "--model", "m", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sẽ chấm 3 đoạn" in out               # 1 (Fighting) + 2 (Abuse)
    assert "có cache" in out


def test_cli_fails_clearly_when_no_cache(tmp_path, capsys):
    ann = _write_annotation(tmp_path)
    rc = main(["--annotation", str(ann), "--cache-dir", str(tmp_path / "empty"), "--model", "m"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "chưa embed" in (captured.out + captured.err)


def test_cli_writes_json_payload_shape(tmp_path):
    """Không chạy model: chỉ kiểm tra phần ghi JSON bằng cách gọi thẳng summarize->payload."""
    from scripts.eval_typer import CaseResult
    results = [CaseResult("v1", EventType.FIGHTING, EventType.FIGHTING, 0.9, 5, 0.0, 1.0)]
    s = summarise(results, 13)
    payload = json.loads(json.dumps({
        "task": "C3-typing", "summary": s,
        "cases": [{"video_id": r.video_id, "gt": r.gt.value, "pred": r.pred.value} for r in results],
    }, ensure_ascii=False))
    assert payload["summary"]["strict_accuracy"] == 1.0
    assert payload["cases"][0]["gt"] == "Fighting"
