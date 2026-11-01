# -*- coding: utf-8 -*-
"""Test C7e — `scripts/eval_vlm_typing.py`. Không GPU, không model, không video thật.

Phần dễ sai nhất không phải "đếm đúng mấy ca", mà là CHỌN CA và CHỌN NGUỒN NHÃN: giao với GT theo
giây, loại cửa sổ không giao, và đếm riêng từng lý do loại. Một bẫy riêng được khóa bằng test: nhãn
VLM phải đọc từ tier-2 THÔ, không phải `verify_results` trong `run_A4c` (đã bị typer ghi đè theo D-2).
"""
import json
from pathlib import Path

import pytest

from scripts.eval_vlm_typing import (
    CaseResult,
    TypingEvalError,
    build_cases,
    evaluate_cases,
    gt_overlap_s,
    main,
    pair_verdicts,
    segments_seconds,
    summarise,
    verdict_text,
)
from vad.datasets.ucf_crime import Annotation
from vad.results import RunResult
from vad.score.typing import TypingResult
from vad.types import EventType, VerifyResult, Window


# --- dụng cụ ---------------------------------------------------------------------

def _ann(video_id: str, label: str, segments=((0, 899),)) -> Annotation:
    return Annotation(video_id=video_id, label=label, segments=tuple(segments))


def _verify(is_anomaly: bool, event_type: EventType = EventType.FIGHTING) -> VerifyResult:
    return VerifyResult(is_anomaly=is_anomaly, event_type=event_type, caption="x", confidence=0.9)


def _run(video_id: str, windows, verdicts) -> RunResult:
    return RunResult(video_id=video_id, sample_fps=1.0, anomaly_scores=(0.0,),
                     candidate_windows=tuple(windows), verify_results=tuple(verdicts))


def _window(video_id: str, start: float, end: float, score: float = 1.5) -> Window:
    return Window(video_id=video_id, start_s=start, end_s=end, score=score)


def _case(gt: EventType, vlm: EventType, typer: EventType = EventType.UNKNOWN,
          video_id: str = "Fighting001_x264") -> CaseResult:
    return CaseResult(video_id=video_id, window_index=0, start_s=0.0, end_s=10.0, score=1.5,
                      overlap_s=10.0, gt=gt, vlm=vlm, typer=typer, typer_confidence=0.5)


# --- hình học thời gian -----------------------------------------------------------

def test_overlap_exact_and_partial():
    assert gt_overlap_s(10.0, 20.0, [(10.0, 20.0)]) == pytest.approx(10.0)
    assert gt_overlap_s(0.0, 10.0, [(5.0, 30.0)]) == pytest.approx(5.0)
    assert gt_overlap_s(0.0, 10.0, [(10.0, 20.0)]) == 0.0          # chạm biên, không giao


def test_overlapping_gt_segments_are_not_double_counted():
    # hai đoạn GT chồng nhau 5..15 và 10..20; cửa sổ 0..30 chỉ được tính 15 giây, không phải 20
    assert gt_overlap_s(0.0, 30.0, [(5.0, 15.0), (10.0, 20.0)]) == pytest.approx(15.0)


def test_overlap_never_exceeds_window_length():
    assert gt_overlap_s(5.0, 15.0, [(0.0, 100.0)]) == pytest.approx(10.0)


def test_invalid_window_raises():
    with pytest.raises(TypingEvalError):
        gt_overlap_s(20.0, 10.0, [(0.0, 30.0)])
    with pytest.raises(TypingEvalError):
        gt_overlap_s(float("nan"), 10.0, [(0.0, 30.0)])


def test_segments_converted_from_30fps_frames():
    # frame 30..59 đóng hai đầu -> [1.0, 2.0): frame 59 chiếm trọn 1/30 giây cuối
    ann = _ann("Arrest001_x264", "Arrest", [(30, 59)])
    seg = segments_seconds(ann)
    assert seg[0][0] == pytest.approx(1.0)
    assert seg[0][1] == pytest.approx(60 / 30)


def test_window_starting_on_the_last_gt_frame_only_grazes_the_incident():
    # GT 105..180 frame = [3.5, 6.033); cửa sổ [6, 26] chỉ chạm 33 ms -> không phải đơn vị chấm
    vid = "Explosion027_x264"
    run = _run(vid, [_window(vid, 6, 26)], [_verify(True, EventType.EXPLOSION)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Explosion", [(105, 180)])},
                                  {vid: (_verify(True, EventType.EXPLOSION),)})
    assert cases == []
    assert excluded["confirmed_overlap_too_small"] == 1


def test_window_with_one_full_sample_inside_gt_is_kept():
    # GT 600..900 frame = [20.0, 30.033); cửa sổ [28, 29] nằm trọn 1 s trong GT -> được chấm
    vid = "Arson035_x264"
    run = _run(vid, [_window(vid, 28, 29)], [_verify(True, EventType.ARSON)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Arson", [(600, 900)])},
                                  {vid: (_verify(True, EventType.ARSON),)})
    assert len(cases) == 1
    assert cases[0].overlap_s == pytest.approx(1.0)
    assert excluded["confirmed_overlap_too_small"] == 0


# --- chọn ca ----------------------------------------------------------------------

def test_confirmed_window_inside_gt_becomes_a_case():
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 10, 20)], [_verify(True, EventType.FIGHTING)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Fighting")},
                                  {vid: (_verify(True, EventType.FIGHTING),)})
    assert len(cases) == 1
    case = cases[0]
    assert case.gt is EventType.FIGHTING
    assert case.vlm is EventType.FIGHTING
    assert case.overlap_s == pytest.approx(10.0)
    assert case.typer is EventType.UNKNOWN          # chưa chấm typer
    assert excluded["confirmed_without_gt_overlap"] == 0


def test_vlm_label_comes_from_raw_verdicts_not_from_the_typer_relabelled_run_file():
    # Bẫy D-2: trong run_A4c, verify_results.event_type đã bị typer ghi đè (Arrest),
    # nhãn THÔ của VLM là Burglary -> ca phải giữ Burglary.
    vid = "Arrest001_x264"
    run = _run(vid, [_window(vid, 10, 20)], [_verify(True, EventType.ARREST)])
    cases, _ = build_cases({vid: run}, {vid: _ann(vid, "Arrest")},
                           {vid: (_verify(True, EventType.BURGLARY),)})
    assert cases[0].vlm is EventType.BURGLARY


def test_confirmed_window_without_gt_overlap_is_excluded_and_counted():
    # video bất thường nhưng cửa sổ nằm ngoài đoạn GT (GT là 0..30 s, cửa sổ 100..110 s)
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 100, 110)], [_verify(True)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Fighting", [(0, 899)])},
                                  {vid: (_verify(True),)})
    assert cases == []
    assert excluded["confirmed_without_gt_overlap"] == 1


def test_rejected_window_is_not_a_case_but_is_counted():
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 10, 20)], [_verify(False, EventType.NORMAL)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Fighting")},
                                  {vid: (_verify(False, EventType.NORMAL),)})
    assert cases == []
    assert excluded["rejected_windows"] == 1


def test_confirmed_window_on_normal_video_has_no_label_to_compare():
    vid = "Normal_Videos_001_x264"
    run = _run(vid, [_window(vid, 10, 20)], [_verify(True, EventType.BURGLARY)])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Normal", ())},
                                  {vid: (_verify(True, EventType.BURGLARY),)})
    assert cases == []
    assert excluded["confirmed_on_normal_video"] == 1


def test_windows_without_any_verdict_are_counted_not_silently_dropped():
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 10, 20)], [])
    cases, excluded = build_cases({vid: run}, {vid: _ann(vid, "Fighting")}, {})
    assert cases == []
    assert excluded["windows_without_verdict"] == 1


def test_video_without_result_file_is_counted():
    vid = "Fighting001_x264"
    cases, excluded = build_cases({}, {vid: _ann(vid, "Fighting")}, {})
    assert cases == []
    assert excluded["videos_without_results"] == 1


def test_mismatched_verdict_count_is_an_error_not_a_zip_truncation():
    vid = "Fighting001_x264"
    run = _run(vid, [_window(vid, 10, 20), _window(vid, 30, 40)], [_verify(True), _verify(True)])
    with pytest.raises(TypingEvalError):
        build_cases({vid: run}, {vid: _ann(vid, "Fighting")}, {vid: (_verify(True),)})


def test_nan_score_is_rejected_not_averaged():
    vid = "Fighting001_x264"
    bad = Window(video_id=vid, start_s=10.0, end_s=20.0, score=float("nan"))
    run = _run(vid, [bad], [_verify(True)])
    with pytest.raises(TypingEvalError):
        build_cases({vid: run}, {vid: _ann(vid, "Fighting")}, {vid: (_verify(True),)})


# --- ghép cửa sổ tier-2 thô với cửa sổ của C7d ------------------------------------

def test_pair_verdicts_accepts_same_times_and_rejects_shifted_ones():
    vid = "Fighting001_x264"
    mine = [_window(vid, 10, 20)]
    assert pair_verdicts(vid, mine, [_window(vid, 10, 20)]) is None
    with pytest.raises(TypingEvalError):
        pair_verdicts(vid, mine, [_window(vid, 12, 22)])
    with pytest.raises(TypingEvalError):
        pair_verdicts(vid, mine, [])


# --- chấm & tổng hợp --------------------------------------------------------------

def test_perfect_typer_scores_1_0():
    cases = [_case(EventType.FIGHTING, EventType.ROBBERY)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.FIGHTING, 0.9, {}, ()))
    summary = summarise(results)
    assert summary["typer"]["accuracy"] == 1.0
    assert summary["vlm"]["accuracy"] == 0.0


def test_always_wrong_typer_scores_0_0():
    cases = [_case(EventType.FIGHTING, EventType.FIGHTING)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.ARSON, 0.9, {}, ()))
    assert summarise(results)["typer"]["accuracy"] == 0.0


def test_non_default_class_is_scored_by_its_own_name():
    # RoadAccidents là lớp hay bị gõ sai ("road accident"); phải khớp qua EventType.from_label
    ann = _ann("RoadAccidents001_x264", "RoadAccidents")
    assert ann.event_type is EventType.ROAD_ACCIDENT
    cases = [_case(EventType.ROAD_ACCIDENT, EventType.ROAD_ACCIDENT)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.ROAD_ACCIDENT, 0.8, {}, ()))
    summary = summarise(results)
    assert summary["vlm"]["correct"] == 1
    assert summary["typer"]["correct"] == 1
    assert summary["vlm"]["per_class"]["RoadAccidents"]["total"] == 1


def test_no_cases_gives_none_accuracy_not_zero():
    summary = summarise([])
    assert summary["n_cases"] == 0
    assert summary["vlm"]["accuracy"] is None
    assert summary["typer"]["accuracy"] is None


def test_chance_level_is_one_over_thirteen_and_normal_share_counted():
    cases = [_case(EventType.FIGHTING, EventType.NORMAL), _case(EventType.FIGHTING, EventType.UNKNOWN),
             _case(EventType.FIGHTING, EventType.FIGHTING)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.FIGHTING, 0.5, {}, ()))
    summary = summarise(results, n_classes=13)
    assert summary["chance_level"] == pytest.approx(1 / 13)
    assert summary["vlm_normal_or_unknown"] == 2
    assert summary["vlm_normal_or_unknown_share"] == pytest.approx(2 / 3)


def test_lenient_group_does_not_affect_strict_accuracy():
    # Assault bị gọi thành Fighting: strict sai, lenient đúng
    cases = [_case(EventType.ASSAULT, EventType.FIGHTING)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.SHOOTING, 0.5, {}, ()))
    summary = summarise(results)
    assert summary["vlm"]["accuracy"] == 0.0
    assert summary["vlm"]["lenient_accuracy"] == 1.0


def test_verdict_text_refuses_to_conclude_on_a_tiny_sample():
    summary = summarise([_case(EventType.FIGHTING, EventType.FIGHTING)], n_classes=13)
    assert "quá ít" in verdict_text(summary)


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    # 4 video, mỗi video 1 ca: 3 ca VLM đúng -> point 0.75, KTC phải phủ 0.75 và tái lập được
    cases = [_case(EventType.FIGHTING, EventType.FIGHTING, video_id=f"v{i}") for i in range(3)]
    cases.append(_case(EventType.FIGHTING, EventType.ARSON, video_id="v3"))
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.FIGHTING, 0.5, {}, ()))
    summary = summarise(results)
    ci = summary["vlm"]["accuracy_ci95"]
    assert ci is not None
    assert ci["point"] == pytest.approx(0.75)
    assert ci["lo"] <= 0.75 <= ci["hi"]
    assert ci["n_videos"] == 4
    assert summarise(results)["vlm"]["accuracy_ci95"] == ci        # seed cố định


def test_bootstrap_ci_is_none_for_a_single_video():
    # một video nhiều cửa sổ: không resample được theo video, phải trả None chứ không bịa KTC hẹp
    cases = [_case(EventType.FIGHTING, EventType.FIGHTING, video_id="same") for _ in range(3)]
    results = evaluate_cases(cases, lambda c: TypingResult(EventType.FIGHTING, 0.5, {}, ()))
    assert summarise(results)["vlm"]["accuracy_ci95"] is None


def test_display_path_is_repo_relative_not_an_absolute_machine_path():
    from scripts.eval_vlm_typing import ROOT, display_path

    assert display_path(ROOT / "results" / "run_A4c") == "results/run_A4c"
    assert display_path(Path("datasets/ucf_crime/x.txt")) == "datasets/ucf_crime/x.txt"
    outside = display_path(Path(r"C:\somewhere\else\file.txt"))
    assert outside == "C:/somewhere/else/file.txt" or outside == r"C:\somewhere\else\file.txt"


# --- CLI dry-run ------------------------------------------------------------------

def test_dry_run_counts_without_writing(tmp_path):
    ann_path = tmp_path / "ann.txt"
    ann_path.write_text("Fighting001_x264 Fighting 0 30\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    run = _run("Fighting001_x264", [_window("Fighting001_x264", 0, 10)],
               [_verify(True, EventType.FIGHTING)])
    (results / "Fighting001_x264.json").write_text(json.dumps(run.to_dict(), ensure_ascii=False),
                                                   encoding="utf-8")
    out = tmp_path / "should_not_exist.json"

    code = main(["--annotation", str(ann_path), "--results-dir", str(results),
                 "--verify-dir", str(tmp_path / "no_tier2"), "--out", str(out), "--dry-run"])
    assert code == 0
    assert not out.exists()


def test_limit_without_explicit_out_refuses_to_overwrite_the_tracked_result(tmp_path):
    # --limit là để thử nhanh; nếu cho ghi vào results/vlm_typing_eval.json thì một lần thử sẽ đè
    # kết quả đầy đủ đang được track trong git, và không ai thấy con số đổi.
    ann_path = tmp_path / "ann.txt"
    ann_path.write_text("Fighting001_x264 Fighting 0 30\n", encoding="utf-8")
    code = main(["--annotation", str(ann_path), "--limit", "1"])
    assert code == 2
