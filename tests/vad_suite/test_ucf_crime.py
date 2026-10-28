import numpy as np
import pytest

from vad.datasets.ucf_crime import frame_labels, load_annotations, load_gt, parse_annotation_line
from vad.types import EventType


def test_parse_single_segment_drops_sentinel():
    ann = parse_annotation_line("Abuse028_x264.mp4  Abuse  165  240  -1  -1")
    assert ann.video_id == "Abuse028_x264"
    assert ann.segments == ((165, 240),)
    assert ann.event_type == EventType.ABUSE


def test_parse_two_instances():
    ann = parse_annotation_line("Fighting003_x264.mp4 Fighting 100 200 400 500")
    assert ann.segments == ((100, 200), (400, 500))


def test_parse_blank_line_is_none():
    assert parse_annotation_line("   \n") is None


@pytest.mark.parametrize("bad", ["OnlyName.mp4", "A.mp4 Abuse 1", "A.mp4 Abuse x y -1 -1", "A.mp4 Abuse 50 10 -1 -1"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_annotation_line(bad)


def test_frame_labels_inclusive_and_clipped():
    y = frame_labels(10, ((2, 4), (8, 99)))
    assert y.tolist() == [False, False, True, True, True, False, False, False, True, True]


def test_frame_labels_segment_past_end_is_ignored():
    assert not frame_labels(5, ((10, 20),)).any()


def test_frame_labels_rejects_negative_length():
    with pytest.raises(ValueError):
        frame_labels(-1, ())


def test_load_gt_unknown_video_is_normal(tmp_path):
    f = tmp_path / "ann.txt"
    f.write_text("\ufeffAbuse028_x264.mp4 Abuse 1 2 -1 -1\n", encoding="utf-8")
    anns = load_annotations(f)  # BOM được chịu
    assert load_gt(anns, "Abuse028_x264", 5).tolist() == [False, True, True, False, False]
    assert not load_gt(anns, "Normal_Videos_001_x264", 5).any()


def test_load_annotations_reports_line_number(tmp_path):
    f = tmp_path / "ann.txt"
    f.write_text("Abuse001.mp4 Abuse 1 2 -1 -1\nbroken\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_annotations(f)


def test_road_accident_label_variants():
    assert EventType.from_label("RoadAccidents") == EventType.ROAD_ACCIDENT
    assert EventType.from_label("road accident") == EventType.ROAD_ACCIDENT
    assert EventType.from_label("totally new") == EventType.UNKNOWN


def test_frame_labels_dtype():
    assert frame_labels(3, ()).dtype == np.bool_
