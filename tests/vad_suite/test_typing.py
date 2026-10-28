# -*- coding: utf-8 -*-
"""Test event typer zero-shot (track T4). Không cần GPU/model: embedder giả hoặc bảng tra."""
import numpy as np
import pytest

from vad.fakes import FakeEmbedder
from vad.score.typing import ZeroShotEventTyper, load_typing_prompts
from vad.types import EventType


class TableEmbedder:
    """Embedder tra bảng: text phải đăng ký trước, ảnh thì trả nguyên vector đưa vào.

    Dùng để kiểm soát CHÍNH XÁC hình học của test — nếu template trong yaml đổi, test sẽ KeyError
    thay vì âm thầm đo một thứ khác.
    """

    model_id = "table-embedder"

    def __init__(self, texts: dict[str, list[float]]):
        self._texts = {k: np.asarray(v, dtype=np.float64) for k, v in texts.items()}

    def embed_texts(self, texts):
        return np.stack([self._texts[t] for t in texts])

    def embed_images(self, frames):
        return np.stack([np.asarray(f, dtype=np.float64) for f in frames])


@pytest.fixture
def tiny_prompts(tmp_path):
    """prompts.yaml tối giản: template biết trước -> text đăng ký trong bảng là đoán được."""
    f = tmp_path / "prompts.yaml"
    f.write_text(
        "event_typing:\n"
        "  template: 'frame of {label}'\n"
        "  labels:\n"
        "    Fighting: 'fists'\n"
        "    Shoplifting: 'a shop'\n",
        encoding="utf-8",
    )
    return f


def _typer(tiny_prompts, **kwargs):
    embedder = TableEmbedder({"frame of fists": [1.0, 0.0], "frame of a shop": [0.0, 1.0]})
    return ZeroShotEventTyper(embedder, prompts_path=tiny_prompts, **kwargs)


def test_picks_class_whose_prompt_matches_the_frame(tiny_prompts):
    typer = _typer(tiny_prompts)
    r = typer.classify(np.array([[1.0, 0.0]]))
    assert r.event_type is EventType.FIGHTING
    assert r.scores["Fighting"] > r.scores["Shoplifting"]


def test_scores_are_a_distribution(tiny_prompts):
    typer = _typer(tiny_prompts)
    r = typer.classify(np.array([[0.3, 0.7]]))
    assert abs(sum(r.scores.values()) - 1.0) < 1e-9
    assert 0.0 <= r.confidence <= 1.0


def test_mean_and_max_can_disagree(tiny_prompts):
    """1 khung rất giống Shoplifting giữa 5 khung hơi giống Fighting.

    mean -> Fighting (đa số), max -> Shoplifting (đỉnh). Đây là hai cách đọc cửa sổ khác nhau,
    nên typer phải cho chọn, và test chốt rằng chúng thật sự khác nhau.
    """
    typer = _typer(tiny_prompts)
    frames = np.array([[0.99, 0.14]] * 5 + [[0.06, 0.998]])
    assert typer.classify(frames, aggregate="mean").event_type is EventType.FIGHTING
    assert typer.classify(frames, aggregate="max").event_type is EventType.SHOPLIFTING


def test_include_normal_adds_normal_class(tiny_prompts):
    embedder = TableEmbedder({
        "frame of fists": [1.0, 0.0],
        "frame of a shop": [0.0, 1.0],
        "frame of a normal scene where nothing unusual is happening": [-1.0, -1.0],
    })
    typer = ZeroShotEventTyper(embedder, prompts_path=tiny_prompts, include_normal=True)
    r = typer.classify(np.array([[-1.0, -1.0]]))
    assert r.event_type is EventType.NORMAL
    assert set(r.scores) == {"Fighting", "Shoplifting", "Normal"}


def test_top_k_is_sorted_and_truncated(tiny_prompts):
    typer = _typer(tiny_prompts)
    r = typer.classify(np.array([[1.0, 0.0]]), top_k=1)
    assert len(r.top_k) == 1 and r.top_k[0][0] is EventType.FIGHTING
    r3 = typer.classify(np.array([[1.0, 0.0]]), top_k=99)
    assert len(r3.top_k) == 2                                        # không vượt số lớp


def test_unknown_label_name_is_rejected(tiny_prompts):
    with pytest.raises(ValueError, match="không có trong prompts.yaml"):
        ZeroShotEventTyper(TableEmbedder({}), prompts_path=tiny_prompts, labels=["Fightting"])


def test_label_that_maps_to_no_event_type_is_rejected(tiny_prompts):
    """Tên lớp lạ trong yaml -> raise ngay ở __init__, không để âm thầm thành UNKNOWN."""
    bad = tiny_prompts.parent / "bad.yaml"
    bad.write_text("event_typing:\n  template: 'x {label}'\n  labels: {Nonsense: 'y'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không map được"):
        ZeroShotEventTyper(TableEmbedder({"x y": [1.0, 0.0]}), prompts_path=bad)


@pytest.mark.parametrize("bad", [np.empty((0, 2)), np.empty((0,))])
def test_empty_frames_raise(tiny_prompts, bad):
    with pytest.raises(ValueError, match="rỗng"):
        _typer(tiny_prompts).classify(bad)


def test_dimension_mismatch_raises(tiny_prompts):
    with pytest.raises(ValueError, match="không khớp"):
        _typer(tiny_prompts).classify(np.array([[1.0, 0.0, 0.0]]))


def test_bad_aggregate_raises(tiny_prompts):
    with pytest.raises(ValueError, match="aggregate"):
        _typer(tiny_prompts).classify(np.array([[1.0, 0.0]]), aggregate="median")


def test_missing_event_typing_block_raises(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("detection: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event_typing"):
        load_typing_prompts(f)


def test_runs_on_real_config_with_fake_embedder_without_gpu():
    """Config thật + FakeEmbedder: 13 lớp UCF, chạy CPU, deterministic, không tải model."""
    typer = ZeroShotEventTyper(FakeEmbedder(dim=32))
    frames = FakeEmbedder(dim=32).embed_images([np.arange(4), np.arange(4) + 1])
    r1 = typer.classify(frames)
    r2 = typer.classify(frames)
    assert len(r1.scores) == 13
    assert r1.event_type is r2.event_type and r1.confidence == r2.confidence
    assert r1.event_type is not EventType.NORMAL      # 13 lớp bất thường, không có Normal mặc định
