import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vad.io.video import iter_sampled_frames, probe_video, sample_indices  # noqa: E402


# ---------- thuần số học: không cần codec ----------
def test_sample_indices_30fps_to_1fps():
    assert sample_indices(90, 30.0, 1.0) == [0, 30, 60]


def test_sample_indices_covers_partial_last_second_and_clamps():
    idx = sample_indices(100, 30.0, 1.0)         # ceil(100/30) = 4 mẫu
    assert idx == [0, 30, 60, 90]
    assert max(sample_indices(10, 30.0, 2.0)) <= 9


def test_sample_indices_matches_expand_to_frames_alignment():
    # Mẫu k phải nằm trong khoảng frame mà expand_to_frames gán cho mẫu k.
    from vad.temporal.ops import expand_to_frames
    n, fps, sf = 95, 30.0, 1.0
    owner = expand_to_frames(np.arange(len(sample_indices(n, fps, sf)), dtype=float), sf, fps, n)
    for k, f in enumerate(sample_indices(n, fps, sf)):
        assert owner[f] == k


def test_sample_indices_validation_and_empty():
    assert sample_indices(0, 30.0, 1.0) == []
    with pytest.raises(ValueError):
        sample_indices(10, 0.0, 1.0)
    with pytest.raises(ValueError):
        sample_indices(10, 30.0, -1.0)


def test_missing_file_and_garbage_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError):
        probe_video(bad)


# ---------- đọc video thật (tự sinh bằng OpenCV) ----------
@pytest.fixture()
def synthetic_video(tmp_path):
    """90 khung 30 fps = 3 giây; độ sáng tăng theo giây (50, 130, 210) để nhận ra mẫu nào là của giây nào."""
    path = tmp_path / "synth.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for i in range(90):
        writer.write(np.full((48, 64, 3), 50 + 80 * (i // 30), dtype=np.uint8))
    writer.release()
    return path


def test_probe_reads_frames_and_fps(synthetic_video):
    n, fps = probe_video(synthetic_video)
    assert fps == pytest.approx(30.0) and abs(n - 90) <= 1


def test_iter_sampled_frames_times_shapes_and_rgb_order(synthetic_video):
    out = list(iter_sampled_frames(synthetic_video, 1.0))
    assert [t for t, _ in out] == [0.0, 1.0, 2.0]
    assert all(f.shape == (48, 64, 3) and f.dtype == np.uint8 for _, f in out)
    means = [float(f.mean()) for _, f in out]
    assert means == sorted(means) and means[2] - means[0] > 100   # đúng thứ tự giây, không bị lẫn khung


def test_higher_sample_rate_gives_more_frames(synthetic_video):
    assert len(list(iter_sampled_frames(synthetic_video, 2.0))) == 6


def test_sample_fps_above_native_repeats_frames_without_hanging(synthetic_video):
    out = list(iter_sampled_frames(synthetic_video, 60.0))
    assert len(out) == 180 and out[1][0] == pytest.approx(1 / 60.0)


def test_color_channels_are_rgb_not_bgr(tmp_path):
    path = tmp_path / "red.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    red_bgr = np.zeros((32, 32, 3), np.uint8)
    red_bgr[..., 2] = 255                               # OpenCV: kênh 2 = đỏ
    for _ in range(10):
        writer.write(red_bgr)
    writer.release()
    _, frame = next(iter_sampled_frames(path, 1.0))
    assert frame[..., 0].mean() > 200 and frame[..., 2].mean() < 60   # RGB: kênh 0 phải là đỏ
