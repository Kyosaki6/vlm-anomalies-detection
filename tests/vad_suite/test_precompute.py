# -*- coding: utf-8 -*-
"""Test đường chạy THẬT của `scripts/precompute_embeddings.py` — không cần GPU/model.

Dùng FakeEmbedder + video tự sinh bằng OpenCV. Mục tiêu: chứng minh video -> cache .npy hoạt động
đúng (số khung, resume, không nạp cả video vào RAM), để lỗi còn lại chỉ có thể là model.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from precompute_embeddings import main, plan_work, process_video, read_manifest  # noqa: E402
from vad.embed.cache import cache_path, has_cached, load_embeddings  # noqa: E402
from vad.fakes import FakeEmbedder  # noqa: E402
from vad.types import VideoRef  # noqa: E402


@pytest.fixture()
def video(tmp_path):
    """3 giây @30 fps = 90 khung."""
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not w.isOpened():
        pytest.skip("OpenCV build này không ghi được mp4v")
    for i in range(90):
        w.write(np.full((48, 64, 3), 40 + 60 * (i // 30), dtype=np.uint8))
    w.release()
    return path


def ref(video, vid="v1"):
    return VideoRef(vid, str(video), "ucf_crime", "test", n_frames=90, fps=30.0)


class BatchCountingEmbedder(FakeEmbedder):
    """FakeEmbedder + đếm số lượt gọi, để chứng minh dữ liệu được xử lý theo lô."""

    def __init__(self, dim=16, batch_size=8):
        super().__init__(dim=dim)
        self.batch_size = batch_size
        self.calls = []

    def embed_images(self, frames):
        self.calls.append(len(frames))
        assert len(frames) <= self.batch_size, "vượt batch_size -> có thể đã nạp cả video vào RAM"
        return super().embed_images(frames)


# --- process_video ----------------------------------------------------------------

def test_video_becomes_cached_npy(video, tmp_path):
    e = BatchCountingEmbedder()
    n = process_video(ref(video), e, 1.0, tmp_path)
    assert n == 3                                        # 3 giây @1 fps
    assert has_cached(e.model_id, "v1", 1.0, tmp_path)
    embs = load_embeddings(e.model_id, "v1", 1.0, tmp_path)
    assert embs.shape == (3, 16)


def test_streams_in_batches_never_holds_whole_video(video, tmp_path):
    """batch_size=1 -> phải gọi đúng 3 lần cho 3 khung, mỗi lần 1 khung."""
    e = BatchCountingEmbedder(batch_size=1)
    process_video(ref(video), e, 1.0, tmp_path)
    assert e.calls == [1, 1, 1]


def test_half_fps_halves_the_frames(video, tmp_path):
    e = BatchCountingEmbedder()
    assert process_video(ref(video), e, 0.5, tmp_path) == 2


def test_cache_is_resumable_and_not_recomputed(video, tmp_path):
    e = BatchCountingEmbedder()
    process_video(ref(video), e, 1.0, tmp_path)
    r = ref(video)
    todo, done = plan_work([r], 0, 1, e.model_id, 1.0, tmp_path)
    assert todo == [] and done == [r]


def test_cache_path_differs_per_fps(video, tmp_path):
    e = BatchCountingEmbedder()
    process_video(ref(video), e, 1.0, tmp_path)
    assert has_cached(e.model_id, "v1", 1.0, tmp_path)
    assert not has_cached(e.model_id, "v1", 2.0, tmp_path)


def test_missing_video_raises_and_does_not_write_cache(tmp_path):
    e = BatchCountingEmbedder()
    with pytest.raises(FileNotFoundError):
        process_video(ref(tmp_path / "nope.mp4"), e, 1.0, tmp_path)
    assert not cache_path(e.model_id, "v1", 1.0, tmp_path).exists()


# --- manifest + CLI ---------------------------------------------------------------

def test_read_manifest_parses_and_rejects_duplicates(video, tmp_path):
    m = tmp_path / "m.csv"
    m.write_text(f"video_id,path,dataset,split\nv1,{video.as_posix()},ucf_crime,test\n", encoding="utf-8")
    refs = read_manifest(m)
    assert refs[0].video_id == "v1" and refs[0].fps == 30.0 and refs[0].n_frames is None

    m.write_text(f"video_id,path,dataset,split\nv1,{video.as_posix()},ucf_crime,test\n"
                 f"v1,{video.as_posix()},ucf_crime,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        read_manifest(m)


def test_dry_run_lists_work_without_touching_the_model(video, tmp_path, capsys):
    m = tmp_path / "m.csv"
    m.write_text(f"video_id,path,dataset,split\nv1,{video.as_posix()},ucf_crime,test\n", encoding="utf-8")
    rc = main(["--manifest", str(m), "--cache-dir", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "[todo] v1" in out and "cần làm" in out

# --- nhiều model, MỘT lượt giải mã (B3 / G8) ----------------------------------------

class NamedFake(BatchCountingEmbedder):
    def __init__(self, model_id, dim=16, batch_size=8):
        super().__init__(dim=dim, batch_size=batch_size)
        self.model_id = model_id

def test_two_models_share_one_decode_pass(video, tmp_path, monkeypatch):
    import precompute_embeddings as pe

    decodes = []
    real = pe.iter_sampled_frames
    monkeypatch.setattr(pe, "iter_sampled_frames", lambda *a, **k: (decodes.append(1), real(*a, **k))[1])

    a, b = NamedFake("model/a", dim=16), NamedFake("model/b", dim=8)
    n = pe.process_video_multi(ref(video), [a, b], 1.0, tmp_path)
    assert n == 3 and len(decodes) == 1                       # giải mã đúng 1 lần cho cả 2 model
    assert load_embeddings("model/a", "v1", 1.0, tmp_path).shape == (3, 16)
    assert load_embeddings("model/b", "v1", 1.0, tmp_path).shape == (3, 8)   # dim khác nhau vẫn ổn

def test_multi_resumes_per_model(video, tmp_path):
    from precompute_embeddings import process_video_multi

    a, b = NamedFake("model/a"), NamedFake("model/b")
    process_video_multi(ref(video), [a], 1.0, tmp_path)       # chỉ a có cache
    a.calls.clear()
    assert process_video_multi(ref(video), [a, b], 1.0, tmp_path) == 3
    assert a.calls == []                                      # a không bị tính lại
    assert b.calls                                            # b được tính
    assert process_video_multi(ref(video), [a, b], 1.0, tmp_path) == 0   # cả hai đã có -> không làm gì

def test_multi_writes_no_cache_if_any_model_fails(video, tmp_path):
    from precompute_embeddings import process_video_multi

    class Boom(NamedFake):
        def embed_images(self, frames):
            raise RuntimeError("OOM")

    a, b = NamedFake("model/a"), Boom("model/b")
    with pytest.raises(RuntimeError, match="OOM"):
        process_video_multi(ref(video), [a, b], 1.0, tmp_path)
    assert not has_cached("model/a", "v1", 1.0, tmp_path)     # không để cache lệch nhau giữa các model

def test_multi_rejects_duplicate_model_ids(video, tmp_path):
    from precompute_embeddings import process_video_multi

    with pytest.raises(ValueError, match="ghi đè"):
        process_video_multi(ref(video), [NamedFake("m"), NamedFake("m")], 1.0, tmp_path)

def test_plan_work_with_two_models_needs_both_cached(video, tmp_path):
    from precompute_embeddings import process_video_multi

    r = ref(video)
    process_video_multi(r, [NamedFake("model/a")], 1.0, tmp_path)
    todo, done = plan_work([r], 0, 1, ["model/a", "model/b"], 1.0, tmp_path)
    assert todo == [r] and done == []
    todo, done = plan_work([r], 0, 1, "model/a", 1.0, tmp_path)
    assert todo == [] and done == [r]

def test_dry_run_accepts_repeated_model_flag(video, tmp_path, capsys):
    m = tmp_path / "m.csv"
    m.write_text(f"video_id,path,dataset,split\nv1,{video.as_posix()},ucf_crime,test\n", encoding="utf-8")
    rc = main(["--manifest", str(m), "--cache-dir", str(tmp_path), "--dry-run",
               "--model", "google/siglip2-base-patch16-384", "--model", "openai/clip-vit-base-patch32"])
    assert rc == 0 and "[todo] v1" in capsys.readouterr().out

def test_fp16_cache_roundtrip_keeps_cosine_above_0_999(tmp_path):
    """Cache lưu fp16; load_embeddings chuẩn hoá lại. Cosine với bản fp32 gốc phải > 0.999."""
    from vad.embed.cache import save_embeddings

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    save_embeddings(x, "m", "v", 1.0, tmp_path)
    y = load_embeddings("m", "v", 1.0, tmp_path)
    assert (np.sum(x * y, axis=1) > 0.999).all()
