"""Hợp đồng giữa các track: bản giả PHẢI thoả Protocol, bản thật chưa xong PHẢI nổ rõ ràng (không trả rác)."""
import csv
import numpy as np
import pytest

from scripts.precompute_embeddings import main as precompute_main, plan_work, read_manifest
from vad.embed.cache import save_embeddings
from vad.embed.siglip import SigLIPEmbedder
from vad.fakes import FakeEmbedder, FakeVerifier
from vad.graph.policy import GraphPolicy
from vad.interfaces import AlertPolicy, Embedder, Verifier
from vad.types import EventType, VideoRef
from vad.verify.vlm import VLMVerifier


def test_fakes_and_real_policy_satisfy_protocols():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeVerifier(), Verifier)
    assert isinstance(GraphPolicy(), AlertPolicy)
    assert isinstance(SigLIPEmbedder(), Embedder)       # chữ ký đúng dù chưa cài thân hàm
    assert isinstance(VLMVerifier("x"), Verifier)


def test_fake_embedder_is_deterministic_unit_norm():
    e = FakeEmbedder(16)
    a, b = e.embed_texts(["hello"]), e.embed_texts(["hello"])
    assert np.array_equal(a, b) and np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-6)
    assert not np.array_equal(a, e.embed_texts(["world"]))
    assert e.embed_images([]).shape == (0, 16)


def test_unimplemented_real_backends_fail_loudly():
    with pytest.raises(NotImplementedError):
        SigLIPEmbedder().embed_images([])
    with pytest.raises(NotImplementedError):
        VLMVerifier("x").verify([], {})


def test_video_ref_is_frozen():
    v = VideoRef("a", "p", "ucf_crime", "test")
    with pytest.raises(Exception):
        v.video_id = "b"  # type: ignore[misc]


# ---------- precompute plan (dry-run, không cần GPU) ----------
def _manifest(tmp_path, ids):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "path", "dataset", "split"])
        for i in ids:
            w.writerow([i, f"/x/{i}.mp4", "ucf_crime", "test"])
    return p


def test_plan_work_skips_cached_and_shards_are_disjoint(tmp_path):
    ids = [f"v{i}" for i in range(40)]
    refs = read_manifest(_manifest(tmp_path, ids))
    cache = tmp_path / "c"
    todo_all = []
    for s in range(3):
        todo, done = plan_work(refs, s, 3, "m", 1.0, cache)
        assert done == []
        todo_all += [r.video_id for r in todo]
    assert sorted(todo_all) == sorted(ids)                 # phủ đủ, không trùng
    first = plan_work(refs, 0, 3, "m", 1.0, cache)[0][0]
    save_embeddings(np.ones((1, 4), np.float32), "m", first.video_id, 1.0, cache)
    todo2, done2 = plan_work(refs, 0, 3, "m", 1.0, cache)
    assert first.video_id in [r.video_id for r in done2] and first.video_id not in [r.video_id for r in todo2]


def test_manifest_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="trùng"):
        read_manifest(_manifest(tmp_path, ["a", "a"]))


def test_precompute_dry_run_cli_and_real_run_refuses(tmp_path, capsys):
    m = _manifest(tmp_path, ["a", "b", "c"])
    assert precompute_main(["--manifest", str(m), "--shard", "1/1", "--dry-run",
                            "--cache-dir", str(tmp_path / "c")]) == 0
    assert "3 cần làm" in capsys.readouterr().out
    with pytest.raises(NotImplementedError):
        precompute_main(["--manifest", str(m), "--cache-dir", str(tmp_path / "c")])


def test_event_type_covers_all_policy_events():
    from vad.graph.policy import load_policy
    for name in load_policy()["events"]:
        assert EventType.from_label(name).value == name
