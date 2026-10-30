# -*- coding: utf-8 -*-
"""Test scripts/fetch_cache.py: offline, không mạng. Canh tar độc hại, SHA sai, ghi đè nhầm."""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import fetch_cache as fc  # noqa: E402


def _make_tar(path: Path, files: dict[str, bytes], extra=None) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            extra(tar)
    return path


def test_extract_ok(tmp_path):
    arc = _make_tar(tmp_path / "a.tar.gz", {"cache/embeddings/x.npy": b"1", "cache/prompts/p.npz": b"2"})
    dest = tmp_path / "out"
    assert fc.extract(arc, dest) == 2
    assert (dest / "cache/embeddings/x.npy").read_bytes() == b"1"


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "cache/embeddings/../../evil.txt",
                                  "other/file.txt", "cache/stride16/x.npy"])
def test_extract_rejects_hostile_or_out_of_scope(tmp_path, name):
    arc = _make_tar(tmp_path / "a.tar.gz", {name: b"x"})
    with pytest.raises(fc.FetchCacheError):
        fc.extract(arc, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_symlink(tmp_path):
    def add_link(tar):
        info = tarfile.TarInfo("cache/embeddings/link.npy")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    arc = _make_tar(tmp_path / "a.tar.gz", {"cache/embeddings/ok.npy": b"1"}, extra=add_link)
    with pytest.raises(fc.FetchCacheError):
        fc.extract(arc, tmp_path / "out")


def test_extract_refuses_overwrite_without_force(tmp_path):
    arc = _make_tar(tmp_path / "a.tar.gz", {"cache/embeddings/x.npy": b"new"})
    dest = tmp_path / "out"
    (dest / "cache/embeddings").mkdir(parents=True)
    (dest / "cache/embeddings/x.npy").write_bytes(b"old")
    with pytest.raises(fc.FetchCacheError):
        fc.extract(arc, dest)
    assert (dest / "cache/embeddings/x.npy").read_bytes() == b"old"       # không bị đụng tới
    fc.extract(arc, dest, force=True)
    assert (dest / "cache/embeddings/x.npy").read_bytes() == b"new"


def test_extract_rejects_empty_tar(tmp_path):
    arc = _make_tar(tmp_path / "a.tar.gz", {})
    with pytest.raises(fc.FetchCacheError):
        fc.extract(arc, tmp_path / "out")


def test_parse_sha256_file():
    h = "a" * 64
    assert fc.parse_sha256_file(f"{h}  file.tar.gz\n") == h
    assert fc.parse_sha256_file(h.upper()) == h
    for bad in ("", "xyz", "a" * 63, "g" * 64):
        with pytest.raises(fc.FetchCacheError):
            fc.parse_sha256_file(bad)


def test_main_sha_mismatch_does_not_extract(tmp_path, monkeypatch):
    arc = _make_tar(tmp_path / "src.tar.gz", {"cache/embeddings/x.npy": b"1"})

    def fake_download(url, dest):
        if url.endswith(".sha256"):
            dest.write_text("0" * 64 + "  src.tar.gz\n")
        else:
            dest.write_bytes(arc.read_bytes())

    monkeypatch.setattr(fc, "download", fake_download)
    out = tmp_path / "out"
    assert fc.main(["--dest", str(out), "--asset", "src.tar.gz"]) == 1
    assert not (out / "cache").exists()


def test_main_happy_path(tmp_path, monkeypatch):
    arc = _make_tar(tmp_path / "src.tar.gz", {"cache/embeddings/x.npy": b"1"})
    good = fc.sha256_of(arc)

    def fake_download(url, dest):
        if url.endswith(".sha256"):
            dest.write_text(f"{good}  src.tar.gz\n")
        else:
            dest.write_bytes(arc.read_bytes())

    monkeypatch.setattr(fc, "download", fake_download)
    out = tmp_path / "out"
    assert fc.main(["--dest", str(out), "--asset", "src.tar.gz"]) == 0
    assert (out / "cache/embeddings/x.npy").read_bytes() == b"1"
