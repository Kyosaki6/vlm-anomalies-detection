# -*- coding: utf-8 -*-
"""Test scripts/fetch_demo.py: offline, downloader giả. Canh file lạ, SHA lệch, thiếu file, ghi đè nhầm."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import fetch_demo as fd  # noqa: E402

FILES = {"a.mp4": b"aaa", "b.mp4": b"bbbb"}
EXPECTED = {k: hashlib.sha256(v).hexdigest() for k, v in FILES.items()}

def _dl(files: dict[str, bytes], sub: str = ""):
    def run(folder_id: str, out: Path) -> None:
        d = out / sub if sub else out
        d.mkdir(parents=True, exist_ok=True)
        for n, data in files.items():
            (d / n).write_bytes(data)
    return run

def test_fetch_writes_verified_files(tmp_path):
    out = fd.fetch(tmp_path / "own", "x", downloader=_dl(FILES, sub="folder"), expected=EXPECTED)
    assert sorted(p.name for p in out) == ["a.mp4", "b.mp4"]
    fd.verify_dir(tmp_path / "own", EXPECTED)

def test_sha_mismatch_rejected_and_nothing_written(tmp_path):
    bad = {**FILES, "b.mp4": b"tampered"}
    with pytest.raises(fd.FetchDemoError, match="SHA-256 lệch"):
        fd.fetch(tmp_path / "own", "x", downloader=_dl(bad), expected=EXPECTED)
    assert not list((tmp_path / "own").glob("*.mp4")) if (tmp_path / "own").exists() else True

def test_missing_file_rejected(tmp_path):
    with pytest.raises(fd.FetchDemoError, match="thiếu b.mp4"):
        fd.fetch(tmp_path / "own", "x", downloader=_dl({"a.mp4": b"aaa"}), expected=EXPECTED)

def test_unexpected_file_rejected(tmp_path):
    with pytest.raises(fd.FetchDemoError, match="File lạ"):
        fd.fetch(tmp_path / "own", "x", downloader=_dl({**FILES, "evil.mp4": b"x"}), expected=EXPECTED)

def test_existing_good_file_kept_existing_bad_file_needs_force(tmp_path):
    dest = tmp_path / "own"
    dest.mkdir()
    (dest / "a.mp4").write_bytes(b"aaa")
    out = fd.fetch(dest, "x", downloader=_dl(FILES), expected=EXPECTED)
    assert [p.name for p in out] == ["b.mp4"]
    (dest / "a.mp4").write_bytes(b"local edit")
    with pytest.raises(fd.FetchDemoError, match="--force"):
        fd.fetch(dest, "x", downloader=_dl(FILES), expected=EXPECTED)
    assert (dest / "a.mp4").read_bytes() == b"local edit"
    fd.fetch(dest, "x", force=True, downloader=_dl(FILES), expected=EXPECTED)
    assert (dest / "a.mp4").read_bytes() == b"aaa"

def test_empty_download_message(tmp_path):
    with pytest.raises(fd.FetchDemoError, match="thiếu"):
        fd.fetch(tmp_path / "own", "x", downloader=_dl({}), expected=EXPECTED)

def test_pinned_table_matches_manifest():
    import csv
    root = Path(__file__).resolve().parents[2]
    ids = {r["video_id"] + ".mp4" for r in csv.DictReader(open(root / "manifests" / "own_demo.csv", encoding="utf-8-sig"))}
    assert ids == set(fd.EXPECTED_SHA256)
