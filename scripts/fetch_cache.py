# -*- coding: utf-8 -*-
"""Tải cache embedding đã tính sẵn từ GitHub Release, kiểm SHA-256, giải nén vào `cache/`.

    python scripts/fetch_cache.py                       # tải bản mặc định (cache-v1)
    python scripts/fetch_cache.py --tag cache-v2 --asset vad-cache-xxx.tar.gz
    python scripts/fetch_cache.py --force               # ghi đè dù đã có file

Vì sao không commit cache vào git: ~106 MB, mỗi lần tính lại là thêm ~100 MB vào lịch sử cho mọi người.
Cache chỉ dùng được khi cùng model + fps + tiền xử lý (xem ghi chú của release) — script này KHÔNG kiểm
điều đó, nó chỉ đảm bảo file tải về đúng từng byte so với `.sha256` đi kèm.

Chỉ dùng thư viện chuẩn (không cần pip install gì thêm).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Kyosaki6/vlm-anomalies-detection"
DEFAULT_TAG = "cache-v1"
DEFAULT_ASSET = "vad-cache-siglip2-clip-fps1-v1.tar.gz"
# Chỉ giải nén dưới các thư mục này; mọi thứ khác trong tar bị từ chối (tar độc hại / đóng gói nhầm).
ALLOWED_TOP = ("cache/embeddings", "cache/prompts")
CHUNK = 1 << 20


class FetchCacheError(RuntimeError):
    """Tải/kiểm/giải nén thất bại. Không bao giờ để lại cache giải nén dở."""


def release_url(repo: str, tag: str, name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256_file(text: str) -> str:
    """Định dạng của `sha256sum`: '<64 hex>  <tên file>'. Lấy token đầu."""
    tokens = text.split()
    if not tokens or len(tokens[0]) != 64 or any(c not in "0123456789abcdef" for c in tokens[0].lower()):
        raise FetchCacheError(f"File .sha256 không đúng định dạng sha256sum: {text[:80]!r}")
    return tokens[0].lower()


def download(url: str, dest: Path) -> None:
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e6:6.1f}/{total / 1e6:.1f} MB", end="", file=sys.stderr)
            print(file=sys.stderr)
    except OSError as e:
        raise FetchCacheError(f"Không tải được {url}: {e}") from e


def safe_members(tar: tarfile.TarFile, dest_root: Path) -> list[tarfile.TarInfo]:
    """Chỉ nhận file thường dưới ALLOWED_TOP, không '..', không đường dẫn tuyệt đối, không symlink.

    Python 3.10 chưa có `extractall(filter='data')`, nên tự kiểm.
    """
    root = dest_root.resolve()
    out: list[tarfile.TarInfo] = []
    for m in tar.getmembers():
        name = m.name.replace("\\", "/")
        if m.isdir():
            continue
        if not m.isfile():
            raise FetchCacheError(f"Tar chứa mục không phải file thường (symlink/thiết bị?): {m.name!r}")
        if name.startswith("/") or ".." in Path(name).parts:
            raise FetchCacheError(f"Đường dẫn nguy hiểm trong tar: {m.name!r}")
        if not any(name == a or name.startswith(a + "/") for a in ALLOWED_TOP):
            raise FetchCacheError(f"Tar chứa file ngoài {ALLOWED_TOP}: {m.name!r}")
        if not (root / name).resolve().is_relative_to(root):
            raise FetchCacheError(f"Đường dẫn thoát khỏi thư mục đích: {m.name!r}")
        out.append(m)
    if not out:
        raise FetchCacheError("Tar rỗng (không có file nào dưới cache/embeddings hoặc cache/prompts)")
    return out


def extract(archive: Path, dest_root: Path, force: bool = False) -> int:
    """Giải nén vào `dest_root` (thư mục chứa `cache/`). Trả về số file ghi. Không đè file có sẵn nếu !force."""
    with tarfile.open(archive, "r:*") as tar:
        members = safe_members(tar, dest_root)
        clash = [m.name for m in members if (dest_root / m.name).exists()]
        if clash and not force:
            raise FetchCacheError(f"{len(clash)} file đã có (vd {clash[0]!r}). Chạy lại với --force để ghi đè.")
        for m in members:
            tar.extract(m, dest_root)
    return len(members)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--asset", default=DEFAULT_ASSET)
    p.add_argument("--dest", default=str(ROOT), help="thư mục chứa `cache/` (mặc định: gốc repo)")
    p.add_argument("--force", action="store_true", help="ghi đè file cache đã có")
    args = p.parse_args(argv)

    dest = Path(args.dest)
    url = release_url(args.repo, args.tag, args.asset)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            archive = tmp_dir / args.asset
            sha_file = tmp_dir / (args.asset + ".sha256")
            print(f"Tải {url}")
            download(url, archive)
            download(url + ".sha256", sha_file)
            want = parse_sha256_file(sha_file.read_text(encoding="utf-8-sig"))
            got = sha256_of(archive)
            if got != want:
                raise FetchCacheError(f"SHA-256 không khớp: file tải về {got}, release khai {want}. Không giải nén.")
            print("SHA-256 khớp.")
            n = extract(archive, dest, force=args.force)
    except FetchCacheError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(f"Đã giải nén {n} file vào {dest / 'cache'}")
    print("Kiểm đủ cache cho manifest: python scripts/build_results.py --manifest manifests/ucf_test.csv --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
