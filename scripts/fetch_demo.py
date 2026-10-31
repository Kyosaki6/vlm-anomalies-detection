# -*- coding: utf-8 -*-
"""Tải bộ clip demo (`datasets/own/*.mp4`) từ Google Drive của nhóm, kiểm SHA-256, đặt đúng chỗ.

    pip install gdown                    # chỉ cần cho script này (không nằm trong requirements)
    python scripts/fetch_demo.py         # tải 7 clip vào datasets/own/
    python scripts/fetch_demo.py --force # ghi đè file đã có
    python scripts/fetch_demo.py --check # chỉ kiểm file đang có, không tải

Vì sao không commit video: `/datasets/` bị .gitignore, và clip stock có watermark/bản quyền.
Script từ chối file lạ, thiếu file, hoặc SHA-256 lệch với `EXPECTED_SHA256` — đổi bộ clip thì phải cập nhật bảng đó
(và `manifests/own_demo.csv` bằng `scripts/build_manifest.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER_ID = "1I01ZQn6-7GW7Rm96Lozf0IQEkPtCP7lA"
DEFAULT_DEST = ROOT / "datasets" / "own"
CHUNK = 1 << 20

EXPECTED_SHA256: dict[str, str] = {
    "own_fight_01.mp4": "0f9f74e8041a1a18bd86f9bf4945944b05e89cc8263f51e885b76e5d95c3ad69",
    "own_fight_02.mp4": "947b17e50743c10515f1f1e7f2e82374c5e5b49dcee55c176333d0ea96355e8e",
    "own_fight_03.mp4": "2bd1967c42aaa060fa2041c2b80c79d5a37a7ae0b4c522169d8fbb3b8bda2ff6",
    "own_normal_01.mp4": "6f3b53a4f88e85a28f534b429769e54ea206f1b40a7d51f11035590f241171e1",
    "own_normal_02.mp4": "18f5514b0c801b0934c8a68fe951df20452f76b0f2bf1235e7932d227bf22cae",
    "own_normal_03.mp4": "4a239dd95d816a93361a48ce63e254b8eb719ccb632cdd45beae9befb9d09905",
    "own_steal_01.mp4": "202df56a5038263316a3c71b9454f7ee4c9b0dc9dee2d0fa7474077d58e97e8f",
}

class FetchDemoError(RuntimeError):
    """Tải/kiểm thất bại. Không để lại file dở trong thư mục đích."""

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()

def verify_dir(directory: Path, expected: dict[str, str] = EXPECTED_SHA256) -> None:
    """Đủ file, đúng SHA-256. Thiếu/lệch -> FetchDemoError liệt kê hết lỗi một lần."""
    problems: list[str] = []
    for name, want in sorted(expected.items()):
        p = directory / name
        if not p.is_file():
            problems.append(f"thiếu {name}")
        elif (got := sha256_of(p)) != want:
            problems.append(f"{name}: SHA-256 lệch (có {got[:12]}…, cần {want[:12]}…)")
    if problems:
        raise FetchDemoError("; ".join(problems))

def gdown_folder(folder_id: str, out_dir: Path) -> None:
    """Tải cả thư mục Drive công khai bằng gdown (import trễ: test không cần cài)."""
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError as e:
        raise FetchDemoError("Thiếu gdown. Chạy: pip install gdown") from e
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        files = gdown.download_folder(url=url, output=str(out_dir), quiet=False, use_cookies=False)
    except Exception as e:  # gdown ném nhiều loại lỗi mạng/quyền khác nhau
        raise FetchDemoError(f"Không tải được thư mục Drive {folder_id}: {e}") from e
    if not files:
        raise FetchDemoError(f"Thư mục Drive {folder_id} không tải được file nào (chưa chia sẻ công khai?)")

def fetch(dest: Path = DEFAULT_DEST, folder_id: str = DEFAULT_FOLDER_ID, force: bool = False,
          downloader: Callable[[str, Path], None] = gdown_folder,
          expected: dict[str, str] = EXPECTED_SHA256) -> list[Path]:
    """Tải vào thư mục tạm, kiểm SHA-256, rồi mới chuyển vào `dest`. File có sẵn bị giữ nguyên nếu thiếu --force."""
    with tempfile.TemporaryDirectory(prefix="fetch_demo_") as tmp:
        tmp_dir = Path(tmp)
        downloader(folder_id, tmp_dir)
        # gdown có thể tạo thư mục con; gom mọi .mp4 theo tên.
        found = {p.name: p for p in tmp_dir.rglob("*.mp4")}
        unexpected = sorted(set(found) - set(expected))
        if unexpected:
            raise FetchDemoError(f"File lạ trên Drive, không có trong EXPECTED_SHA256: {unexpected}")
        flat = tmp_dir / "_verified"
        flat.mkdir()
        for name, p in found.items():
            shutil.copy2(p, flat / name)
        verify_dir(flat, expected)
        dest.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name in sorted(expected):
            target = dest / name
            if target.exists() and not force:
                if sha256_of(target) != expected[name]:
                    raise FetchDemoError(f"{target} đã có nhưng khác bản chuẩn; chạy lại với --force để ghi đè")
                continue
            shutil.copy2(flat / name, target)
            written.append(target)
        return written

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    ap.add_argument("--force", action="store_true", help="ghi đè file đã có")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm SHA-256 file đang có, không tải")
    args = ap.parse_args(argv)
    try:
        if args.check:
            verify_dir(args.dest)
            print(f"OK: {len(EXPECTED_SHA256)} clip trong {args.dest} đúng SHA-256")
            return 0
        written = fetch(args.dest, args.folder_id, args.force)
    except FetchDemoError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(f"Ghi {len(written)} file mới vào {args.dest} ({len(EXPECTED_SHA256) - len(written)} đã có, giữ nguyên)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
