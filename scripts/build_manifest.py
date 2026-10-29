# -*- coding: utf-8 -*-
"""Quét thư mục video -> manifest CSV (B1). Cột: video_id,path,dataset,split,n_frames,fps.

    # 290 video test (danh sách chính thức + annotation để kiểm chéo):
    python scripts/build_manifest.py --root datasets/ucf_crime/test --split test \\
        --list datasets/ucf_crime/Anomaly_Test.txt \\
        --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \\
        --expect 290 --out manifests/ucf_test.csv
    # ~60 video train-normal để hiệu chỉnh ngưỡng (chọn ngẫu nhiên CÓ seed, ổn định giữa các máy):
    python scripts/build_manifest.py --root datasets/ucf_crime/train_normal --split calib_normal \\
        --limit 60 --expect 60 --out manifests/ucf_calib_normal.csv

Quy tắc:
- `path` trong manifest PHẢI tương đối theo gốc repo (dấu `/`), để MỘT file commit chạy được trên cả 4 máy.
  Thư mục video nằm ngoài repo -> báo lỗi (trừ khi --allow-absolute, và khi đó đặt tên `*.local.csv`).
- Video không mở được, `video_id` trùng, thiếu video trong --list, thiếu tên trong --annotation -> RAISE,
  không bỏ qua im lặng (manifest sai làm hỏng cả chuỗi shard phía sau).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.ucf_crime import load_annotations  # noqa: E402
from vad.io.video import probe_video  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
FIELDS = ["video_id", "path", "dataset", "split", "n_frames", "fps"]
SPLITS = ("test", "calib_normal", "demo")
DATASETS = ("ucf_crime", "shanghaitech", "own")


def find_videos(root: Path) -> dict[str, Path]:
    """{video_id: path} cho mọi video dưới root (đệ quy). Trùng stem ở hai thư mục -> ValueError."""
    if not root.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục video: {root}")
    found: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            if p.stem in found:
                raise ValueError(f"video_id trùng '{p.stem}': {found[p.stem]} và {p}")
            found[p.stem] = p
    return found


def read_id_list(path: Path) -> list[str]:
    """Đọc danh sách tên (Anomaly_Test.txt: mỗi dòng 'Abuse/Abuse028_x264.mp4' hoặc tên trần) -> video_id."""
    ids: list[str] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            tok = line.strip().split()
            if not tok:
                continue
            ids.append(Path(tok[0].replace("\\", "/")).stem)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: danh sách có tên trùng")
    return ids


def pick_limit(ids: Sequence[str], limit: int, seed: int) -> list[str]:
    """Chọn `limit` id bằng băm SHA-256 + seed: cùng kết quả trên mọi máy/Python, không phụ thuộc thứ tự quét."""
    if limit > len(ids):
        raise ValueError(f"--limit {limit} > số video có sẵn {len(ids)}")
    ranked = sorted(ids, key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
    return sorted(ranked[:limit])


def rel_path(p: Path, allow_absolute: bool) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        if allow_absolute:
            return p.resolve().as_posix()
        raise ValueError(
            f"{p} nằm ngoài repo ({REPO_ROOT}): manifest commit chung phải dùng đường dẫn tương đối. "
            "Đặt video dưới datasets/ hoặc dùng --allow-absolute và đặt tên *.local.csv."
        ) from None


def build_rows(videos: dict[str, Path], dataset: str, split: str, allow_absolute: bool = False) -> list[dict]:
    """probe từng video -> dòng manifest. Gom MỌI video hỏng rồi mới raise một lần."""
    rows, bad = [], []
    for vid in sorted(videos):
        p = videos[vid]
        try:
            n_frames, fps = probe_video(p)
        except (RuntimeError, FileNotFoundError) as e:
            bad.append(f"{vid}: {e}")
            continue
        if n_frames <= 0:
            bad.append(f"{vid}: n_frames={n_frames}")
            continue
        rows.append({"video_id": vid, "path": rel_path(p, allow_absolute), "dataset": dataset, "split": split,
                     "n_frames": n_frames, "fps": f"{fps:g}"})
    if bad:
        raise RuntimeError(f"{len(bad)} video hỏng/không đọc được:\n  " + "\n  ".join(bad))
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    ids = [r["video_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("video_id trùng trong manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="thư mục chứa video (quét đệ quy)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--dataset", default="ucf_crime", choices=DATASETS)
    p.add_argument("--list", default=None, help="file danh sách video chính thức; chỉ lấy các video này, thiếu = lỗi")
    p.add_argument("--annotation", default=None, help="Temporal annotation; mọi tên trong đó phải có trong manifest")
    p.add_argument("--limit", type=int, default=None, help="chỉ lấy N video (băm có seed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expect", type=int, default=None, help="số dòng mong đợi; lệch = lỗi")
    p.add_argument("--allow-absolute", action="store_true")
    args = p.parse_args(argv)

    videos = find_videos(Path(args.root))
    if args.list:
        wanted = read_id_list(Path(args.list))
        missing = [v for v in wanted if v not in videos]
        if missing:
            raise SystemExit(f"Thiếu {len(missing)} video trong {args.root}, ví dụ: {missing[:5]}")
        videos = {v: videos[v] for v in wanted}
    if args.limit is not None:
        videos = {v: videos[v] for v in pick_limit(sorted(videos), args.limit, args.seed)}
    if args.annotation:
        ann_missing = [v for v in load_annotations(args.annotation) if v not in videos]
        if ann_missing:
            raise SystemExit(f"{len(ann_missing)} video có trong annotation nhưng không có trong manifest: {ann_missing[:5]}")

    rows = build_rows(videos, args.dataset, args.split, args.allow_absolute)
    if args.expect is not None and len(rows) != args.expect:
        raise SystemExit(f"Mong đợi {args.expect} dòng, thực có {len(rows)}. Không ghi file.")
    write_manifest(rows, Path(args.out))
    print(f"ghi {len(rows)} dòng -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
