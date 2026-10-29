# -*- coding: utf-8 -*-
"""Tải UCF-Crime test (290 video) + N video train-normal (hiệu chỉnh ngưỡng) + annotation từ mirror Hugging Face.

    python scripts/download_ucf_crime.py --dry-run          # chỉ liệt kê việc sẽ tải + dung lượng (không tải video)
    python scripts/download_ucf_crime.py                     # tải (~8.7 GB), chạy lại = bỏ qua file đã có

Nguồn: HF dataset `AllenXeon/ucf_crime` (CC0, không cần đăng nhập, từng file .mp4 riêng).
Kết quả (thư mục `datasets/` đã gitignore, KHÔNG commit video):
    datasets/ucf_crime/Anomaly_Test.txt, Anomaly_Train.txt, Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    datasets/ucf_crime/test/...          (290 video, giữ nguyên cấu trúc thư mục của repo HF)
    datasets/ucf_crime/train_normal/...  (N video train-normal, chọn bằng băm SHA-256 + seed, ổn định giữa các máy)

BẪY: repo HF có TRÙNG tên file ở nhiều thư mục (`test/Abuse/Abuse001_x264.mp4` là train video, còn
`Normal_Videos_for_Event_Recognition/` trùng tên với `Testing_Normal_Videos_Anomaly/`). Vì vậy KHÔNG tra theo
basename: mỗi entry của Anomaly_Test.txt/Anomaly_Train.txt được ánh xạ sang ĐÚNG MỘT đường dẫn trong repo, 0 hoặc >1 khớp = lỗi.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_manifest import pick_limit  # noqa: E402

REPO_ID = "AllenXeon/ucf_crime"
META_DIR = "UCF_Crimes-Train-Test-Split"
META_FILES = {
    "Anomaly_Test.txt": f"{META_DIR}/Anomaly_Detection_splits/Anomaly_Test.txt",
    "Anomaly_Train.txt": f"{META_DIR}/Anomaly_Detection_splits/Anomaly_Train.txt",
    "Temporal_Anomaly_Annotation_for_Testing_Videos.txt": f"{META_DIR}/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
}
# Nơi video ANOMALY của cả train lẫn test nằm trong repo: Anomaly-Videos-Part-{1..4}/<Class>/<name>.
ANOMALY_PREFIX = "Anomaly-Videos-Part-"
TEST_NORMAL_DIR = "Testing_Normal_Videos_Anomaly"
TRAIN_NORMAL_DIRS = ("Training-Normal-Videos-Part-1", "Training-Normal-Videos-Part-2")


def parse_split_entries(text: str) -> list[str]:
    out = [ln.strip().replace("\\", "/") for ln in text.lstrip("\ufeff").splitlines() if ln.strip()]
    if len(out) != len(set(out)):
        raise ValueError("Danh sách split có entry trùng")
    return out


def resolve_entry(entry: str, repo_files: set[str]) -> str:
    """Entry của file split -> đường dẫn DUY NHẤT trong repo HF. 0 hoặc >1 khớp -> ValueError."""
    parts = entry.split("/")
    if len(parts) != 2:
        raise ValueError(f"Entry không phải '<thư mục>/<tên>.mp4': {entry!r}")
    folder, name = parts
    if folder in ("Testing_Normal_Videos_Anomaly",):
        candidates = [f"{TEST_NORMAL_DIR}/{name}"]
    elif folder in ("Training_Normal_Videos_Anomaly", "Training-Normal-Videos"):
        candidates = [f"{d}/{name}" for d in TRAIN_NORMAL_DIRS]
    else:  # tên lớp anomaly: Abuse, Arson, ...
        candidates = [f"{ANOMALY_PREFIX}{i}/{folder}/{name}" for i in range(1, 5)]
    hit = [c for c in candidates if c in repo_files]
    if len(hit) != 1:
        raise ValueError(f"Entry {entry!r}: {len(hit)} khớp trong repo (cần đúng 1): {hit or candidates}")
    return hit[0]


def plan_downloads(test_entries: list[str], train_entries: list[str], repo_files: set[str],
                   calib_count: int, seed: int = 0, sizes: dict[str, int] | None = None,
                   max_calib_mb: float | None = None) -> tuple[list[str], list[str]]:
    """-> (đường dẫn repo của video test, đường dẫn repo của video calib).

    `max_calib_mb` (cần `sizes`): loại video train-normal quá lớn TRƯỚC khi băm chọn. Video train có cái dài
    hàng giờ (tới 8.7 GB); hiệu chỉnh ngưỡng không cần chúng mà giải mã + embed rất tốn. Kích thước lấy từ API HF
    nên mọi máy chọn ra cùng một tập.
    """
    if len(test_entries) != 290:
        raise ValueError(f"Anomaly_Test.txt phải có 290 entry, thực có {len(test_entries)}")
    test_paths = [resolve_entry(e, repo_files) for e in test_entries]
    train_normal = [e for e in train_entries if e.split("/")[-1].startswith("Normal_Videos")]
    if max_calib_mb is not None:
        if sizes is None:
            raise ValueError("max_calib_mb cần sizes")
        train_normal = [e for e in train_normal if sizes[resolve_entry(e, repo_files)] <= max_calib_mb * 1e6]
    chosen = set(pick_limit([Path(e).stem for e in train_normal], calib_count, seed))
    calib_paths = sorted(resolve_entry(e, repo_files) for e in train_normal if Path(e).stem in chosen)
    if len(calib_paths) != calib_count:
        raise ValueError(f"Chọn được {len(calib_paths)} video calib, cần {calib_count}")
    overlap = {Path(p).stem for p in test_paths} & {Path(p).stem for p in calib_paths}
    if overlap:
        raise ValueError(f"Video calib trùng video test: {sorted(overlap)[:5]}")
    return test_paths, calib_paths


def _download(hf_hub_download, repo_path: str, local_dir: Path) -> str:
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_path, local_dir=str(local_dir))


def run_downloads(paths: Iterable[str], local_dir: Path, threads: int, hf_hub_download) -> list[str]:
    paths = list(paths)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(_download, hf_hub_download, p, local_dir): p for p in paths}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                fut.result()
            except Exception as e:  # file lỗi không làm dừng cả lượt; chạy lại là tải tiếp phần thiếu
                failed.append(futs[fut])
                print(f"  LỖI {futs[fut]}: {type(e).__name__}: {e}")
            if i % 25 == 0 or i == len(paths):
                print(f"  [{i}/{len(paths)}] -> {local_dir}")
    return failed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="datasets/ucf_crime")
    p.add_argument("--calib-count", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-calib-mb", type=float, default=100.0,
                   help="bỏ video train-normal lớn hơn mức này (MB) trước khi chọn; 0 = không giới hạn")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    repo_files = set(sizes)

    for local_name, repo_path in META_FILES.items():
        src = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_path, local_dir=str(out / "_meta"))
        (out / local_name).write_bytes(Path(src).read_bytes())
    test_entries = parse_split_entries((out / "Anomaly_Test.txt").read_text(encoding="utf-8-sig"))
    train_entries = parse_split_entries((out / "Anomaly_Train.txt").read_text(encoding="utf-8-sig"))
    test_paths, calib_paths = plan_downloads(test_entries, train_entries, repo_files, args.calib_count, args.seed,
                                           sizes, args.max_calib_mb or None)

    gb = lambda ps: sum(sizes[x] for x in ps) / 1e9  # noqa: E731
    print(f"test: {len(test_paths)} video, {gb(test_paths):.2f} GB | calib: {len(calib_paths)} video, {gb(calib_paths):.2f} GB")
    if args.dry_run:
        return 0
    failed = run_downloads(test_paths, out / "test", args.threads, hf_hub_download)
    failed += run_downloads(calib_paths, out / "train_normal", args.threads, hf_hub_download)
    if failed:
        print(f"{len(failed)} file lỗi; chạy lại cùng lệnh để tải phần còn thiếu: {failed[:5]}")
        return 1
    print("xong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
