# -*- coding: utf-8 -*-
"""A1redo (Phước): dựng tập 16 cửa sổ benchmark THẬT cho tier-2 (redo plan §3.1, sửa F3).

16 cửa sổ = 8 anomalous (video UCF-Crime TEST có GT segment, 8 lớp khác nhau) + 8 normal
(từ manifests/ucf_calib_normal.csv — KHÔNG đụng tập test). Mỗi cửa sổ = k=4 khung cách đều
trong đoạn GT (hoặc trong một span 4-8 giây ngẫu nhiên với cửa sổ normal), seed cố định 0,
chọn xác định giữa các máy. Không chọn tay, không re-roll sau khi thấy kết quả model.

    python scripts/make_bench_windows.py --fetch       # tải annotation + 16 video cần thiết rồi trích khung
    python scripts/make_bench_windows.py                # chỉ trích khung (video đã có sẵn)

Kết quả:
    notes/vlm-bench-windows.json   (spec cửa sổ: video_id, label, gt_event, frame_indices, frame_paths)
    bench_windows/<window_id>/frame_<i>.jpg  (khung 320x240 gốc, JPEG q95 — CHỈ commit zip `notes/bench_windows.zip`, không commit thư mục)

Chỉ tải đúng 16 video cần (vài trăm MB), KHÔNG tải 9.2 GB cả bộ.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.ucf_crime import load_annotations  # noqa: E402
from vad.io.video import probe_video  # noqa: E402

REPO_ID = "AllenXeon/ucf_crime"
# Annotation GT nằm trong repo HF ở đường dẫn này (xem scripts/download_ucf_crime.py).
ANNOTATION_REPO_PATH = (
    "UCF_Crimes-Train-Test-Split/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
)
# Manifest `path` = đường dẫn local = prefix + đường dẫn trong repo HF.
PREFIX_BY_SPLIT = {
    "test": "datasets/ucf_crime/test/",
    "calib_normal": "datasets/ucf_crime/train_normal/",
}
JPEG_QUALITY = 95


def evenly_spaced(start: int, end: int, k: int) -> list[int]:
    """k chỉ số cách đều ĐÓNG trong [start, end] (hai đầu mút luôn nằm trong tập).

    Cùng quy ước "k khung cách đều" mà A3 (scripts/run_verify.py) sẽ dùng.
    """
    if k <= 0:
        raise ValueError("k phải > 0")
    if end < start:
        raise ValueError(f"Đoạn ngược: [{start}, {end}]")
    if k == 1:
        return [(start + end) // 2]
    return [start + round(i * (end - start) / (k - 1)) for i in range(k)]


def read_manifest(path: str | Path) -> list[dict]:
    """Đọc manifest CSV (utf-8-sig) -> list row có int n_frames, float fps."""
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "video_id": row["video_id"],
                "path": row["path"],
                "split": row["split"],
                "n_frames": int(row["n_frames"]),
                "fps": float(row["fps"]),
            })
    if not rows:
        raise ValueError(f"Manifest rỗng: {path}")
    return rows


def repo_path_of(local_path: str, split: str) -> str:
    """Đường dẫn manifest -> đường dẫn trong repo HF (prefix đã biết theo split)."""
    prefix = PREFIX_BY_SPLIT.get(split)
    if prefix is None or not local_path.startswith(prefix):
        raise ValueError(f"Path {local_path!r} không khớp prefix của split {split!r}")
    return local_path[len(prefix):]


def choose_windows(
    test_rows: list[dict],
    calib_rows: list[dict],
    annotations: dict,
    seed: int = 0,
    k: int = 4,
    n_anomaly: int = 8,
    n_normal: int = 8,
) -> list[dict]:
    """Chọn xác định (seed cố định) 16 cửa sổ. Hàm thuần: nhận rows + annotation đã nạp.

    Anomalous: 8 lớp khác nhau, mỗi lớp 1 video test có GT segment dài >= k khung,
    khung = evenly_spaced trong segment (segment chọn bằng rng).
    Normal: 8 video calib, span 4-8 giây ngẫu nhiên, khung = evenly_spaced trong span.
    """
    rng = random.Random(seed)

    # --- Anomalous: nhóm video theo lớp, chỉ giữ video có segment đủ dài ---
    by_class: dict[str, list[tuple[dict, list[tuple[int, int]]]]] = {}
    for row in test_rows:
        ann = annotations.get(row["video_id"])
        if ann is None:
            continue
        # Bẫy UCF: annotation inclusive [start, end] có thể trỏ tới đúng khung cuối+1 của container
        # (Arson011: end=1267 nhưng container chỉ có 1266 khung) -> clamp về n_frames-1.
        eligible: list[tuple[int, int]] = []
        for s, e in ann.segments:
            e_clamped = min(e, row["n_frames"] - 1)
            if e_clamped - s >= k - 1:
                eligible.append((s, e_clamped))
        if eligible:
            by_class.setdefault(ann.label, []).append((row, eligible))
    if not by_class:
        raise ValueError("Không có video test nào có segment đủ dài (annotation đúng? video đúng?)")

    classes = sorted(by_class)
    rng.shuffle(classes)
    if len(classes) < n_anomaly:
        raise ValueError(f"Chỉ có {len(classes)} lớp có GT, cần {n_anomaly}")

    windows: list[dict] = []
    for cls in classes[:n_anomaly]:
        candidates = by_class[cls]
        rng.shuffle(candidates)
        row, eligible = candidates[0]
        seg = eligible[rng.randrange(len(eligible))]
        windows.append({
            "video_id": row["video_id"],
            "label": "anomaly",
            "gt_event": cls,
            "fps": row["fps"],
            "n_frames": row["n_frames"],
            "video_path": row["path"],
            "segment": [seg[0], seg[1]],
            "frame_indices": evenly_spaced(seg[0], seg[1], k),
            "source": "ucf_crime_test_gt",
        })

    # --- Normal: từ calib (train-normal), KHÔNG đụng tập test ---
    normals = list(calib_rows)
    rng.shuffle(normals)
    if len(normals) < n_normal:
        raise ValueError(f"Manifest calib chỉ có {len(normals)} video, cần {n_normal}")
    for row in normals[:n_normal]:
        span_f = max(k - 1, round(rng.uniform(4.0, 8.0) * row["fps"]))
        span_f = min(span_f, row["n_frames"] - 1)
        start = rng.randrange(0, row["n_frames"] - span_f)
        windows.append({
            "video_id": row["video_id"],
            "label": "normal",
            "gt_event": "Normal",
            "fps": row["fps"],
            "n_frames": row["n_frames"],
            "video_path": row["path"],
            "segment": [start, start + span_f],
            "frame_indices": evenly_spaced(start, start + span_f, k),
            "source": "ucf_calib_normal",
        })

    for i, w in enumerate(windows, start=1):
        w["window_id"] = f"w{i:02d}"
    return windows


def grab_frames(video_path: str | Path, indices: list[int]) -> list:
    """Lấy các khung theo chỉ số bằng grab TUẦN TÙ (đúng với codec x264; xem vad/io/video.py).

    Trả về list[np.ndarray RGB uint8] cùng thứ tự `indices`.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV không mở được: {video_path}")
    try:
        out = []
        frame_idx = 0
        for target in sorted(set(indices)):
            while frame_idx <= target:
                if not cap.grab():
                    raise RuntimeError(
                        f"Hết khung sớm hơn dự kiến ở chỉ số {target} (video {video_path})"
                    )
                frame_idx += 1
            ok, bgr = cap.retrieve()
            if not ok:
                raise RuntimeError(f"retrieve() thất bại ở khung {target} ({video_path})")
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # Trả về đúng thứ tự indices (indices đã sorted ở trên mặt số, nhưng giữ guard):
        order = {idx: pos for pos, idx in enumerate(sorted(set(indices)))}
        return [out[order[i]] for i in indices]
    finally:
        cap.release()


def annotation_path(root: Path) -> Path:
    return root / "datasets" / "ucf_crime" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"


def fetch_annotation(root: Path) -> None:
    """Tải file annotation GT từ HF (nhỏ, vài KB)."""
    from huggingface_hub import hf_hub_download

    out_dir = root / "datasets" / "ucf_crime"
    ann_local = annotation_path(root)
    if ann_local.exists():
        return
    print("Tải annotation GT...", flush=True)
    src = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                          filename=ANNOTATION_REPO_PATH, local_dir=str(out_dir / "_meta"))
    shutil.copyfile(src, ann_local)


def fetch_videos(windows: list[dict], root: Path, threads: int = 4) -> None:
    """Tải đúng các video mà windows cần, từ HF. Không tải cả bộ."""
    from huggingface_hub import hf_hub_download

    out_dir = root / "datasets" / "ucf_crime"
    missing = [w for w in windows if not (root / w["video_path"]).exists()]
    if not missing:
        print("Đủ 16 video, không tải thêm.", flush=True)
        return
    print(f"Tải {len(missing)} video cần thiết...", flush=True)
    failed: list[str] = []

    def one(w: dict) -> str:
        split = "test" if w["label"] == "anomaly" else "calib_normal"
        # manifest path = <datasets/ucf_crime>/<split>/<repo_path> -> local_dir phải là <...>/<split>
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                        filename=repo_path_of(w["video_path"], split),
                        local_dir=str(out_dir / split))
        return w["video_id"]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(one, w) for w in missing]
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                fut.result()
            except Exception as e:  # không nuốt lỗi: ghi danh sách, fail ở cuối
                failed.append(f"fut {i}: {type(e).__name__}: {e}")
            if i % 5 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}]", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} video tải lỗi: {failed[:3]}")


def extract_all(windows: list[dict], root: Path, frames_dir: Path) -> tuple[list[dict], list[dict]]:
    """Trích khung cho từng cửa sổ. Cửa sổ lỗi được ghi vào `skipped`, không nuốt lẫn không crash cả lượt."""
    from PIL import Image

    good: list[dict] = []
    skipped: list[dict] = []
    for w in windows:
        vid_path = root / w["video_path"]
        try:
            n_actual, fps_actual = probe_video(vid_path)
            if max(w["frame_indices"]) >= n_actual:
                raise ValueError(
                    f"frame_indices {w['frame_indices']} vượt n_frames thật {n_actual}"
                )
            frames = grab_frames(vid_path, w["frame_indices"])
            wdir = frames_dir / w["window_id"]
            wdir.mkdir(parents=True, exist_ok=True)
            rel_paths = []
            for i, img in enumerate(frames):
                fp = wdir / f"frame_{i}.jpg"
                Image.fromarray(img).save(fp, quality=JPEG_QUALITY)
                rel_paths.append(str(PurePosixPath("bench_windows", w["window_id"],
                                                   f"frame_{i}.jpg")))
            out = dict(w)
            out["frame_paths"] = rel_paths
            out["fps_actual"] = round(fps_actual, 3)
            good.append(out)
        except Exception as e:
            print(f"  LỖI {w['window_id']} ({w['video_id']}): {type(e).__name__}: {e}", flush=True)
            skipped.append({"window_id": w["window_id"], "video_id": w["video_id"],
                            "error": f"{type(e).__name__}: {e}"})
    return good, skipped


def main(argv: list[str] | None = None) -> int:
    # Console Windows mặc định cp1252: in tiếng Việt sẽ raise -> bật utf-8 (xem bench_vlm.py).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="thư mục gốc repo")
    ap.add_argument("--out", default="notes/vlm-bench-windows.json")
    ap.add_argument("--frames-dir", default="bench_windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="số khung mỗi cửa sổ (A1 spec: 4)")
    ap.add_argument("--fetch", action="store_true",
                    help="tải annotation + 16 video cần từ HF trước khi trích khung")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    frames_dir = root / "bench_windows"

    if args.fetch:
        fetch_annotation(root)
    if not annotation_path(root).exists():
        print("Thiếu annotation GT — chạy lại với --fetch để tải.", file=sys.stderr)
        return 2

    annotations = load_annotations(annotation_path(root))
    test_rows = read_manifest(root / "manifests/ucf_test.csv")
    calib_rows = read_manifest(root / "manifests/ucf_calib_normal.csv")

    windows = choose_windows(test_rows, calib_rows, annotations, seed=args.seed, k=args.k)
    n_anom = sum(1 for w in windows if w["label"] == "anomaly")
    print(f"Chọn {len(windows)} cửa sổ ({n_anom} anomaly / {len(windows) - n_anom} normal), "
          f"seed={args.seed}, k={args.k}")
    for w in windows:
        print(f"  {w['window_id']} {w['label']:7s} {w['gt_event']:13s} {w['video_id']} "
              f"seg={w['segment']} idx={w['frame_indices']}")

    if args.fetch:
        fetch_videos(windows, root)

    good, skipped = extract_all(windows, root, frames_dir)
    if len(good) != len(windows):
        print(f"CẢNH BÁO: chỉ trích được {len(good)}/{len(windows)} cửa sổ "
              f"(xem skipped trong JSON)", file=sys.stderr)

    commit = None
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass

    payload = {
        "header": {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "builder_commit": commit,
            "seed": args.seed,
            "k": args.k,
            "n_anomaly": n_anom,
            "n_normal": len(windows) - n_anom,
            "source_test": "manifests/ucf_test.csv + Temporal_Anomaly_Annotation (GT segment)",
            "source_normal": "manifests/ucf_calib_normal.csv (train-normal, không đụng test)",
            "jpeg_quality": JPEG_QUALITY,
            "skipped": skipped,
            "note": "Chọn xác định seed 0; không chọn tay; 4 khung cách đều (quy ước A3).",
        },
        "windows": good,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Đã ghi {out_path} ({len(good)} cửa sổ, {len(skipped)} bỏ qua)")
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
