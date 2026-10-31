# -*- coding: utf-8 -*-
"""A6a: rút mẫu 20 cửa sổ để NGƯỜI tự kiểm tay (task A6, Phước).

    python scripts/draw_a6_sample.py --seed 20261031 --out notes/a6-handcheck-sample.csv

Nguyên tắc chống thiên vị (ghi trước, thực thi bằng code):
  - Mẫu rút TRƯỚC khi đọc caption: file CSV KHÔNG có cột caption/mô tả.
  - 20 mẫu = 10 cửa sổ VLM xác nhận + 10 VLM từ chối, phân tầng theo
    video có sự cố (annotation) / video bình thường, seed cố định -> ai chạy lại
    cũng ra đúng danh sách này.
  - Cột `gt_overlap` = cửa sổ có chồng lấn đoạn GT hay không (người chấm sẽ đối
    chiếu) — được tính từ annotation, KHÔNG dùng để chọn mẫu (chỉ ghi lại).
  - BẮT BUỘC đủ 307 file kết quả, trừ khi `--allow-partial` (chỉ để chạy thử;
    mẫu thiếu không được dùng cho báo cáo).

Đầu vào:
  results/windows/<video_id>.json   — cửa sổ tier-1 (metadata.fps / n_frames)
  results/verify/<video_id>/window_*.json — kết quả tier-2 (verify_results[0].is_anomaly)
  datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.ucf_crime import load_annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_DIR = ROOT / "results" / "windows"
DEFAULT_VERIFY_DIR = ROOT / "results" / "verify"
DEFAULT_ANNOTATION = ROOT / "datasets" / "ucf_crime" / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
DEFAULT_OUT = ROOT / "notes" / "a6-handcheck-sample.csv"

PER_VERDICT = 10  # 10 xác nhận + 10 từ chối = 20 (task sheet A6)

FIELDS = [
    "video_id", "window_index", "start_s", "end_s", "duration_s", "score",
    "verdict", "video_class", "gt_overlap", "seed", "source_file",
]


def load_windows(windows_dir: Path) -> dict[tuple[str, int], dict]:
    """{(video_id, window_index): {start_s, end_s, score, fps, n_frames}} từ file tier-1."""
    index_path = windows_dir / "_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Thiếu {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    windows: dict[tuple[str, int], dict] = {}
    for row in index.get("windows", []):
        video_id = str(row["video_id"])
        w_index = int(row["window_index"])
        w_file = windows_dir / f"{video_id}.json"
        if not w_file.is_file():
            raise FileNotFoundError(f"_index trỏ tới file windows không tồn tại: {w_file}")
        data = json.loads(w_file.read_text(encoding="utf-8-sig"))
        meta = data.get("metadata") or {}
        if "fps" not in meta or "n_frames" not in meta:
            raise ValueError(f"{w_file} thiếu metadata.fps/n_frames (cần để đổi giây -> frame)")
        windows[(video_id, w_index)] = {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "score": float(row.get("score", 0.0)),
            "fps": float(meta["fps"]),
            "n_frames": int(meta["n_frames"]),
        }
    if not windows:
        raise ValueError(f"Không có window nào trong {windows_dir}")
    return windows


def load_verdicts(verify_dir: Path) -> dict[tuple[str, int], dict]:
    """{(video_id, window_index): verify_results[0]} từ tier-2. File hỏng = raise (không nuốt)."""
    if not verify_dir.is_dir():
        raise FileNotFoundError(f"Thiếu thư mục kết quả tier-2: {verify_dir}")
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(verify_dir.glob("*/window_*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        meta = data.get("metadata") or {}
        video_id = str(data.get("video_id") or meta.get("video_id") or path.parent.name)
        if "window_index" not in meta:
            raise ValueError(f"{path} thiếu metadata.window_index")
        results = data.get("verify_results") or []
        if len(results) != 1:
            raise ValueError(f"{path}: cần đúng 1 verify_result, nhận {len(results)}")
        vr = results[0]
        if not isinstance(vr.get("is_anomaly"), bool):
            raise ValueError(f"{path}: is_anomaly phải là bool, nhận {vr.get('is_anomaly')!r}")
        out[(video_id, int(meta["window_index"]))] = {
            "is_anomaly": bool(vr["is_anomaly"]),
            # as_posix(): file CSV được commit cho cả nhóm + Colab -> luôn dùng '/' (không phụ thuộc Windows)
            "source_file": path.relative_to(verify_dir.parent).as_posix(),
        }
    if not out:
        raise ValueError(f"Không có file kết quả tier-2 nào trong {verify_dir}")
    return out


def gt_overlap(w: dict, annotations: dict) -> tuple[bool, bool]:
    """(video_có_sự_cố, cửa_sổ_chồng_GT). Chồng theo frame, [start, end] đóng như load_gt."""
    ann = annotations.get(w["video_id"]) if "video_id" in w else None
    start_f = int(w["start_s"] * w["fps"])
    end_f = int(round(w["end_s"] * w["fps"]))  # cận trên đóng (quy ước ucf_crime.py)
    has_event = ann is not None and bool(ann.segments)
    if not has_event:
        return False, False
    overlap = any(start_f <= seg_end and seg_start <= end_f for seg_start, seg_end in ann.segments)
    return True, bool(overlap)


def allocate(stratum_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Chia n mẫu theo tỉ lệ các tầng (largest-remainder), tầng không rỗng được >= 1 nếu n đủ."""
    total = sum(stratum_sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in stratum_sizes}
    if n >= total:
        return dict(stratum_sizes)
    raw = {k: n * v / total for k, v in stratum_sizes.items()}
    take = {k: int(x) for k, x in raw.items()}
    # mỗi tầng không rỗng ít nhất 1 (nếu tổng <= n), phần dư phân bổ theo phần dư lớn nhất
    for k, v in stratum_sizes.items():
        if v > 0 and take[k] == 0 and sum(min(take[k2], v2) for k2, v2 in stratum_sizes.items() if v2 > 0) < n:
            take[k] = 1
    remainder = n - sum(take.values())
    order = sorted(stratum_sizes, key=lambda k: (raw[k] - int(raw[k])), reverse=True)
    i = 0
    while remainder > 0 and order:
        k = order[i % len(order)]
        if take[k] < stratum_sizes[k]:
            take[k] += 1
            remainder -= 1
        i += 1
        if i > 10 * len(order) + n:
            break
    return take


def draw_sample(
    windows: dict[tuple[str, int], dict],
    verdicts: dict[tuple[str, int], dict],
    annotations: dict,
    seed: int,
    per_verdict: int = PER_VERDICT,
) -> list[dict]:
    """Rút mẫu phân tầng: mỗi verdict (xác nhận/từ chối) -> per_verdict mẫu,
    chia đều theo tầng (video có sự cố / video bình thường) có trong verdict đó."""
    rows: list[dict] = []
    for key, w in windows.items():
        video_id, w_index = key
        if key not in verdicts:
            continue
        has_event, overlap = gt_overlap({**w, "video_id": video_id}, annotations)
        rows.append({
            "video_id": video_id,
            "window_index": w_index,
            "start_s": w["start_s"],
            "end_s": w["end_s"],
            "duration_s": round(w["end_s"] - w["start_s"], 3),
            "score": w["score"],
            "verdict": "confirmed" if verdicts[key]["is_anomaly"] else "rejected",
            "video_class": "anomalous" if has_event else "normal",
            "gt_overlap": "yes" if overlap else "no",
            "seed": seed,
            "source_file": verdicts[key]["source_file"],
        })

    rng = random.Random(seed)
    picked: list[dict] = []
    for verdict in ("confirmed", "rejected"):
        group = [r for r in rows if r["verdict"] == verdict]
        strata: dict[str, list[dict]] = defaultdict(list)
        for r in group:
            strata[r["video_class"]].append(r)
        if not group:
            raise ValueError(f"Kết quả tier-2 không có cửa sổ nào verdict={verdict} — mẫu sẽ mất cân bằng")
        take = allocate({k: len(v) for k, v in strata.items()}, per_verdict)
        for cls in sorted(strata):
            pool = sorted(strata[cls], key=lambda r: (r["video_id"], r["window_index"]))
            rng.shuffle(pool)
            picked.extend(pool[: take[cls]])
    return sorted(picked, key=lambda r: (r["verdict"], r["video_id"], r["window_index"]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows-dir", default=str(DEFAULT_WINDOWS_DIR))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY_DIR))
    p.add_argument("--annotation", default=str(DEFAULT_ANNOTATION))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--per-verdict", type=int, default=PER_VERDICT)
    p.add_argument("--allow-partial", action="store_true",
                   help="cho phép thiếu file tier-2 (CHỈ để chạy thử, không dùng cho báo cáo)")
    args = p.parse_args(argv)

    windows = load_windows(Path(args.windows_dir))
    verdicts = load_verdicts(Path(args.verify_dir))
    if len(verdicts) < len(windows) and not args.allow_partial:
        missing = sorted(set(windows) - set(verdicts))
        print(f"LỖI: thiếu {len(missing)}/{len(windows)} file tier-2 "
              f"(ví dụ {missing[:3]}) — mẫu rút khi thiếu sẽ méo. "
              f"Dùng --allow-partial chỉ để chạy thử.", file=sys.stderr)
        return 1
    annotations = load_annotations(Path(args.annotation))

    try:
        picked = draw_sample(windows, verdicts, annotations, args.seed, args.per_verdict)
    except ValueError as e:
        # một bên không có cửa sổ (hoặc thiếu hụt) -> từ chối, KHÔNG rút mẫu lệch
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    n_conf = sum(1 for r in picked if r["verdict"] == "confirmed")
    if n_conf != args.per_verdict or len(picked) - n_conf != args.per_verdict:
        print(f"LỖI: mẫu {len(picked)} (confirmed {n_conf}) != {2 * args.per_verdict} — "
              f"số cửa sổ mỗi phía không đủ {args.per_verdict}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(picked)
    tmp.replace(out)  # ghi nguyên tử

    n_overlap = sum(1 for r in picked if r["gt_overlap"] == "yes")
    print(f"đã ghi {out}: {len(picked)} mẫu "
          f"(confirmed {n_conf}, rejected {len(picked) - n_conf}; "
          f"video có GT {sum(1 for r in picked if r['video_class'] == 'anomalous')}, "
          f"trong đó chồng lấn GT {n_overlap})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
