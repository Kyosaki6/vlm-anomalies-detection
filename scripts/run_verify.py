# -*- coding: utf-8 -*-
"""Chạy tier-2 trên từng candidate window (Task A3).

    python scripts/run_verify.py --manifest manifests/ucf_test.csv --windows-dir results/windows \\
        --out-dir results/verify --k 4

Mỗi window -> MỘT file JSON trong `results/verify/<video_id>/window_<i>.json`.
File được ghi atomic; chạy lại sẽ bỏ qua window đã xong (resume), trừ khi `--force`.
Script chỉ phụ thuộc protocol `Verifier`, nên `VLMVerifier` thật cắm vào sau mà không phải sửa.

`--fake` dùng `FakeVerifier` cho dev/test và BẮT BUỘC dùng `--out-dir` khác `results/verify`
để kết quả giả không lẫn vào kết quả thật.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("Thiếu opencv: pip install -r requirements.txt") from e

from vad.fakes import FakeVerifier  # noqa: E402
from vad.interfaces import Verifier  # noqa: E402
from vad.io.video import probe_video  # noqa: E402
from vad.results import DEFAULT_RESULTS_DIR, RunResult, make_metadata  # noqa: E402
from vad.types import VerifyResult, Window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_DIR = DEFAULT_RESULTS_DIR / "windows"
DEFAULT_VERIFY_DIR = DEFAULT_RESULTS_DIR / "verify"


@dataclass
class VerifySummary:
    """Kết quả một lần chạy: số window đã xét, đã verify, đã bỏ qua."""

    examined: int = 0
    verified: int = 0
    skipped: int = 0


def _check_video_id(video_id: str) -> str:
    """`video_id` đi thẳng vào tên thư mục -> chặn đường dẫn (`../x`, `a/b`, rỗng)."""
    if (not video_id or video_id in {".", ".."} or "/" in video_id or "\\" in video_id
            or "\x00" in video_id):
        raise ValueError(f"video_id không hợp lệ để làm tên thư mục: {video_id!r}")
    return video_id


def load_manifest(path: str | Path) -> dict[str, Path]:
    """Đọc manifest CSV -> {video_id: đường dẫn video}. Trùng id hoặc thiếu cột -> lỗi."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        if not {"video_id", "path"} <= fields:
            raise ValueError(f"Manifest thiếu cột video_id/path: {p}")
        out: dict[str, Path] = {}
        for row in reader:
            video_id = str(row.get("video_id") or "").strip()
            video_path = str(row.get("path") or "").strip()
            if not video_id or not video_path:
                raise ValueError(f"Manifest có dòng thiếu video_id/path: {row!r}")
            if video_id in out:
                raise ValueError(f"Manifest có video_id trùng nhau: {video_id!r}")
            out[video_id] = Path(video_path)
    return out


def list_window_files(windows_dir: str | Path) -> list[Path]:
    """Các file `results/windows/*.json`, sắp xếp ổn định. Không có file -> lỗi rõ ràng."""
    base = Path(windows_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục windows: {base}")
    files = sorted(p for p in base.glob("*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"Không có file windows nào trong: {base}")
    return files


def pick_frame_times(start_s: float, end_s: float, k: int) -> list[float]:
    """`k` mốc thời gian cách đều trong (start_s, end_s), không lấy hai biên."""
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k phải là số nguyên >= 1, nhận {k!r}")
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError(f"start_s/end_s phải hữu hạn, nhận ({start_s!r}, {end_s!r})")
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"Cửa sổ không hợp lệ (cần 0 <= start_s < end_s): ({start_s}, {end_s})")
    duration = end_s - start_s
    return [start_s + (i + 0.5) * duration / k for i in range(k)]


def frame_indices_for_times(times: Sequence[float], fps: float, n_frames: int) -> list[int]:
    """Đổi thời điểm -> chỉ số frame gốc. Trùng frame hoặc ra ngoài video -> lỗi."""
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps phải hữu hạn và > 0, nhận {fps!r}")
    if not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(f"n_frames phải là số nguyên >= 1, nhận {n_frames!r}")
    duration = n_frames / fps
    indices: list[int] = []
    for t in times:
        if not math.isfinite(t) or t < 0 or t > duration + 1e-9:
            raise ValueError(f"Thời điểm {t!r} nằm ngoài video (0..{duration:.3f}s)")
        index = int(round(t * fps))
        if index >= n_frames:
            if t <= duration + 1e-9:
                index = n_frames - 1
            else:
                raise ValueError(f"Thời điểm {t!r} vượt quá frame cuối ({n_frames - 1})")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(
            "Các mốc thời gian trỏ vào cùng frame; cửa sổ quá ngắn để lấy k frame khác nhau"
        )
    return indices


def read_frames_at(video_path: str | Path, frame_indices: Sequence[int]) -> dict[int, Any]:
    """Đọc video một lượt tuần tự, chỉ retrieve các frame cần. Trả {index: RGB uint8}."""
    targets = sorted({int(i) for i in frame_indices})
    if not targets:
        return {}
    if targets[0] < 0:
        raise ValueError(f"frame index phải >= 0, nhận {targets[0]}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV không mở được video: {video_path}")
        out: dict[int, Any] = {}
        cursor = 0
        for target in targets:
            while cursor <= target:
                if not cap.grab():
                    raise RuntimeError(
                        f"Video kết thúc trước frame {target}: {video_path} "
                        f"(container báo sai n_frames?)"
                    )
                cursor += 1
            ok, bgr = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Không retrieve được frame {target}: {video_path}")
            out[target] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return out
    finally:
        cap.release()


def extract_window_frames(video_path: str | Path, windows: Sequence[Window],
                          k: int) -> tuple[list[list[Any]], list[list[float]]]:
    """Cắt `k` frame cho mỗi window, giải mã video đúng MỘT lượt."""
    if not windows:
        return [], []
    n_frames, fps = probe_video(video_path)
    times_by_window = [pick_frame_times(w.start_s, w.end_s, k) for w in windows]
    indices_by_window = [frame_indices_for_times(times, fps, n_frames) for times in times_by_window]
    all_indices = [index for indices in indices_by_window for index in indices]
    frame_by_index = read_frames_at(video_path, all_indices)
    frames_by_window = [[frame_by_index[index] for index in indices] for indices in indices_by_window]
    return frames_by_window, times_by_window


def verifier_name(verifier: Verifier | None) -> str:
    """Tên đưa vào fingerprint/metadata. FakeVerifier luôn là 'fake'."""
    if verifier is None:
        return "dry-run"
    if isinstance(verifier, FakeVerifier):
        return "fake"
    return str(getattr(verifier, "model_id", type(verifier).__name__))


def make_fingerprint(window: Window, k: int, name: str) -> str:
    """Hash cấu hình + window để quyết định resume có hợp lệ hay không."""
    payload = {
        "video_id": window.video_id,
        "start_s": float(window.start_s),
        "end_s": float(window.end_s),
        "score": float(window.score),
        "k": int(k),
        "verifier": str(name),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def output_path(out_dir: str | Path, video_id: str, window_index: int) -> Path:
    """Đường dẫn cố định cho một window; chặn video_id kiểu đường dẫn."""
    _check_video_id(video_id)
    if not isinstance(window_index, int) or window_index < 0:
        raise ValueError(f"window_index phải là số nguyên >= 0, nhận {window_index!r}")
    return Path(out_dir) / video_id / f"window_{window_index:04d}.json"


def _validate_window(video_id: str, window: Window, k: int) -> None:
    if window.video_id != video_id:
        raise ValueError(
            f"Window có video_id {window.video_id!r} khác file windows {video_id!r}"
        )
    if not (math.isfinite(window.start_s) and math.isfinite(window.end_s)
            and math.isfinite(window.score)):
        raise ValueError(f"Window có start_s/end_s/score không hữu hạn: {window}")
    if window.start_s < 0 or window.end_s <= window.start_s:
        raise ValueError(f"Window không hợp lệ (cần 0 <= start_s < end_s): {window}")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k phải là số nguyên >= 1, nhận {k!r}")


def is_completed(path: str | Path, video_id: str, window_index: int, window: Window,
                 fingerprint: str) -> bool:
    """File output đã tồn tại, đọc được và khớp đúng window + fingerprint."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        loaded = RunResult.from_json(p)
    except (TypeError, ValueError, OSError):
        return False
    if loaded.video_id != video_id:
        return False
    if len(loaded.candidate_windows) != 1 or loaded.candidate_windows[0] != window:
        return False
    if len(loaded.verify_results) != 1:
        return False
    return (loaded.metadata.get("window_index") == window_index
            and loaded.metadata.get("fingerprint") == fingerprint)


def _build_result(video_id: str, sample_fps: float, window: Window, result: VerifyResult,
                  metadata: dict[str, Any]) -> RunResult:
    return RunResult(
        video_id=video_id,
        sample_fps=sample_fps,
        anomaly_scores=(),
        candidate_windows=(window,),
        verify_results=(result,),
        alerts=(),
        metadata=metadata,
    )


def run_verify(manifest_path: str | Path, windows_dir: str | Path = DEFAULT_WINDOWS_DIR,
               out_dir: str | Path = DEFAULT_VERIFY_DIR, verifier: Verifier | None = None,
               k: int = 4, limit: int | None = None, force: bool = False,
               dry_run: bool = False, video_ids: Sequence[str] | None = None) -> VerifySummary:
    """Chạy verify cho mọi window trong `windows_dir`; trả thống kê."""
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k phải là số nguyên >= 1, nhận {k!r}")
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError(f"limit phải là số nguyên >= 0 hoặc None, nhận {limit!r}")
    if not dry_run and verifier is None:
        raise ValueError("Cần verifier khi không chạy --dry-run")
    if (not dry_run and isinstance(verifier, FakeVerifier)
            and Path(out_dir).resolve() == DEFAULT_VERIFY_DIR.resolve()):
        raise ValueError(
            "FakeVerifier không được ghi vào results/verify; "
            "dùng --out-dir khác để kết quả giả không lẫn vào kết quả thật"
        )

    manifest = load_manifest(manifest_path)
    wanted = set(video_ids) if video_ids else None
    files = list_window_files(windows_dir)
    name = verifier_name(verifier)
    summary = VerifySummary()

    for windows_file in files:
        source = RunResult.from_json(windows_file)
        if wanted is not None and source.video_id not in wanted:
            continue
        _check_video_id(source.video_id)
        video_path = manifest.get(source.video_id)
        if video_path is None:
            raise ValueError(f"Không có {source.video_id!r} trong manifest {manifest_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video của {source.video_id!r}: {video_path}")

        pending: list[tuple[int, Window, str, Path]] = []
        stop = False
        for index, window in enumerate(source.candidate_windows):
            if limit is not None and summary.examined >= limit:
                stop = True
                break
            summary.examined += 1
            _validate_window(source.video_id, window, k)
            fingerprint = make_fingerprint(window, k, name)
            target = output_path(out_dir, source.video_id, index)
            if not force and is_completed(target, source.video_id, index, window, fingerprint):
                summary.skipped += 1
                continue
            pending.append((index, window, fingerprint, target))

        if stop and not pending:
            break

        if dry_run:
            if pending:
                print(f"[dry-run] {source.video_id}: {len(pending)} window cần verify")
            if stop:
                break
            continue

        frames_by_window, times_by_window = extract_window_frames(
            video_path, [window for _, window, _, _ in pending], k
        )
        for (index, window, fingerprint, target), frames, times in zip(
                pending, frames_by_window, times_by_window):
            context = {
                "video_id": source.video_id,
                "start_s": window.start_s,
                "end_s": window.end_s,
                "score": window.score,
                "window_index": index,
            }
            result = verifier.verify(frames, context)
            if not isinstance(result, VerifyResult):
                raise TypeError(
                    f"Verifier phải trả VerifyResult, nhận {type(result).__name__}"
                )
            metadata = make_metadata(
                model=name,
                commit="",
                window_index=index,
                k=k,
                frame_times_s=list(times),
                source_windows=str(windows_file),
                fingerprint=fingerprint,
                fake=isinstance(verifier, FakeVerifier),
            )
            _build_result(source.video_id, source.sample_fps, window, result, metadata).to_json(target)
            summary.verified += 1

        if stop:
            break

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="CSV có cột video_id,path,...")
    parser.add_argument("--windows-dir", default=str(DEFAULT_WINDOWS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_VERIFY_DIR))
    parser.add_argument("--k", type=int, default=4, help="số frame mỗi window (mặc định 4)")
    parser.add_argument("--limit", type=int, default=None, help="chỉ xét N window đầu tiên")
    parser.add_argument("--video-id", action="append", default=None,
                        help="chỉ chạy video này (lặp lại được)")
    parser.add_argument("--force", action="store_true", help="làm lại cả window đã có file")
    parser.add_argument("--fake", action="store_true", help="dùng FakeVerifier (chỉ dev/test)")
    parser.add_argument("--verifier-model", default="", help="HF id của VLM thật")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in việc sẽ làm")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k phải >= 1")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit phải >= 0")

    if args.fake:
        if not args.dry_run and Path(args.out_dir).resolve() == DEFAULT_VERIFY_DIR.resolve():
            parser.error("--fake phải dùng --out-dir khác results/verify để không ghi kết quả giả")
        verifier: Verifier | None = FakeVerifier()
    elif args.dry_run:
        verifier = None
    else:
        if not args.verifier_model:
            parser.error("cần --verifier-model khi không dùng --fake")
        from vad.verify.vlm import VLMVerifier
        verifier = VLMVerifier(model_id=args.verifier_model)

    try:
        summary = run_verify(
            manifest_path=args.manifest,
            windows_dir=args.windows_dir,
            out_dir=args.out_dir,
            verifier=verifier,
            k=args.k,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
            video_ids=args.video_id,
        )
    except (ValueError, OSError, RuntimeError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    print(f"xong: {summary.examined} window xét, {summary.verified} verify, "
          f"{summary.skipped} bỏ qua")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
