# -*- coding: utf-8 -*-
"""Chạy tier-2 trên từng candidate window (Task A3).

    python scripts/run_verify.py --manifest manifests/ucf_test.csv --windows-dir results/windows \\
        --out-dir results/verify --k 4

Mỗi window -> MỘT file JSON trong `results/verify/<video_id>/window_<i>.json`.
File được ghi atomic; chạy lại sẽ bỏ qua window đã xong (resume), trừ khi `--force`.
Script chỉ phụ thuộc protocol `Verifier`, nên `VLMVerifier` thật cắm vào sau mà không phải sửa.

Danh sách video đi theo MANIFEST: với mỗi `video_id` mở đúng `<windows-dir>/<video_id>.json`
(không glob `*.json`, vì thư mục có `_index.json` không phải RunResult). Video không có file windows bị bỏ qua.

`--fake` dùng `FakeVerifier` cho dev/test và BẮT BUỘC dùng `--out-dir` khác `results/verify`
để kết quả giả không lẫn vào kết quả thật.

`--continue-on-error`: một window/video lỗi thì ghi nhận, chạy tiếp; exit code 2 nếu có lỗi.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
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
    """Kết quả một lần chạy: số window đã xét, đã verify, đã bỏ qua, bị lỗi."""

    examined: int = 0
    verified: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


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


def windows_file_for(windows_dir: str | Path, video_id: str) -> Path | None:
    """`<windows_dir>/<video_id>.json`, hoặc None nếu video không có candidate window.

    Đi theo `video_id` thay vì glob `*.json`: thư mục windows còn `_index.json` (index tier-1)
    không phải RunResult.
    """
    _check_video_id(video_id)
    path = Path(windows_dir) / f"{video_id}.json"
    return path if path.is_file() else None


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


def plan_window_frames(video_path: str | Path, windows: Sequence[Window],
                       k: int) -> list[tuple[list[float], list[int]] | ValueError]:
    """Mỗi window -> (mốc thời gian, chỉ số frame) hoặc ValueError nếu window không lấy được k frame.

    Lỗi từng window được TRẢ VỀ (không raise) để người gọi chọn dừng hay đi tiếp.
    """
    n_frames, fps = probe_video(video_path)
    plans: list[tuple[list[float], list[int]] | ValueError] = []
    for w in windows:
        try:
            times = pick_frame_times(w.start_s, w.end_s, k)
            plans.append((times, frame_indices_for_times(times, fps, n_frames)))
        except ValueError as e:
            plans.append(e)
    return plans


def verifier_name(verifier: Verifier | None) -> str:
    """Tên đưa vào fingerprint/metadata. FakeVerifier luôn là 'fake'."""
    if verifier is None:
        return "dry-run"
    if isinstance(verifier, FakeVerifier):
        return "fake"
    return str(getattr(verifier, "model_id", type(verifier).__name__))


def verifier_config(verifier: Verifier | None) -> dict[str, Any]:
    """Mọi thứ ngoài `model_id` làm đổi đầu ra của VLM; đổi cái nào là phải verify lại.

    `quantization`, `max_attempts`, `max_new_tokens` và hash prompt (sha1[:8] của system+user,
    cùng công thức `bench_vlm.py`: `3d71fdba` ở A1). Verifier không có thuộc tính nào thì trường đó là None.
    """
    if verifier is None or isinstance(verifier, FakeVerifier):
        return {}
    system = getattr(verifier, "_system", None)
    user = getattr(verifier, "_user_prompt", None)
    prompt_hash = (hashlib.sha1((str(system) + str(user)).encode("utf-8")).hexdigest()[:8]
                   if system is not None and user is not None else None)
    return {
        "quantization": getattr(verifier, "quantization", None),
        "max_attempts": getattr(verifier, "max_attempts", None),
        "max_new_tokens": getattr(verifier, "max_new_tokens", None),
        "prompt_hash": prompt_hash,
    }


def make_fingerprint(window: Window, k: int, name: str,
                     config: dict[str, Any] | None = None) -> str:
    """Hash cấu hình + window để quyết định resume có hợp lệ hay không."""
    payload = {
        "video_id": window.video_id,
        "start_s": float(window.start_s),
        "end_s": float(window.end_s),
        "score": float(window.score),
        "k": int(k),
        "verifier": str(name),
        "verifier_config": dict(config or {}),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_commit() -> str:
    """Short sha của HEAD để ghi vào metadata; không có git -> 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


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
               dry_run: bool = False, video_ids: Sequence[str] | None = None,
               continue_on_error: bool = False) -> VerifySummary:
    """Chạy verify cho mọi window của các video trong manifest; trả thống kê.

    Mặc định lỗi đầu tiên dừng cả lượt (resume cứu được lần sau). Với `continue_on_error=True`
    lỗi của một window/video được ghi vào `summary.errors` + `summary.failed` rồi chạy tiếp.
    """
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
    wanted = list(dict.fromkeys(video_ids)) if video_ids else None
    if wanted is not None:
        unknown = [v for v in wanted if v not in manifest]
        if unknown:
            raise ValueError(f"--video-id không có trong manifest {manifest_path}: {unknown}")
    name = verifier_name(verifier)
    config = verifier_config(verifier)
    commit = git_commit() if not dry_run else ""
    summary = VerifySummary()

    def fail(message: str, exc: Exception) -> None:
        if not continue_on_error:
            raise exc
        summary.failed += 1
        summary.errors.append(message)
        print(f"[lỗi] {message}", file=sys.stderr)

    for video_id, video_path in manifest.items():
        if wanted is not None and video_id not in wanted:
            continue
        if limit is not None and summary.examined >= limit:
            break
        windows_file = windows_file_for(windows_dir, video_id)
        if windows_file is None:
            continue
        try:
            source = RunResult.from_json(windows_file)
            if source.video_id != video_id:
                raise ValueError(
                    f"{windows_file} có video_id {source.video_id!r}, manifest nói {video_id!r}"
                )
        except (TypeError, ValueError, OSError) as e:
            fail(f"{video_id}: không đọc được windows ({e})", e)
            continue

        pending: list[tuple[int, Window, str, Path]] = []
        for index, window in enumerate(source.candidate_windows):
            if limit is not None and summary.examined >= limit:
                break
            summary.examined += 1
            try:
                _validate_window(video_id, window, k)
            except ValueError as e:
                fail(f"{video_id}[{index}]: {e}", e)
                continue
            fingerprint = make_fingerprint(window, k, name, config)
            target = output_path(out_dir, video_id, index)
            if not force and is_completed(target, video_id, index, window, fingerprint):
                summary.skipped += 1
                continue
            pending.append((index, window, fingerprint, target))

        if not pending:
            continue
        if dry_run:
            print(f"[dry-run] {video_id}: {len(pending)} window cần verify")
            continue

        # Chỉ cần file video khi còn window phải verify (máy chỉ giữ cache verify vẫn resume được).
        if not video_path.is_file():
            fail(f"{video_id}: không tìm thấy video {video_path}",
                 FileNotFoundError(f"Không tìm thấy video của {video_id!r}: {video_path}"))
            continue
        try:
            plans = plan_window_frames(video_path, [w for _, w, _, _ in pending], k)
        except (OSError, RuntimeError) as e:
            fail(f"{video_id}: không probe được video ({e})", e)
            continue

        usable = [(item, plan) for item, plan in zip(pending, plans)
                  if not isinstance(plan, ValueError)]
        for item, plan in zip(pending, plans):
            if isinstance(plan, ValueError):
                fail(f"{video_id}[{item[0]}]: {plan}", plan)
        if not usable:
            continue
        try:
            frame_by_index = read_frames_at(
                video_path, [i for _, (_, indices) in usable for i in indices])
        except (OSError, RuntimeError) as e:
            fail(f"{video_id}: không đọc được frame ({e})", e)
            continue

        for (index, window, fingerprint, target), (times, indices) in usable:
            context = {
                "video_id": video_id,
                "start_s": window.start_s,
                "end_s": window.end_s,
                "score": window.score,
                "window_index": index,
            }
            try:
                result = verifier.verify([frame_by_index[i] for i in indices], context)
                if not isinstance(result, VerifyResult):
                    raise TypeError(
                        f"Verifier phải trả VerifyResult, nhận {type(result).__name__}"
                    )
                metadata = make_metadata(
                    model=name,
                    commit=commit,
                    window_index=index,
                    k=k,
                    frame_times_s=list(times),
                    source_windows=str(windows_file),
                    fingerprint=fingerprint,
                    fake=isinstance(verifier, FakeVerifier),
                    verifier_config=config,
                )
                _build_result(video_id, source.sample_fps, window, result,
                              metadata).to_json(target)
            except Exception as e:  # noqa: BLE001 - ghi nhận từng window, không nuốt im
                fail(f"{video_id}[{index}]: {type(e).__name__}: {e}", e)
                continue
            summary.verified += 1

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
    parser.add_argument("--continue-on-error", action="store_true",
                        help="window/video lỗi thì ghi nhận rồi chạy tiếp (exit 2 nếu có lỗi)")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k phải >= 1")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit phải >= 0")

    if args.fake:
        if not args.dry_run and Path(args.out_dir).resolve() == DEFAULT_VERIFY_DIR.resolve():
            parser.error("--fake phải dùng --out-dir khác results/verify để không ghi kết quả giả")
        verifier: Verifier | None = FakeVerifier()
    elif args.dry_run and not args.verifier_model:
        verifier = None  # không biết model nào -> fingerprint "dry-run", mọi window đều tính là pending
    else:
        if not args.verifier_model:
            parser.error("cần --verifier-model khi không dùng --fake")
        # VLMVerifier lazy-load model ở lần verify đầu, nên dựng ở đây rẻ và cho fingerprint
        # đúng: `--dry-run --verifier-model X` thấy được window nào đã xong.
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
            continue_on_error=args.continue_on_error,
        )
    except (ValueError, OSError, RuntimeError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1

    print(f"xong: {summary.examined} window xét, {summary.verified} verify, "
          f"{summary.skipped} bỏ qua, {summary.failed} lỗi")
    return 2 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
