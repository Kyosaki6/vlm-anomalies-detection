# -*- coding: utf-8 -*-
"""C7e — VLM hay typer gọi đúng TÊN loại sự kiện hơn? (giao thức chốt TRƯỚC khi chạy)

    python scripts/eval_vlm_typing.py --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    python scripts/eval_vlm_typing.py --annotation ... --dry-run     # chỉ đếm, không nạp model

CÂU HỎI DUY NHẤT: trong những cửa sổ mà tier-2 ĐÃ XÁC NHẬN là bất thường và thật sự nằm trong đoạn có
nhãn, hệ thống gọi đúng tên lớp (13 lớp UCF-Crime) bao nhiêu lần — bản của VLM, hay bản của typer?

GIAO THỨC (viết trước khi thấy số; thấy số xấu cũng không sửa):
 1. Đơn vị chấm = MỘT CỬA SỔ tier-1 (`candidate_windows[i]`) thoả cả ba điều:
    - VLM trả `is_anomaly = true`;
    - video là video BẤT THƯỜNG theo annotation;
    - cửa sổ có ÍT NHẤT MỘT MẪU CACHE nằm trong đoạn GT (>= 1/fps giây, mặc định 1 s), với đoạn GT
      hiểu là `[start, end+1)` frame như `frame_labels`. Ngưỡng 1 mẫu để cửa sổ chỉ chạm mép đoạn GT
      1 frame (33 ms) không được tính là "nằm trong sự cố"; xem `segments_seconds`.
 2. Nhãn đúng = lớp của VIDEO (nhãn gốc UCF), không dùng nhãn của đoạn.
 3. Hai dự đoán trên ĐÚNG cùng những cửa sổ đó (so sánh mới công bằng):
    (a) VLM  : `event_type` THÔ trong `results/verify/<video>/window_NNNN.json`. KHÔNG đọc
        `verify_results` của `results/run_A4c/<video>.json`, vì từ C7b (D-2) trường đó đã bị ghi đè
        bằng nhãn typer — đọc ở đó thì hai dòng bằng nhau vô nghĩa;
    (b) typer: `ZeroShotEventTyper.classify` trên trung bình embedding các mẫu trong cửa sổ, dùng
        `window_sample_slice` của `scripts/build_results.py` — cùng hàm C7d đã dùng để gán nhãn cảnh báo.
 4. Mốc so sánh = 1/13 = 7,7 % (đoán mò đều 13 lớp).
 5. LOẠI KHỎI MẪU, ĐẾM RIÊNG (không im lặng): cửa sổ xác nhận nhưng không giao GT (gồm cả video toàn
    bình thường); cửa sổ VLM loại (không phải đơn vị chấm); video không có file kết quả.
 6. Báo cáo thêm tỉ lệ VLM trả `Normal`/`Unknown`: nếu tỉ lệ này cao thì "gọi tên" của VLM phần lớn là
    KHÔNG gọi gì, và top-1 phải đọc cạnh tỉ lệ đó chứ không đứng một mình.
 7. Số `lenient` (gộp nhóm tên gần nghĩa) chỉ là thông tin phụ, dùng đúng `LENIENT_GROUPS` của C3.
 8. Mọi tỉ lệ in kèm KTC 95 % resample THEO VIDEO (`bootstrap_accuracy_ci`), vì 62 ca đến từ 56
    video — resample từng cửa sổ sẽ coi 3 cửa sổ của cùng một video là độc lập và làm KTC hẹp giả.

Script KHÔNG trả lời: "cửa sổ này có bất thường hay không" (đó là AUC của tier-1 và precision của tier-2)
— ở đây mọi đơn vị đều đã được VLM xác nhận, nên con số này chỉ nói về việc GỌI TÊN.

Không cần GPU để test: `build_cases` / `summarise` nhận dữ liệu thuần, typer được tiêm vào.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_results import (BuildResultsError, DEFAULT_TYPING_CACHE, PAIR_TOLERANCE,  # noqa: E402
                           load_or_embed_typer, load_verify_window_dir, window_sample_slice)
from eval_typer import LENIENT_GROUPS, SOURCE_FPS  # noqa: E402  (cùng định nghĩa với C3, không chép lại)
from vad.datasets.ucf_crime import Annotation, load_annotations  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached, load_embeddings  # noqa: E402
from vad.embed.siglip import DEFAULT_MODEL_ID as DEFAULT_MODEL  # noqa: E402
from vad.pipeline import DEFAULT_PROMPTS  # noqa: E402
from vad.results import RunResult  # noqa: E402
from vad.score.typing import load_typing_prompts  # noqa: E402
from vad.types import EventType, VerifyResult, Window  # noqa: E402

CALL_LABELS = ("Normal", "Unknown")     # VLM "gọi tên" nhưng thực ra không gọi lớp nào


class TypingEvalError(ValueError):
    """Đầu vào không chấm được (annotation/kết quả/cache lệch nhau, số liệu không hợp lệ)."""


# --- hình học thời gian -------------------------------------------------------------

def segments_seconds(ann: Annotation, source_fps: float = SOURCE_FPS) -> tuple[tuple[float, float], ...]:
    """Đoạn GT từ frame nguồn (30 fps) -> khoảng NỬA MỞ [start, end+1) tính bằng giây.

    `frame_labels` (vad/datasets/ucf_crime.py) đánh dấu frame `start..end` ĐÓNG hai đầu, nên quãng thời
    gian của đoạn là `[start/fps, (end+1)/fps)` — frame cuối chiếm trọn 1/30 giây của nó. Bản đầu tiên
    của hàm này lấy `end/fps`, làm mép phải hụt đúng 1 frame: một cửa sổ bắt đầu ngay sau frame cuối
    bị coi là "không giao" trong khi theo cách đánh nhãn của `frame_labels` thì nó vẫn chạm 33 ms cuối.
    Sai khác nhỏ nhưng đổi được MẪU CHẤM (62 hay 64 cửa sổ), nên phải chốt rõ; `build_cases` còn đòi
    thêm ≥1 mẫu cache nằm trong đoạn GT để một cú chạm mép không thành "nằm trong sự cố".
    """
    if source_fps <= 0:
        raise TypingEvalError(f"source_fps phải > 0, nhận {source_fps}")
    out: list[tuple[float, float]] = []
    for start, end in ann.segments:
        if start < 0 or end < start:
            raise TypingEvalError(f"{ann.video_id}: đoạn GT không hợp lệ {start}..{end}")
        out.append((start / source_fps, (end + 1) / source_fps))
    return tuple(out)


def gt_overlap_s(start_s: float, end_s: float, segments: Sequence[tuple[float, float]]) -> float:
    """Tổng thời lượng giao giữa cửa sổ và các đoạn GT (giây, đoạn GT nửa mở). Không giao -> 0.0.

    Hợp các phần giao TRƯỚC khi cộng: hai đoạn GT chồng nhau (UCF có video có nhiều đoạn sát nhau)
    không được tính hai lần, nếu không overlap có thể lớn hơn cả độ dài cửa sổ.
    """
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s < start_s:
        raise TypingEvalError(f"cửa sổ không hợp lệ: [{start_s}, {end_s}]")
    pieces: list[tuple[float, float]] = []
    for gt_start, gt_end in segments:
        lo, hi = max(start_s, gt_start), min(end_s, gt_end)
        if hi > lo:
            pieces.append((lo, hi))
    if not pieces:
        return 0.0
    pieces.sort()
    total, cur_lo, cur_hi = 0.0, pieces[0][0], pieces[0][1]
    for lo, hi in pieces[1:]:
        if lo <= cur_hi:                     # dính/chồng -> kéo dài đoạn đang cộng
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


# --- dựng ca chấm -------------------------------------------------------------------

@dataclass(frozen=True)
class CaseResult:
    """Một cửa sổ đã chấm: nhãn video, VLM gọi gì, typer gọi gì."""
    video_id: str
    window_index: int
    start_s: float
    end_s: float
    score: float
    overlap_s: float
    gt: EventType
    vlm: EventType
    typer: EventType
    typer_confidence: float

    @property
    def vlm_correct(self) -> bool:
        return self.vlm is self.gt

    @property
    def typer_correct(self) -> bool:
        return self.typer is self.gt

    @property
    def vlm_lenient(self) -> bool:
        return self.vlm in LENIENT_GROUPS.get(self.gt, frozenset({self.gt}))

    @property
    def typer_lenient(self) -> bool:
        return self.typer in LENIENT_GROUPS.get(self.gt, frozenset({self.gt}))

    @property
    def vlm_says_nothing(self) -> bool:
        return self.vlm.value in CALL_LABELS


def _check_window(video_id: str, index: int, window: Any) -> None:
    start, end, score = float(window.start_s), float(window.end_s), float(window.score)
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
        raise TypingEvalError(f"{video_id}: cửa sổ #{index} không hợp lệ [{start}, {end}]")
    if not math.isfinite(score):
        raise TypingEvalError(f"{video_id}: cửa sổ #{index} có score NaN/Inf ({score!r})")


def build_cases(runs: Mapping[str, RunResult], annotations: Mapping[str, Annotation],
                verdicts: Mapping[str, Sequence[VerifyResult]],
                source_fps: float = SOURCE_FPS,
                min_overlap_s: float = 1.0) -> tuple[list[CaseResult], dict[str, int]]:
    """Chọn đơn vị chấm theo giao thức; trả (các ca, số bị loại theo từng lý do).

    `verdicts` PHẢI là nhãn VLM THÔ đọc từ `results/verify/<video>/window_NNNN.json`, không phải
    `verify_results` trong `results/run_A4c/<video>.json`: từ C7b (quyết định D-2) file kết quả đã bị
    `apply_event_typing` GHI ĐÈ `event_type` bằng nhãn của typer, đọc ở đó thì "VLM vs typer" thành
    "typer vs typer" và hai dòng số liệu luôn bằng nhau. Đây là cái bẫy đã sập một lần khi chạy thử.

    `min_overlap_s` (mặc định 1.0 = một mẫu cache ở 1 fps): cửa sổ phải có ít nhất một mẫu NẰM TRONG
    đoạn GT mới được chấm. Không có ngưỡng này thì hai cửa sổ chỉ chạm mép đoạn GT 33 ms (một frame
    30 fps) cũng thành "nằm trong sự cố" — mẫu chấm đổi từ 62 lên 64 ca mà không ai thấy lý do.

    Cửa sổ chưa có nhãn VLM bị bỏ qua và đếm vào `windows_without_verdict` — đó là lỗi đường ống,
    không phải "VLM nói bình thường", nên phải nhìn thấy được.
    """
    if not math.isfinite(min_overlap_s) or min_overlap_s < 0:
        raise TypingEvalError(f"min_overlap_s phải hữu hạn và >= 0, nhận {min_overlap_s}")
    cases: list[CaseResult] = []
    excluded = {
        "confirmed_without_gt_overlap": 0,     # xác nhận nhưng nằm ngoài đoạn có nhãn
        "confirmed_overlap_too_small": 0,      # chỉ chạm mép GT (< min_overlap_s)
        "confirmed_on_normal_video": 0,        # video toàn bình thường, không có nhãn để so
        "rejected_windows": 0,                 # VLM loại -> không phải đơn vị chấm
        "windows_without_verdict": 0,          # chưa có tier-2 (kết quả chỉ có tier-1)
        "videos_without_results": 0,           # annotation có nhưng thiếu file kết quả
    }

    for video_id, ann in annotations.items():
        run = runs.get(video_id)
        if run is None:
            excluded["videos_without_results"] += 1
            continue
        windows = run.candidate_windows
        video_verdicts = verdicts.get(video_id, ())
        if windows and not video_verdicts:
            excluded["windows_without_verdict"] += len(windows)
            continue
        if len(video_verdicts) != len(windows):
            raise TypingEvalError(
                f"{video_id}: có {len(windows)} cửa sổ nhưng {len(video_verdicts)} nhãn VLM — lệch cặp, "
                "không ghép bừa (kiểm lại results/verify của video này).")
        segments = segments_seconds(ann, source_fps)
        for index, (window, verdict) in enumerate(zip(windows, video_verdicts)):
            _check_window(video_id, index, window)
            if not verdict.is_anomaly:
                excluded["rejected_windows"] += 1
                continue
            if not segments:
                excluded["confirmed_on_normal_video"] += 1
                continue
            overlap = gt_overlap_s(float(window.start_s), float(window.end_s), segments)
            if overlap <= 0.0:
                excluded["confirmed_without_gt_overlap"] += 1
                continue
            if overlap + 1e-9 < min_overlap_s:
                excluded["confirmed_overlap_too_small"] += 1
                continue
            cases.append(CaseResult(
                video_id=video_id, window_index=index,
                start_s=float(window.start_s), end_s=float(window.end_s), score=float(window.score),
                overlap_s=overlap, gt=ann.event_type, vlm=verdict.event_type,
                typer=EventType.UNKNOWN, typer_confidence=0.0,   # điền ở `evaluate_cases`
            ))
    return cases, excluded


def evaluate_cases(cases: Iterable[CaseResult],
                   classify: Callable[[CaseResult], Any]) -> list[CaseResult]:
    """Gán dự đoán của typer cho từng ca. `classify(case) -> TypingResult` được tiêm vào để test."""
    out: list[CaseResult] = []
    for case in cases:
        result = classify(case)
        out.append(CaseResult(
            video_id=case.video_id, window_index=case.window_index,
            start_s=case.start_s, end_s=case.end_s, score=case.score, overlap_s=case.overlap_s,
            gt=case.gt, vlm=case.vlm, typer=result.event_type,
            typer_confidence=float(result.confidence),
        ))
    return out


# --- tổng hợp số liệu ---------------------------------------------------------------

def display_path(path: str | Path, root: Path = ROOT) -> str:
    """Đường dẫn để GHI VÀO JSON CÔNG KHAI: tương đối với repo nếu nằm trong repo.

    Bản đầu tiên ghi nguyên đường dẫn tuyệt đối, tức là `E:\\Personal\\Hoc\\...\\vlm-anomalies-detection\\
    results\\run_A4c` lọt vào file theo dõi trong git — vừa lệ thuộc máy, vừa lộ cấu trúc thư mục nhà.
    Không phải bí mật, nhưng một file kết quả phải đọc được ở máy khác.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def bootstrap_accuracy_ci(results: Sequence[CaseResult], key: str, n_boot: int = 2000,
                          seed: int = 20261101, alpha: float = 0.05) -> dict[str, Any] | None:
    """KTC `alpha` cho accuracy, resample theo VIDEO (không theo cửa sổ), trả None nếu < 2 video.

    Vì sao theo video: 62 ca đến từ 56 video, có video góp 3 cửa sổ; resample từng cửa sổ coi chúng
    độc lập nên KTC hẹp giả và có thể che mất chuyện typer chỉ ngang mức đoán lớp phổ biến nhất.
    """
    if not results:
        return None
    by_video: dict[str, list[CaseResult]] = {}
    for r in results:
        by_video.setdefault(r.video_id, []).append(r)
    if len(by_video) < 2 or n_boot <= 0:
        return None
    point = sum(1 for r in results if getattr(r, f"{key}_correct")) / len(results)
    videos = sorted(by_video)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        pool = [r for _ in videos for r in by_video[rng.choice(videos)]]
        stats.append(sum(1 for r in pool if getattr(r, f"{key}_correct")) / len(pool))
    stats.sort()
    lo = stats[max(0, int(round((alpha / 2) * (n_boot - 1))))]
    hi = stats[min(n_boot - 1, int(round((1 - alpha / 2) * (n_boot - 1))))]
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "n_videos": len(by_video)}


def _bucket(results: Sequence[CaseResult], key: str) -> dict[str, Any]:
    """Một dòng số liệu cho `vlm` hoặc `typer`, để hai bên in cạnh nhau đúng cùng một cách."""
    total = len(results)
    correct = sum(bool(getattr(r, f"{key}_correct")) for r in results)
    lenient = sum(bool(getattr(r, f"{key}_lenient")) for r in results)
    per_class: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        pred = getattr(r, key)
        bucket = per_class.setdefault(r.gt.value, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(pred is r.gt)
        confusion.setdefault(r.gt.value, {})
        confusion[r.gt.value][pred.value] = confusion[r.gt.value].get(pred.value, 0) + 1
    return {
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "accuracy_ci95": bootstrap_accuracy_ci(results, key),
        "lenient_correct": lenient,
        "lenient_accuracy": (lenient / total) if total else None,
        "per_class": per_class,
        "confusion": confusion,
    }


def summarise(results: Sequence[CaseResult], excluded: Mapping[str, int] | None = None,
              n_classes: int = 13) -> dict[str, Any]:
    """Toàn bộ số liệu. Không có ca nào -> accuracy None (không bịa 0.0), `n_cases = 0`."""
    total = len(results)
    says_nothing = sum(r.vlm_says_nothing for r in results)
    return {
        "n_cases": total,
        "n_videos": len({r.video_id for r in results}),
        "chance_level": (1.0 / n_classes) if n_classes else None,
        "vlm": _bucket(results, "vlm"),
        "typer": _bucket(results, "typer"),
        "vlm_normal_or_unknown": says_nothing,
        "vlm_normal_or_unknown_share": (says_nothing / total) if total else None,
        "excluded": dict(excluded or {}),
    }


def verdict_text(summary: Mapping[str, Any], min_cases: int = 30) -> str:
    """Một câu kết luận, so cùng đơn vị (0..1). Ít ca thì không kết luận."""
    n = summary["n_cases"]
    if not n:
        return "không có ca nào"
    if n < min_cases:
        return f"chỉ {n} ca — quá ít để kết luận, chỉ kiểm tra đường ống"
    vlm, typer, chance = (summary["vlm"]["accuracy"], summary["typer"]["accuracy"],
                          summary["chance_level"])
    if vlm is None or typer is None:
        return "không có ca nào"
    if vlm >= chance * 2 and vlm > typer:
        return "VLM hơn typer rõ rệt và hơn đoán mò"
    if typer >= chance * 2 and typer > vlm:
        return "typer hơn VLM rõ rệt và hơn đoán mò"
    if max(vlm, typer) <= chance:
        return "cả hai KHÔNG hơn đoán mò — phải nói thật trên slide"
    return "có tín hiệu nhưng chưa gấp đôi đoán mò"


# --- CLI ----------------------------------------------------------------------------

def _load_results(results_dir: Path, wanted: Iterable[str]) -> dict[str, RunResult]:
    """Đọc `<results_dir>/<video>.json`; file hỏng/lạ schema -> lỗi, không bỏ qua im lặng."""
    out: dict[str, RunResult] = {}
    for video_id in wanted:
        path = results_dir / f"{video_id}.json"
        if not path.is_file():
            continue
        try:
            out[video_id] = RunResult.from_json(path)
        except (ValueError, KeyError, TypeError) as e:
            raise TypingEvalError(f"{path}: không đọc được kết quả ({e})") from e
    return out


def pair_verdicts(video_id: str, windows: Sequence[Window], raw_windows: Sequence[Window]) -> None:
    """Cửa sổ trong file tier-2 THÔ phải trùng mốc giây với cửa sổ của C7d; lệch -> dừng.

    Không kiểm `score` ở đây vì `load_verify_window_dir` đã kiểm fingerprint của từng file; cái cần
    chặn là ghép nhãn của lần chạy khác (cùng số cửa sổ nhưng khác thời điểm).
    """
    if len(windows) != len(raw_windows):
        raise TypingEvalError(
            f"{video_id}: {len(windows)} cửa sổ trong kết quả nhưng {len(raw_windows)} cửa sổ trong "
            "thư mục tier-2 — tier-2 chạy trên bản tier-1 khác, không ghép bừa.")
    for index, (mine, raw) in enumerate(zip(windows, raw_windows)):
        if (abs(mine.start_s - raw.start_s) > PAIR_TOLERANCE
                or abs(mine.end_s - raw.end_s) > PAIR_TOLERANCE):
            raise TypingEvalError(
                f"{video_id}: cửa sổ #{index} là [{mine.start_s:g}, {mine.end_s:g}] trong kết quả nhưng "
                f"[{raw.start_s:g}, {raw.end_s:g}] trong tier-2 — không ghép bừa.")


def load_raw_verdicts(verify_dir: Path, video_id: str, windows: Sequence[Window],
                      allow_fake: bool = False) -> tuple[VerifyResult, ...]:
    """Nhãn VLM THÔ của một video từ `<verify_dir>/<video>/window_NNNN.json` (mặc định results/verify).

    Dùng lại loader đã được review của C7a (kiểm fingerprint, đủ/đúng thứ tự `window_index`,
    cùng một cấu hình verifier) thay vì tự viết lại — và từ chối file tự khai `fake=true`, vì nhãn
    giả thì mọi con số của C7e vô nghĩa.
    """
    try:
        raw_windows, raw_results, fake = load_verify_window_dir(Path(verify_dir) / video_id, video_id)
    except BuildResultsError as e:
        raise TypingEvalError(f"{video_id}: {e}") from e
    if fake and not allow_fake:
        raise TypingEvalError(
            f"{video_id}: file tier-2 tự khai `fake=true` (FakeVerifier) — không chấm C7e trên nhãn giả. "
            "Nếu cố ý chạy thử, thêm --allow-fake.")
    pair_verdicts(video_id, windows, raw_windows)
    return raw_results


def _print_excluded(excluded: Mapping[str, int]) -> None:
    if not excluded:
        return
    print("\nBị loại (đếm riêng, không im lặng):")
    for name, count in excluded.items():
        print(f"  {name:<32} {count}")


def _print_summary(summary: Mapping[str, Any]) -> None:
    n = summary["n_cases"]
    if not n:
        print("Không có ca nào để chấm (xem `excluded` để biết vì sao).")
    else:
        chance = (summary["chance_level"] or 0) * 100
        print(f"\nSố ca: {n} cửa sổ / {summary['n_videos']} video")

        def _ci(bucket: Mapping[str, Any]) -> str:
            ci = bucket.get("accuracy_ci95")
            if not ci:
                return ""
            return f"  KTC95 (theo video) [{ci['lo'] * 100:.1f}, {ci['hi'] * 100:.1f}]"

        print(f"  VLM   đúng nguyên tên : {summary['vlm']['correct']}/{n}"
              f" = {(summary['vlm']['accuracy'] or 0) * 100:.1f}%"
              f"   (lenient {(summary['vlm']['lenient_accuracy'] or 0) * 100:.1f}%)"
              f"{_ci(summary['vlm'])}")
        print(f"  typer đúng nguyên tên : {summary['typer']['correct']}/{n}"
              f" = {(summary['typer']['accuracy'] or 0) * 100:.1f}%"
              f"   (lenient {(summary['typer']['lenient_accuracy'] or 0) * 100:.1f}%)"
              f"{_ci(summary['typer'])}")
        print(f"  đoán mò (1/13)        : {chance:.1f}%")
        print(f"  VLM trả Normal/Unknown: {summary['vlm_normal_or_unknown']}/{n}"
              f" = {(summary['vlm_normal_or_unknown_share'] or 0) * 100:.1f}%")
        print(f"  -> {verdict_text(summary)}")
    _print_excluded(summary["excluded"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C7e: VLM vs typer vs đoán mò — cùng những cửa sổ đã xác nhận")
    ap.add_argument("--annotation", type=Path, required=True,
                    help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "run_A4c",
                    help="nơi có <video>.json của C7d (mặc định results/run_A4c)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--typing-cache-dir", type=Path, default=DEFAULT_TYPING_CACHE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--verify-dir", type=Path, default=ROOT / "results" / "verify",
                    help="thư mục tier-2 THÔ của run_verify.py (mặc định results/verify)")
    ap.add_argument("--allow-fake", action="store_true",
                    help="cho phép nhãn tier-2 tự khai `fake=true` (chỉ để chạy thử đường ống)")
    ap.add_argument("--fps", type=float, default=1.0, help="sample_fps của cache (phải khớp lúc embed)")
    ap.add_argument("--aggregate", choices=["mean", "max"], default="mean")
    ap.add_argument("--out", type=Path, default=None,
                    help="ghi JSON (mặc định results/vlm_typing_eval.json)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N video đầu (thử nhanh)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm ca, KHÔNG nạp model, KHÔNG ghi file")
    args = ap.parse_args(argv)

    if args.fps <= 0:
        print(f"--fps phải > 0, nhận {args.fps}", file=sys.stderr)
        return 2
    if args.limit and args.out is None and not args.dry_run:
        # --limit ghi vào đúng file mặc định thì một lần "chạy nhanh" sẽ ĐÈ kết quả đầy đủ trong git.
        print("--limit chỉ để thử: phải chỉ rõ --out (đừng ghi đè results/vlm_typing_eval.json).",
              file=sys.stderr)
        return 2

    annotations = load_annotations(args.annotation)
    wanted = list(annotations)
    if args.limit:
        wanted = wanted[:args.limit]
    runs = _load_results(args.results_dir, wanted)

    # Nhãn VLM THÔ (results/verify), không phải verify_results đã bị typer ghi đè trong run_A4c.
    verdicts: dict[str, tuple[VerifyResult, ...]] = {}
    missing_verify: list[str] = []
    for video_id in wanted:
        run = runs.get(video_id)
        if run is None or not run.candidate_windows:
            continue
        if not (args.verify_dir / video_id).is_dir():
            missing_verify.append(video_id)
            continue
        verdicts[video_id] = load_raw_verdicts(args.verify_dir, video_id, run.candidate_windows,
                                               allow_fake=args.allow_fake)

    cases, excluded = build_cases(runs, {k: annotations[k] for k in wanted}, verdicts,
                                  min_overlap_s=1.0 / args.fps)
    if missing_verify:
        excluded["videos_without_verdict_dir"] = len(missing_verify)
    _, all_labels = load_typing_prompts(args.prompts)
    n_classes = len(all_labels)

    print(f"annotation: {len(annotations)} video | kết quả: {len(runs)} file | "
          f"tier-2 thô: {len(verdicts)} thư mục | ca chấm được: {len(cases)}")
    to_check = sorted({c.video_id for c in cases})
    missing_cache = [v for v in to_check if not has_cached(args.model, v, args.fps, args.cache_dir)]
    if missing_cache:
        print(f"  thiếu cache embedding cho {len(missing_cache)} video, ví dụ: {missing_cache[:3]}",
              file=sys.stderr)

    if args.dry_run:
        # Không nạp model, không ghi file: chỉ in số ca và số bị loại để kiểm giao thức.
        print(f"  {len(cases)} cửa sổ sẽ chấm trên {len(to_check)} video")
        _print_excluded(excluded)
        print("[dry-run] không nạp model, không ghi file.")
        return 0

    if not cases:
        print("Không có ca nào — không ghi file (tránh một JSON rỗng bị đọc như kết quả).", file=sys.stderr)
        return 1
    if missing_cache:
        raise TypingEvalError(
            f"{len(missing_cache)} video có ca chấm nhưng chưa embed ({args.model} @ {args.fps} fps); "
            "chạy scripts/precompute_embeddings.py trước, hoặc truyền --model/--fps cho khớp cache.")

    typer = load_or_embed_typer(args.model, prompts_path=args.prompts, cache_dir=args.typing_cache_dir)
    embs: dict[str, np.ndarray] = {}

    def classify(case: CaseResult) -> Any:
        if case.video_id not in embs:
            embs[case.video_id] = load_embeddings(args.model, case.video_id, args.fps, args.cache_dir)
        frames = embs[case.video_id]
        window = Window(video_id=case.video_id, start_s=case.start_s, end_s=case.end_s,
                        score=case.score)
        chunk = frames[window_sample_slice(window, len(frames), args.fps)]
        if chunk.size == 0:
            raise TypingEvalError(f"{case.video_id}: cửa sổ #{case.window_index} không có mẫu nào")
        return typer.classify(chunk, aggregate=args.aggregate, top_k=1)

    results = evaluate_cases(cases, classify)
    summary = summarise(results, excluded, n_classes)
    _print_summary(summary)

    out = args.out or (ROOT / "results" / "vlm_typing_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "C7e-vlm-typing",
        "protocol_version": 1,
        "annotation_file": display_path(args.annotation),
        "results_dir": display_path(args.results_dir),
        "verify_dir": display_path(args.verify_dir),
        "model": args.model,
        "sample_fps": args.fps,
        "aggregate": args.aggregate,
        "n_classes": n_classes,
        "summary": summary,
        "cases": [
            {"video_id": r.video_id, "window_index": r.window_index,
             "start_s": r.start_s, "end_s": r.end_s, "score": r.score, "overlap_s": r.overlap_s,
             "gt": r.gt.value, "vlm": r.vlm.value, "typer": r.typer.value,
             "typer_confidence": round(r.typer_confidence, 6),
             "vlm_correct": r.vlm_correct, "typer_correct": r.typer_correct,
             "vlm_lenient": r.vlm_lenient, "typer_lenient": r.typer_lenient}
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
