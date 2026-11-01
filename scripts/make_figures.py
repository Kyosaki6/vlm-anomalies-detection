# -*- coding: utf-8 -*-
"""C7f (= E6) — 5 hình "score theo thời gian" cho báo cáo, CHỌN THEO LUẬT, không chọn bằng mắt.

    python scripts/make_figures.py --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    python scripts/make_figures.py --annotation ... --dry-run      # chỉ in bảng chọn, không vẽ

Vì sao phải có luật: hình trong báo cáo là bằng chứng, không phải minh hoạ. Nếu người viết tự chọn
"video đẹp" thì hình chỉ chứng minh người viết giỏi chọn. Năm hình dưới đây phủ cả thành công lẫn
thất bại, và mỗi hình có một câu luật in ra kèm số liệu để người đọc tự kiểm:

  1. `confirmed_hit`        video bất thường, cửa sổ VLM xác nhận, giao GT nhiều nhất
                            -> "hệ thống bắt đúng và gọi đúng chỗ".
  2. `removed_false_alarm`  video toàn bình thường, cửa sổ bị VLM loại, điểm tier-1 cao nhất
                            -> "tier-1 báo động nhầm, tier-2 dập được".
  3. `surviving_false_alarm` video toàn bình thường, cửa sổ ĐƯỢC xác nhận, điểm cao nhất
                            -> "cả hai tầng cùng sai" (nếu danh mục rỗng thì in rõ là rỗng).
  4. `missed_incident`      video bất thường nhưng KHÔNG có cửa sổ nào (tier-1 không vượt ngưỡng),
                            chọn video có điểm trung bình trong đoạn GT cao nhất
                            -> "sự cố xảy ra mà không có cảnh báo".
  5. `lost_true_positive`   video bất thường, cửa sổ bị VLM loại nhưng giao GT nhiều nhất
                            -> "tier-2 loại mất một cửa sổ đúng".

Mỗi hình: đường score tier-1, đường ngưỡng τ, các đoạn GT tô nền, các cửa sổ candidate tô màu theo
kết luận của VLM. Cửa sổ được chọn tô đậm hơn để thấy ngay nó nằm ở đâu.

Ghi ra `results/figures/*.png` + `results/figures/selection.json` (luật, số liệu, caption của VLM).
Danh mục rỗng KHÔNG bị bỏ qua im lặng: in ra "rỗng" và ghi vào JSON.

`matplotlib` chỉ được import khi thật sự vẽ, nên test chọn-hình chạy được trên máy không có nó.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_vlm_typing import display_path, gt_overlap_s, segments_seconds  # noqa: E402
from vad.datasets.ucf_crime import Annotation, load_annotations  # noqa: E402
from vad.results import RunResult  # noqa: E402
from vad.types import VerifyResult, Window  # noqa: E402

CATEGORIES = ("confirmed_hit", "removed_false_alarm", "surviving_false_alarm",
              "missed_incident", "lost_true_positive")

TITLES = {
    "confirmed_hit": "Bắt đúng — cửa sổ VLM xác nhận, giao GT nhiều nhất",
    "removed_false_alarm": "Báo động nhầm bị dập — video bình thường, VLM loại cửa sổ điểm cao nhất",
    "surviving_false_alarm": "Báo động nhầm còn sống — video bình thường mà VLM vẫn xác nhận",
    "missed_incident": "Sự cố bị bỏ sót — video bất thường nhưng không có cửa sổ nào",
    "lost_true_positive": "Mất một cửa sổ đúng — VLM loại cửa sổ có giao GT",
}

RULES = {
    "confirmed_hit": "video bất thường; cửa sổ is_anomaly=true; max overlap GT (hòa -> điểm cao hơn)",
    "removed_false_alarm": "video toàn bình thường; is_anomaly=false; max score tier-1",
    "surviving_false_alarm": "video toàn bình thường; is_anomaly=true; max score tier-1",
    "missed_incident": "video bất thường, KHÔNG có candidate window; max điểm trung bình trong đoạn GT",
    "lost_true_positive": "video bất thường; is_anomaly=false; overlap GT > 0; max overlap (hòa -> score)",
}


class FigureError(ValueError):
    """Đầu vào không chọn/vẽ được hình (thiếu kết quả, ngưỡng không rõ, số liệu hỏng)."""


@dataclass(frozen=True)
class Pick:
    """Một hình: chọn video/cửa sổ nào, vì sao, và số liệu để người đọc tự kiểm."""
    category: str
    video_id: str
    label: str                      # lớp theo annotation ('Normal' cho video bình thường)
    rule: str
    window_index: int | None        # None = danh mục 'missed_incident' (không có cửa sổ)
    start_s: float | None
    end_s: float | None
    score: float | None
    overlap_s: float | None
    evidence: float                 # con số dùng để xếp hạng (overlap giây / score / điểm TB)
    n_windows: int
    is_confirmed: bool | None
    vlm_type: str | None
    caption: str | None
    reason: str                     # một câu tiếng Việt giải thích lựa chọn
    missing: bool = False           # True = danh mục rỗng, ghi lại chứ không giả hình

    @property
    def window_duration_s(self) -> float | None:
        if self.start_s is None or self.end_s is None:
            return None
        return self.end_s - self.start_s

    @property
    def overlap_fraction(self) -> float | None:
        """Phần cửa sổ nằm trong GT (0..1). Ghi kèm để thấy luật "nhiều giây nhất" ưu ái cửa sổ dài."""
        duration = self.window_duration_s
        if duration is None or not duration or self.overlap_s is None:
            return None
        return self.overlap_s / duration


def _verdict(run: RunResult, index: int) -> VerifyResult | None:
    if not run.verify_results:
        return None
    return run.verify_results[index]


def _overlap(window: Window, segments: Sequence[tuple[float, float]]) -> float:
    return gt_overlap_s(float(window.start_s), float(window.end_s), segments)


def _normal_video(ann: Annotation) -> bool:
    return not ann.segments


def select_cases(runs: Mapping[str, RunResult],
                 annotations: Mapping[str, Annotation]) -> tuple[list[Pick], list[str]]:
    """Chọn 5 hình theo đúng luật trong `RULES`; trả (các lựa chọn, tên danh mục rỗng).

    Chỉ xét video có trong `annotations` (để có nhãn cho hình) và có file kết quả. Video bất thường
    nhưng không có cửa sổ nào vẫn là ứng viên của `missed_incident` — đó chính là điều hình 4 nói.
    """
    prepared: dict[str, tuple[Annotation, RunResult, tuple[tuple[float, float], ...]]] = {}
    for video_id, run in runs.items():
        ann = annotations.get(video_id)
        if ann is None:
            continue
        prepared[video_id] = (ann, run, segments_seconds(ann))

    picks: list[Pick] = []
    empty: list[str] = []

    def best(candidates: list[Pick], category: str) -> None:
        # Nhóm ứng viên ghi thẳng vào câu chọn: "cao nhất trong N" nói rõ N là bao nhiêu, và N = 1
        # thì không còn nghe như một cuộc thi (bản đầu cho hình 3 chỉ có đúng một ứng viên).
        if candidates and category != "missed_incident":      # hình 4 đã có "trong N video" riêng
            size = len(candidates)
            pool = "ứng viên duy nhất (1/1)" if size == 1 else f"1 trong {size} ứng viên"
            candidates = [replace(c, reason=f"{c.reason}; {pool}") for c in candidates]
        if not candidates:
            empty.append(category)
            picks.append(Pick(
                category=category, video_id="", label="", rule=RULES[category],
                window_index=None, start_s=None, end_s=None, score=None, overlap_s=None,
                evidence=0.0, n_windows=0, is_confirmed=None, vlm_type=None, caption=None,
                reason="danh mục rỗng trên dữ liệu hiện tại — không vẽ hình thay thế",
                missing=True))
            return
        picks.append(max(candidates, key=lambda p: (p.evidence, p.score or 0.0)))

    # 1) bắt đúng: cửa sổ xác nhận, giao GT nhiều nhất
    hits: list[Pick] = []
    for video_id, (ann, run, segments) in prepared.items():
        if _normal_video(ann) or not run.verify_results:
            continue
        for index, window in enumerate(run.candidate_windows):
            verdict = run.verify_results[index]
            if not verdict.is_anomaly:
                continue
            overlap = _overlap(window, segments)
            if overlap <= 0:
                continue
            hits.append(Pick(
                category="confirmed_hit", video_id=video_id, label=ann.label,
                rule=RULES["confirmed_hit"], window_index=index,
                start_s=float(window.start_s), end_s=float(window.end_s), score=float(window.score),
                overlap_s=overlap, evidence=overlap, n_windows=len(run.candidate_windows),
                is_confirmed=True, vlm_type=verdict.event_type.value, caption=verdict.caption,
                reason=f"giao GT {overlap:.1f} giây — nhiều nhất trong các cửa sổ được xác nhận"))
    best(hits, "confirmed_hit")

    # 2) báo động nhầm bị dập: video bình thường, cửa sổ bị loại, điểm cao nhất
    removed: list[Pick] = []
    for video_id, (ann, run, segments) in prepared.items():
        if not _normal_video(ann) or not run.verify_results:
            continue
        for index, window in enumerate(run.candidate_windows):
            verdict = run.verify_results[index]
            if verdict.is_anomaly:
                continue
            removed.append(Pick(
                category="removed_false_alarm", video_id=video_id, label=ann.label,
                rule=RULES["removed_false_alarm"], window_index=index,
                start_s=float(window.start_s), end_s=float(window.end_s), score=float(window.score),
                overlap_s=None, evidence=float(window.score), n_windows=len(run.candidate_windows),
                is_confirmed=False, vlm_type=verdict.event_type.value, caption=verdict.caption,
                reason=f"điểm tier-1 {window.score:.3f} — cao nhất trong các cửa sổ bị VLM loại "
                       "trên video bình thường"))
    best(removed, "removed_false_alarm")

    # 3) báo động nhầm còn sống: video bình thường, cửa sổ được xác nhận, điểm cao nhất
    surviving: list[Pick] = []
    for video_id, (ann, run, segments) in prepared.items():
        if not _normal_video(ann) or not run.verify_results:
            continue
        for index, window in enumerate(run.candidate_windows):
            verdict = run.verify_results[index]
            if not verdict.is_anomaly:
                continue
            surviving.append(Pick(
                category="surviving_false_alarm", video_id=video_id, label=ann.label,
                rule=RULES["surviving_false_alarm"], window_index=index,
                start_s=float(window.start_s), end_s=float(window.end_s), score=float(window.score),
                overlap_s=None, evidence=float(window.score), n_windows=len(run.candidate_windows),
                is_confirmed=True, vlm_type=verdict.event_type.value, caption=verdict.caption,
                reason=f"điểm tier-1 {window.score:.3f} — cao nhất trong các cửa sổ VLM xác nhận "
                       "trên video bình thường"))
    best(surviving, "surviving_false_alarm")

    # 4) bỏ sót: video bất thường, không có cửa sổ nào; xếp theo điểm trung bình trong đoạn GT
    missed: list[Pick] = []
    for video_id, (ann, run, segments) in prepared.items():
        if _normal_video(ann) or run.candidate_windows:
            continue
        if not segments:
            continue
        fps = run.sample_fps or 1.0
        inside: list[float] = []
        for start, end in segments:
            i0 = max(0, int(round(start * fps)))
            i1 = min(len(run.anomaly_scores), int(round(end * fps)) + 1)
            inside.extend(run.anomaly_scores[i0:i1])
        if not inside:
            continue
        mean_score = sum(inside) / len(inside)
        missed.append(Pick(
            category="missed_incident", video_id=video_id, label=ann.label,
            rule=RULES["missed_incident"], window_index=None, start_s=None, end_s=None,
            score=None, overlap_s=None, evidence=mean_score, n_windows=0,
            is_confirmed=None, vlm_type=None, caption=None,
            reason=f"không có cửa sổ nào vượt ngưỡng; điểm trung bình trong đoạn GT {mean_score:.3f} "
                   f"— cao nhất trong POOL video bất thường không có cửa sổ"))
    best(missed, "missed_incident")
    if missed:
        pool = len(missed)
        picks[-1] = replace(picks[-1], reason=picks[-1].reason.replace("POOL", f"{pool}"))

    # 5) mất cửa sổ đúng: video bất thường, cửa sổ bị loại nhưng giao GT
    lost: list[Pick] = []
    for video_id, (ann, run, segments) in prepared.items():
        if _normal_video(ann) or not run.verify_results:
            continue
        for index, window in enumerate(run.candidate_windows):
            verdict = run.verify_results[index]
            if verdict.is_anomaly:
                continue
            overlap = _overlap(window, segments)
            # cùng ngưỡng với C7e: phải có ≥ 1 mẫu cache nằm trong GT, không tính cú chạm mép 33 ms
            if overlap + 1e-9 < 1.0 / (run.sample_fps or 1.0):
                continue
            lost.append(Pick(
                category="lost_true_positive", video_id=video_id, label=ann.label,
                rule=RULES["lost_true_positive"], window_index=index,
                start_s=float(window.start_s), end_s=float(window.end_s), score=float(window.score),
                overlap_s=overlap, evidence=overlap, n_windows=len(run.candidate_windows),
                is_confirmed=False, vlm_type=verdict.event_type.value, caption=verdict.caption,
                reason=f"giao GT {overlap:.1f} giây nhưng VLM loại"
                       f" ({verdict.event_type.value}) — mất một cửa sổ đúng"))
    best(lost, "lost_true_positive")

    order = {name: i for i, name in enumerate(CATEGORIES)}
    picks.sort(key=lambda p: order[p.category])
    return picks, empty


# --- vẽ ----------------------------------------------------------------------------

def _wrap(text: str | None, width: int = 108) -> str:
    return "\n".join(textwrap.wrap(text or "", width=width))


def plot_pick(pick: Pick, run: RunResult, ann: Annotation, threshold: float,
              out_path: Path, dpi: int = 150) -> Path:
    """Vẽ một hình. `matplotlib` chỉ được import ở đây để phần chọn-hình test được trên máy không có nó."""
    import matplotlib

    matplotlib.use("Agg")                    # không cần màn hình, chạy được trên CI/SSH
    import matplotlib.pyplot as plt

    fps = run.sample_fps or 1.0
    times = [i / fps for i in range(len(run.anomaly_scores))]
    segments = segments_seconds(ann)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(times, run.anomaly_scores, color="#1f77b4", linewidth=1.0, label="tier-1 score")
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0,
               label=f"ngưỡng τ = {threshold:g}")

    # Hai dải "gantt" ở trên đỉnh thay vì tô kín cả hình: tô kín thì GT (cam) và cửa sổ (xanh) chồng
    # lên nhau thành một mảng bùn, không đọc được cái nào ở đâu.
    for start, end in segments:
        ax.axvspan(start, end, ymin=0.90, ymax=0.965, facecolor="#ff7f0e", alpha=0.75,
                   edgecolor="#b34700", linewidth=0.5, label="đoạn GT")

    seen: set[str] = set()
    for index, window in enumerate(run.candidate_windows):
        verdict = _verdict(run, index)
        confirmed = bool(verdict and verdict.is_anomaly)
        key = "confirmed" if confirmed else "rejected"
        color = "#2ca02c" if confirmed else "#7f7f7f"
        is_pick = (pick.window_index == index)
        label = None if key in seen else ("cửa sổ VLM xác nhận" if confirmed else "cửa sổ VLM loại")
        seen.add(key)
        ax.axvspan(window.start_s, window.end_s, ymin=0.78 if not is_pick else 0.76,
                   ymax=0.875 if not is_pick else 0.885,
                   facecolor=color, alpha=0.45 if not is_pick else 0.85,
                   edgecolor="black" if is_pick else color,
                   linewidth=1.6 if is_pick else 0.5,
                   hatch=("" if confirmed else "///"), label=label)

    if pick.window_index is not None and pick.start_s is not None:
        for edge in (pick.start_s, pick.end_s):
            ax.axvline(edge, color="black", linewidth=0.8, linestyle=":", ymin=0, ymax=0.76)

    # Chừa chỗ phía trên cho hai dải: đỉnh trục = max(score, τ) + 25 %.
    top = max([threshold] + list(run.anomaly_scores))
    bottom = min(list(run.anomaly_scores) + [threshold])
    ax.set_ylim(bottom - 0.06 * (top - bottom or 1.0), top + 0.25 * (top - bottom or 1.0))
    ax.set_xlim(0, max(times[-1] if times else 1.0, 1.0))

    ax.set_xlabel("thời gian (giây)")
    ax.set_ylabel("điểm bất thường tier-1 (chuẩn hoá)")
    ax.set_title(f"{TITLES[pick.category]}\n{pick.video_id} · lớp {ann.label}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    lines = [f"Luật: {pick.rule}", f"Vì sao chọn: {pick.reason}"]
    if pick.caption:
        lines.append(f"VLM: {pick.vlm_type} — “{pick.caption}”")
    fig.text(0.01, 0.01, _wrap(" · ".join(lines)), fontsize=7, va="bottom", color="#333333")

    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --- CLI ---------------------------------------------------------------------------

def _load_runs(results_dir: Path) -> dict[str, RunResult]:
    """Đọc mọi `<video>.json` (bỏ `windows/`, `_index.json`); file hỏng -> lỗi rõ ràng."""
    if not results_dir.is_dir():
        raise FigureError(f"Chưa có thư mục kết quả: {results_dir}")
    runs: dict[str, RunResult] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            run = RunResult.from_json(path)
        except (ValueError, KeyError, TypeError) as e:
            raise FigureError(f"{path}: không đọc được kết quả ({e})") from e
        runs[run.video_id] = run
    if not runs:
        raise FigureError(f"{results_dir}: không có file kết quả nào")
    return runs


def _threshold_from(runs: Mapping[str, RunResult], override: float | None) -> float:
    """τ dùng để vẽ: ưu tiên --threshold, nếu không thì lấy từ metadata; các file phải cùng một τ."""
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise FigureError(f"--threshold phải hữu hạn và > 0, nhận {override}")
        return float(override)
    values = {float(r.metadata["threshold"]) for r in runs.values() if "threshold" in r.metadata}
    missing = [r.video_id for r in runs.values() if "threshold" not in r.metadata]
    if missing:
        # một phần file không có ngưỡng = kết quả ghép từ hai lần chạy; vẽ chung một τ là kể sai chuyện
        raise FigureError(f"{len(missing)} file kết quả không có metadata.threshold "
                          f"(ví dụ {missing[0]}) — truyền --threshold rõ ràng.")
    if len(values) > 1:
        raise FigureError(f"Kết quả trộn nhiều ngưỡng {sorted(values)} — truyền --threshold rõ ràng.")
    return values.pop()


def _print_selection(picks: Sequence[Pick], empty: Sequence[str]) -> None:
    print("\nChọn hình theo luật:")
    for pick in picks:
        if pick.missing:
            print(f"  [RỖNG] {pick.category}")
            continue
        where = (f"[{pick.start_s:g}-{pick.end_s:g}]" if pick.start_s is not None else "không có cửa sổ")
        extra = ""
        if pick.window_duration_s is not None:
            if pick.overlap_fraction is None:
                extra = f" dài {pick.window_duration_s:g}s (video bình thường — không có đoạn GT)"
            else:
                extra = (f" dài {pick.window_duration_s:g}s "
                         f"({pick.overlap_fraction:.0%} nằm trong GT)")
        print(f"  {pick.category:<22} {pick.video_id:<24} {where:<16}{extra}")
        print(f"       {pick.reason}")
    if empty:
        print(f"\nDanh mục RỖNG (không vẽ hình thay thế): {', '.join(empty)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C7f: 5 hình score-theo-thời-gian, chọn theo luật")
    ap.add_argument("--annotation", type=Path, required=True,
                    help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "run_A4c")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--threshold", type=float, default=None, help="τ để vẽ (mặc định: đọc metadata)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true", help="chỉ chọn và in, KHÔNG vẽ, KHÔNG ghi file")
    args = ap.parse_args(argv)

    annotations = load_annotations(args.annotation)
    runs = _load_runs(args.results_dir)
    threshold = _threshold_from(runs, args.threshold)
    picks, empty = select_cases(runs, annotations)
    print(f"kết quả: {len(runs)} video | annotation: {len(annotations)} video | τ = {threshold:g}")
    _print_selection(picks, empty)

    if args.dry_run:
        print("\n[dry-run] không vẽ, không ghi file.")
        return 0

    written: list[str] = []
    for order, pick in enumerate(picks, start=1):
        if pick.missing:
            continue
        ann = annotations[pick.video_id]
        name = f"{order:02d}_{pick.category}.png"
        plot_pick(pick, runs[pick.video_id], ann, threshold, args.out_dir / name, dpi=args.dpi)
        written.append(name)
        print(f"  đã vẽ {args.out_dir / name}")

    selection = {
        "task": "C7f-figures",
        "results_dir": display_path(args.results_dir),
        "annotation_file": display_path(args.annotation),
        "threshold": threshold,
        "rules": RULES,
        "empty_categories": empty,
        "figures": written,
        "selection": [
            {**asdict(p), "window_duration_s": p.window_duration_s,
             "overlap_fraction": p.overlap_fraction}
            for p in picks
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "selection.json"
    out.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {out} và {len(written)} hình")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

