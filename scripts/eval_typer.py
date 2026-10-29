# -*- coding: utf-8 -*-
"""C3 — đo độ chính xác của bộ đoán loại sự kiện (13 lớp) trên đoạn bất thường có nhãn thật.

    python scripts/eval_typer.py --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    python scripts/eval_typer.py --annotation ... --dry-run      # xem sẽ chấm bao nhiêu ca, KHÔNG cần cache/model

CÂU HỎI DUY NHẤT file này trả lời: "khi biết chắc đoạn này có bất thường, hệ thống gọi đúng TÊN
loại sự kiện bao nhiêu lần?" — KHÁC với detection AUC ("có bất thường hay không"). Báo cáo phải để
hai con số này riêng, đừng cộng gộp.

Cách chấm (một ca = một ĐOẠN bất thường, không phải một video):
  1. Annotation cho đoạn ở frame nguồn 30 fps: [start, end].
  2. Đổi sang chỉ số khung đã lấy mẫu theo `sample_fps` (xem `segment_sample_range`).
  3. Lấy trung bình embedding các khung đó (khớp `aggregate='mean'` của typer).
  4. Typer chọn 1 trong 13 lớp -> so với nhãn gốc của đoạn.

Hai con số, in cạnh nhau, KHÔNG thay thế nhau:
  - `strict`  : so khớp đúng nguyên tên lớp (con số chính, dùng trên slide).
  - `lenient` : gộp các cặp tên gần nghĩa của UCF (xem `LENIENT_GROUPS`) — chỉ là thông tin phụ,
                vì "Assault" bị gọi thành "Fighting" không hoàn toàn sai về mặt ngữ nghĩa, nhưng cũng
                không phải đúng nhãn. Ai đọc báo cáo phải thấy cả hai.
  - `chance`  : 1/13 = 7.7% — mức đoán mò. Dưới mức này thì typer vô dụng, phải nói thẳng.

Không có GPU vẫn chạy được phần tính toán: `evaluate_cases` nhận sẵn vector nên test dùng
`FakeEmbedder`, còn CLI mới cần cache thật.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.datasets.ucf_crime import Annotation, load_annotations  # noqa: E402
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached, load_embeddings  # noqa: E402
from vad.score.typing import ZeroShotEventTyper, load_typing_prompts  # noqa: E402
from vad.types import EventType  # noqa: E402

SOURCE_FPS = 30.0          # frame trong annotation UCF-Crime là 30 fps
DEFAULT_MODEL = "google/siglip2-base-patch16-384"

# Nhóm gần nghĩa: dùng CHỈ để báo cáo phụ. Cố tình để nhỏ và giải thích được.
# UCF-Crime coi đây là các lớp riêng, nên `strict` mới là con số chính.
LENIENT_GROUPS: dict[EventType, frozenset[EventType]] = {
    EventType.FIGHTING: frozenset({EventType.FIGHTING, EventType.ASSAULT, EventType.ABUSE}),
    EventType.ASSAULT: frozenset({EventType.FIGHTING, EventType.ASSAULT, EventType.ABUSE}),
    EventType.ABUSE: frozenset({EventType.FIGHTING, EventType.ASSAULT, EventType.ABUSE}),
    EventType.SHOOTING: frozenset({EventType.SHOOTING, EventType.EXPLOSION}),
    EventType.EXPLOSION: frozenset({EventType.SHOOTING, EventType.EXPLOSION}),
    EventType.STEALING: frozenset({EventType.STEALING, EventType.SHOPLIFTING,
                                   EventType.BURGLARY, EventType.ROBBERY}),
    EventType.SHOPLIFTING: frozenset({EventType.STEALING, EventType.SHOPLIFTING,
                                      EventType.BURGLARY, EventType.ROBBERY}),
    EventType.BURGLARY: frozenset({EventType.STEALING, EventType.SHOPLIFTING,
                                   EventType.BURGLARY, EventType.ROBBERY}),
    EventType.ROBBERY: frozenset({EventType.STEALING, EventType.SHOPLIFTING,
                                  EventType.BURGLARY, EventType.ROBBERY}),
}


# --- hình học thời gian ------------------------------------------------------------

def segment_sample_range(start_frame: int, end_frame: int, sample_fps: float,
                         n_samples: int, source_fps: float = SOURCE_FPS) -> tuple[int, int] | None:
    """Đoạn frame nguồn [start, end] -> khoảng chỉ số khung đã lấy mẫu [i0, i1], hoặc None nếu rỗng.

    Vì sao cần hàm này: annotation đếm frame ở 30 fps, còn cache của mình là 1-2 khung/giây.
    Đoạn 100 frame = 3.3 giây; ở 1 fps thì chỉ còn 3 khung để chấm.

    Quy ước: `frame -> floor(frame * sample_fps / source_fps)`, đoạn đóng hai đầu.
    `i1` bị cắt vào trong `n_samples - 1` để không đọc quá cache (video ngắn hơn annotation).
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps phải > 0, nhận {sample_fps}")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"đoạn không hợp lệ: {start_frame}..{end_frame}")
    if n_samples <= 0:
        return None
    ratio = sample_fps / source_fps
    i0 = int(np.floor(start_frame * ratio))
    i1 = int(np.floor(end_frame * ratio))
    i0 = max(i0, 0)
    i1 = min(i1, n_samples - 1)
    if i1 < i0:
        # Đoạn quá ngắn so với bước lấy mẫu: vẫn lấy khung gần nhất, không bỏ ca (bỏ ca = thiên lệch).
        i0 = i1 = min(max(int(np.floor((start_frame + end_frame) / 2 * ratio)), 0), n_samples - 1)
    return i0, i1


# --- chấm điểm ---------------------------------------------------------------------

@dataclass(frozen=True)
class CaseResult:
    """Một đoạn bất thường đã chấm."""
    video_id: str
    gt: EventType
    pred: EventType
    confidence: float
    n_frames: int
    start_s: float
    end_s: float

    @property
    def correct(self) -> bool:
        return self.pred is self.gt

    @property
    def lenient_correct(self) -> bool:
        return self.pred in LENIENT_GROUPS.get(self.gt, frozenset({self.gt}))


def build_cases(annotations: dict[str, Annotation], sample_fps: float,
                cache_len: dict[str, int]) -> list[tuple[Annotation, tuple[int, int, int, int]]]:
    """[(annotation, (start_frame, end_frame, i0, i1))]: mọi đoạn bất thường có cache.

    Chỉ lấy video có trong `cache_len` (tức là đã embed) và chỉ đoạn bất thường — typer không được
    chấm trên video toàn bình thường vì nó không có lớp "Normal" mặc định.
    """
    out = []
    for video_id, ann in annotations.items():
        n = cache_len.get(video_id)
        if not n or not ann.segments:
            continue
        for start, end in ann.segments:
            rng = segment_sample_range(start, end, sample_fps, n)
            if rng is not None:
                out.append((ann, (start, end, rng[0], rng[1])))
    return out


def evaluate_cases(cases, typer: ZeroShotEventTyper, load_embs, aggregate: str = "mean") -> list[CaseResult]:
    """Chạy typer trên từng ca. `load_embs(video_id) -> (T, D)` được tiêm vào để test bằng FakeEmbedder."""
    results: list[CaseResult] = []
    for ann, (start, end, i0, i1) in cases:
        embs = load_embs(ann.video_id)
        if embs is None:
            continue
        window = embs[i0:i1 + 1]
        if window.size == 0:
            continue
        res = typer.classify(window, aggregate=aggregate, top_k=1)
        results.append(CaseResult(
            video_id=ann.video_id, gt=ann.event_type, pred=res.event_type,
            confidence=res.confidence, n_frames=int(window.shape[0]),
            start_s=start / SOURCE_FPS, end_s=end / SOURCE_FPS,
        ))
    return results


def summarise(results: list[CaseResult], n_classes: int) -> dict:
    """-> dict số liệu. Rỗng -> accuracy None (không bịa 0.0 hay 1.0)."""
    total = len(results)
    strict = sum(r.correct for r in results)
    lenient = sum(r.lenient_correct for r in results)

    per_class: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        c = per_class.setdefault(r.gt.value, {"total": 0, "correct": 0})
        c["total"] += 1
        c["correct"] += int(r.correct)
        confusion.setdefault(r.gt.value, {})
        confusion[r.gt.value][r.pred.value] = confusion[r.gt.value].get(r.pred.value, 0) + 1

    return {
        "n_cases": total,
        "n_videos": len({r.video_id for r in results}),
        "strict_correct": strict,
        "strict_accuracy": (strict / total) if total else None,
        "lenient_correct": lenient,
        "lenient_accuracy": (lenient / total) if total else None,
        "chance_level": 1.0 / n_classes if n_classes else None,
        "per_class": per_class,
        "confusion": confusion,
    }


# --- CLI ---------------------------------------------------------------------------

def verdict_text(s: dict, min_cases: int = 30) -> str:
    """Một câu kết luận, tính bằng CÙNG đơn vị (0..1) để tránh lỗi so sánh %/tỉ lệ.

    Ít ca thì không kết luận: 3/3 đúng không nói lên điều gì về 290 video.
    """
    n = s["n_cases"]
    if not n:
        return "không có ca nào"
    if n < min_cases:
        return f"chỉ {n} ca — quá ít để kết luận, chỉ dùng để kiểm tra đường ống"
    acc, chance = s["strict_accuracy"], s["chance_level"]
    if acc >= chance * 2:
        return "TỐT HƠN đoán mò rõ rệt"
    if acc <= chance:
        return "KHÔNG hơn đoán mò — phải nói thật trên slide"
    return "hơn đoán mò nhưng chưa gấp đôi — nói là 'có tín hiệu, còn yếu'"


def _print_summary(s: dict, results: list[CaseResult]) -> None:
    if not s["n_cases"]:
        print("Không có ca nào để chấm (thiếu cache hoặc annotation chỉ có video bình thường).")
        return
    chance = s["chance_level"] * 100
    print(f"\nSố ca: {s['n_cases']} đoạn bất thường / {s['n_videos']} video")
    print(f"  STRICT  (đúng nguyên tên lớp) : {s['strict_correct']}/{s['n_cases']}"
          f" = {s['strict_accuracy'] * 100:.1f}%")
    print(f"  lenient (gộp tên gần nghĩa)   : {s['lenient_correct']}/{s['n_cases']}"
          f" = {s['lenient_accuracy'] * 100:.1f}%")
    print(f"  đoán mò (1/{round(1 / s['chance_level'])} lớp)      : {chance:.1f}%")
    print(f"  -> {verdict_text(s)}")

    print("\nTheo lớp:")
    for name, c in sorted(s["per_class"].items()):
        acc = c["correct"] / c["total"] * 100
        print(f"  {name:<14} {c['correct']:>3}/{c['total']:<3} = {acc:5.1f}%")

    print("\nSai nhiều nhất (nhãn thật -> đoán):")
    wrong = [(r.gt.value, r.pred.value) for r in results if not r.correct]
    if not wrong:
        print("  (không sai ca nào)")
    else:
        counts: dict[tuple[str, str], int] = {}
        for pair in wrong:
            counts[pair] = counts.get(pair, 0) + 1
        for (gt, pred), n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {gt:<14} -> {pred:<14} {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C3: đo top-1 accuracy của event typer trên UCF-Crime")
    ap.add_argument("--annotation", type=Path, required=True,
                    help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=1.0, help="sample_fps của cache (phải khớp lúc embed)")
    ap.add_argument("--aggregate", choices=["mean", "max"], default="mean")
    ap.add_argument("--out", type=Path, default=None, help="ghi JSON (mặc định: results/typing_eval.json)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N video đầu (để thử nhanh)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm ca, KHÔNG nạp model")
    args = ap.parse_args(argv)

    annotations = load_annotations(args.annotation)
    cache_len: dict[str, int] = {}
    for video_id in annotations:
        if has_cached(args.model, video_id, args.fps, args.cache_dir):
            cache_len[video_id] = 0          # sẽ điền số khung khi đọc thật
    if args.limit:
        cache_len = dict(list(cache_len.items())[:args.limit])

    print(f"annotation: {len(annotations)} video | có cache ({args.model} @ {args.fps} fps): {len(cache_len)}")
    missing = len(annotations) - len(cache_len)
    if missing:
        print(f"  bỏ qua {missing} video chưa embed (chạy precompute trước)")

    if args.dry_run:
        n_segments = sum(len(annotations[v].segments) for v in cache_len)
        print(f"[dry-run] sẽ chấm {n_segments} đoạn bất thường từ {len(cache_len)} video. Không nạp model.")
        by_class: dict[str, int] = {}
        for v in cache_len:
            by_class[annotations[v].label] = by_class.get(annotations[v].label, 0) + len(annotations[v].segments)
        for name, n in sorted(by_class.items()):
            print(f"  {name:<14} {n}")
        return 0

    if not cache_len:
        print("Không có cache nào — chạy scripts/precompute_embeddings.py trước.", file=sys.stderr)
        return 1

    # Đọc thật để biết số khung từng video
    embs_cache: dict[str, np.ndarray] = {}
    for v in cache_len:
        try:
            embs_cache[v] = load_embeddings(args.model, v, args.fps, args.cache_dir)
            cache_len[v] = int(embs_cache[v].shape[0])
        except (FileNotFoundError, ValueError) as e:
            print(f"  LỖI đọc cache {v}: {e}")
    cases = build_cases(annotations, args.fps, cache_len)

    from vad.embed.siglip import SigLIPEmbedder   # chỉ import khi chạy thật

    embedder = SigLIPEmbedder(model_id=args.model, device=None)
    typer = ZeroShotEventTyper(embedder)
    results = evaluate_cases(cases, typer, lambda vid: embs_cache.get(vid), aggregate=args.aggregate)

    _, all_labels = load_typing_prompts()          # số lớp thật trong prompts.yaml (13)
    summary = summarise(results, n_classes=len(all_labels))
    _print_summary(summary, results)

    out = args.out or (Path(__file__).resolve().parents[1] / "results" / "typing_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "C3-typing",
        "model": args.model,
        "sample_fps": args.fps,
        "aggregate": args.aggregate,
        "annotation_file": str(args.annotation),
        "summary": summary,
        "cases": [
            {"video_id": r.video_id, "gt": r.gt.value, "pred": r.pred.value,
             "correct": r.correct, "lenient_correct": r.lenient_correct,
             "confidence": r.confidence, "n_frames": r.n_frames,
             "start_s": r.start_s, "end_s": r.end_s}
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nđã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
