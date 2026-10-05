# -*- coding: utf-8 -*-
"""
eval.py — Đo recall@1 / recall@3 của pipeline VLM-Anomalies-Detection (ENG-10, Nam).

Ý tưởng (như đã chốt trong Todo-Tuan-3):
  với MỖI ảnh trong data/, loại chính ảnh đó khỏi gallery (leave-one-out),
  chạy search(topk=3) → lấy top-3, rồi so với sự kiện ĐÚNG của ảnh đó trong events.json.

Ba chỉ số báo cáo:
  1. recall@1 (event-id) — top-1 đúng id sự kiện của ảnh query.
  2. recall@3 (event-id) — id sự kiện của query nằm bất kỳ đâu trong top-3.
  3. recall@1 (category) — ảnh top-1 cùng NHÓM với query (danh_nhau / duong_pho / tai_nan_xe).
     Nhóm lấy từ tiền tố tên file trước số thứ tự: danh_nhau_3.jpg -> "danh_nhau".

⚠️ Không được tự-so-với-chính-nó (lúc đó recall = 100% vô nghĩa — đã thấy trong smoke test),
   nên gallery mỗi lần chỉ còn 14 ảnh, không chứa ảnh query.

Cách chạy:
    python eval.py                          # gallery mặc định: data/ (15 ảnh)
    python eval.py --model google/siglip2-small-patch16-256   # so sánh model (ENG-11)
    python eval.py --gallery data --device cpu --out ../notes/2026-10-05-eval-topk.md

Thuộc tính đo được:
  - recall@1, recall@3 (theo id sự kiện) và recall@1 theo nhóm, đều ở dạng X/15
  - danh sách lệch: ảnh nào bị nhận nhầm thành sự kiện nào (chất liệu slide)
  - thời gian trung bình / ảnh (để đối chiếu 5 máy, ENG-07)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import search  # contract nhóm: search(image_path, gallery_dir, model_name, device, topk)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
EVENTS_FILE = ROOT / "events.json"
TOPK = 3  # hệ thống gợi ý top-3, người giám sát chọn


@dataclass(frozen=True)
class Hit:
    """Kết quả của MỘT ảnh query trong một lần chạy leave-one-out."""
    query: str            # tên ảnh query
    top: tuple            # ((tên_file, độ_khớp_%), ...) tối đa TOPK phần tử, giảm dần
    expected_id: str      # id sự kiện đúng của query ("?" nếu events.json thiếu ảnh này)
    hit1_id: bool         # top-1 đúng id sự kiện
    hit3_id: bool         # id sự kiện nằm trong top-3
    hit1_cat: bool        # top-1 cùng nhóm với query
    seconds: float        # thời gian search() cho ảnh này


@dataclass(frozen=True)
class Report:
    """Kết quả đo được của cả một lần chạy eval.py."""
    model: str
    device: str
    gallery: str
    ran_at: str
    machine: str
    max_id_sharing: int   # số ảnh nhiều nhất dùng chung một id sự kiện (1 = mỗi id một ảnh)
    hits: tuple           # tuple[Hit, ...]
    r1_id: int
    r3_id: int
    r1_cat: int
    avg_seconds: float

    @property
    def total(self) -> int:
        return len(self.hits)


def load_truth() -> dict:
    """events.json → {tên_file: id_sự_kiện}. File do Huy giữ (ENG-09)."""
    if not EVENTS_FILE.exists() or EVENTS_FILE.stat().st_size == 0:
        sys.exit(f"[eval.py] Không tìm thấy {EVENTS_FILE} — nhờ Huy đẩy lên (ENG-09).")
    with open(EVENTS_FILE, encoding="utf-8-sig") as f:
        events = json.load(f)
    return {ev["file"]: str(ev["id"]) for ev in events}


def category_of(filename: str) -> str:
    """'danh_nhau_3.jpg' -> 'danh_nhau' (phần trước số thứ tự cuối cùng)."""
    stem = Path(filename).stem
    head, sep, _ = stem.rpartition("_")
    return head if sep else stem


def leave_one_out_gallery(images: list, excluded: Path) -> Path:
    """Copy 14 ảnh còn lại vào thư mục tạm (loại ảnh query ra khỏi gallery)."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_gallery_"))
    for img in images:
        if img.name != excluded.name:
            shutil.copy2(img, tmp / img.name)
    return tmp


def evaluate(files: list, truth: dict, args) -> tuple:
    """Chạy leave-one-out từng ảnh → tuple[Hit, ...]."""
    device = search.pick_device(args.device)
    hits = []
    for img in files:
        expected = truth.get(img.name, "?")
        tmp = leave_one_out_gallery(files, img)
        t0 = time.time()
        try:
            top = search.search(str(img), gallery_dir=str(tmp),
                                model_name=args.model, device=device, topk=TOPK)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        seconds = time.time() - t0

        if not isinstance(top, list) or not top:
            sys.exit(f"[eval.py] search(topk={TOPK}) trả về {top!r} — sai contract.")
        names = [name for name, _ in top]
        hits.append(Hit(
            query=img.name,
            top=tuple(top),
            expected_id=expected,
            hit1_id=expected != "?" and truth.get(names[0], "?") == expected,
            hit3_id=expected != "?" and any(truth.get(n, "?") == expected for n in names),
            hit1_cat=category_of(names[0]) == category_of(img.name),
            seconds=seconds,
        ))
    return tuple(hits)


def build_report(hits: tuple, args, gallery: Path) -> Report:
    """Gộp kết quả từng ảnh thành số liệu tổng + thông tin lần chạy."""
    return Report(
        model=args.model,
        device=str(search.pick_device(args.device)),
        gallery=str(gallery),
        ran_at=time.strftime("%Y-%m-%d %H:%M"),
        machine=os.environ.get("COMPUTERNAME", "?"),
        max_id_sharing=max(Counter(load_truth().values()).values()),
        hits=hits,
        r1_id=sum(h.hit1_id for h in hits),
        r3_id=sum(h.hit3_id for h in hits),
        r1_cat=sum(h.hit1_cat for h in hits),
        avg_seconds=sum(h.seconds for h in hits) / len(hits),
    )


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def name_at(hit: Hit, index: int) -> str:
    """Tên ảnh ở hạng `index` (0 = top-1)."""
    return hit.top[index][0]


def print_table(report: Report) -> None:
    """In bảng ra terminal: mỗi ảnh 2 dòng (dòng kết quả + dòng top-3)."""
    print("-" * 78)
    for h in report.hits:
        print(f"{h.query:<19} → {name_at(h, 0):<19} "
              f"ID {mark(h.hit1_id)}  NHÓM {mark(h.hit1_cat)}  "
              f"{h.top[0][1]:>3}%  {h.seconds:5.1f}s")
        print(f"{'':<19}   top-{TOPK}: "
              + "   ".join(f"{rank}) {n} {s}%" for rank, (n, s) in enumerate(h.top, start=1)))
    print("-" * 78)
    print(f"  recall@1 (event-id) = {report.r1_id}/{report.total} = {report.r1_id / report.total:.3f}")
    print(f"  recall@3 (event-id) = {report.r3_id}/{report.total} = {report.r3_id / report.total:.3f}")
    print(f"  recall@1 (category) = {report.r1_cat}/{report.total} = {report.r1_cat / report.total:.3f}")
    print(f"  TB {report.avg_seconds:.1f}s/ảnh")
    print("=" * 78)


def write_notes(path: Path, report: Report) -> None:
    """Ghi báo cáo tiếng Việt (UTF-8) — mọi số liệu lấy thẳng từ lần chạy này."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = report.total
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# EVAL top-{TOPK} — {report.model}\n\n")

        f.write("## 1. Thông tin lần chạy\n\n")
        f.write(f"- Ngày chạy: **{report.ran_at}** · máy: `{report.machine}`\n")
        f.write(f"- Model: `{report.model}`\n")
        f.write(f"- Thiết bị: `{report.device}`\n")
        f.write(f"- Gallery: `{report.gallery}` — {n} ảnh, mỗi lần đo còn {n - 1} ảnh (đã bỏ ảnh query)\n\n")

        f.write("## 2. Cách đo\n\n")
        f.write("**Leave-one-out.** Với mỗi ảnh trong `data/`, tạo một thư mục gallery tạm chỉ chứa "
                f"**{n - 1} ảnh còn lại** — loại chính ảnh query ra khỏi gallery — rồi chạy "
                f"`search(..., topk={TOPK})` để lấy {TOPK} kết quả giống nhất.\n\n")
        f.write("**Tại sao không self-match (không tự-so-với-chính-nó):** nếu để ảnh query nằm trong gallery "
                "thì cosine similarity của nó với chính nó = 1,0 (100%), nó luôn đứng top-1 và recall ra "
                "15/15 = 100% — hoàn toàn vô nghĩa về mặt đánh giá. Vòng smoke test đầu tiên đã cho ra đúng "
                "kết quả 100% giả đó. Chỉ loại ảnh query khỏi gallery thì mới đo được khả năng hệ thống thật "
                "sự khớp được một ảnh mới với ảnh cũ.\n\n")
        f.write(f"**Ba chỉ số.** `recall@1 (event-id)`: top-1 có đúng id sự kiện của query. "
                f"`recall@3 (event-id)`: id sự kiện của query nằm bất kỳ đâu trong top-{TOPK}. "
                f"`recall@1 (category)`: ảnh top-1 cùng **nhóm** với query.\n\n")
        f.write("**Quy tắc nhóm (category):** nhóm lấy từ tiền tố tên file trước số thứ tự cuối cùng — "
                "`danh_nhau_3.jpg` → `danh_nhau`, `duong_pho_1.jpg` → `duong_pho`, "
                "`tai_nan_xe_5.jpg` → `tai_nan_xe`. Ba nhóm × 5 ảnh = 15 ảnh.\n\n")

        f.write(f"## 3. Kết quả từng ảnh (top-{TOPK})\n\n")
        f.write(f"| Query | Top-1 | Top-{TOPK} | Đúng id? | Đúng nhóm? | Score top-1 | Giây |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for h in report.hits:
            cells = " · ".join(f"{n_} ({s}%)" for n_, s in h.top)
            f.write(f"| {h.query} | {name_at(h, 0)} | {cells} | {mark(h.hit1_id)} | "
                    f"{mark(h.hit1_cat)} | {h.top[0][1]}% | {h.seconds:.1f} |\n")

        f.write("\n## 4. Tổng kết\n\n")
        f.write("| Chỉ số | Kết quả | Tỉ lệ |\n|---|---|---|\n")
        f.write(f"| recall@1 (event-id) | **{report.r1_id}/{n}** | {report.r1_id / n:.3f} |\n")
        f.write(f"| recall@3 (event-id) | **{report.r3_id}/{n}** | {report.r3_id / n:.3f} |\n")
        f.write(f"| recall@1 (category) | **{report.r1_cat}/{n}** | {report.r1_cat / n:.3f} |\n\n")
        total_s = sum(h.seconds for h in report.hits)
        f.write(f"- Thời gian trung bình: **{report.avg_seconds:.1f}s/ảnh** trên `{report.device}` "
                f"(tổng {total_s:.0f}s cho {n} ảnh).\n\n")

        f.write("## 5. Đọc kết quả\n\n")
        f.write(f"- `events.json`: mỗi id sự kiện dùng cho nhiều nhất **{report.max_id_sharing} ảnh**. "
                + ("Ảnh query đã bị loại khỏi gallery nên không ảnh nào trong gallery mang được id của query — "
                   "recall@1 và recall@3 theo event-id bằng 0 **theo cấu trúc dữ liệu**, không phải vì model yếu: "
                   "hai chỉ số này đo khả năng phân biệt các sự kiện *khác nhau*, mà 15 ảnh này mỗi ảnh là một "
                   "sự kiện khác nhau.\n" if report.max_id_sharing <= 1 else
                   "Vì vậy recall@3 theo event-id mới thật sự phản ánh chất lượng model.\n"))
        f.write(f"- Vì vậy con số **đáng dùng cho demo** là `recall@1 (category)` = **{report.r1_cat}/{n}**: "
                "hệ thống không tách được 15 sự kiện riêng biệt, nhưng vẫn giữ đúng nhóm tình huống ở ảnh đầu "
                f"tiên — đúng mức để demo câu chuyện \"gợi ý top-{TOPK}, người giám sát chọn\".\n")
        wrong_cat = ", ".join("`" + h.query + "`" for h in report.hits if not h.hit1_cat)
        f.write("- Ảnh lọt sang nhóm khác ở top-1: "
                + (wrong_cat + ".\n" if wrong_cat else "không có ảnh nào.\n"))
        f.write(f"- {n - report.r1_id}/{n} ảnh không khớp id ở top-1 — xem bảng mục 3 để biết ảnh nào bị "
                "nhận nhầm sang ảnh nào.\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=f"EVAL leave-one-out: recall@1 / recall@{TOPK} của retrieval 15 ảnh (ENG-10)."
    )
    p.add_argument("--gallery", default=search.DEFAULT_GALLERY,
                   help=f"thư mục ảnh gallery (mặc định: {search.DEFAULT_GALLERY})")
    p.add_argument("--model", default=search.DEFAULT_MODEL,
                   help="model HF (default: SigLIP 2). So sánh model thì đổi bản small (ENG-11).")
    p.add_argument("--device", default=None, help="cpu / cuda ... (mặc định tự chọn).")
    p.add_argument("--out", default=None,
                   help="ghi báo cáo markdown ra file (vd ../notes/2026-10-05-eval-topk.md)")
    args = p.parse_args(argv)

    gallery = Path(args.gallery)
    if not gallery.is_dir():
        sys.exit(f"[eval.py] Không tìm thấy gallery '{gallery}'.")
    files = search.list_images(gallery)
    if not files:
        sys.exit(f"[eval.py] Gallery '{gallery}' không có ảnh nào.")

    truth = load_truth()
    missing = [f.name for f in files if f.name not in truth]
    if missing:
        print(f"[eval.py] ⚠️ {len(missing)} ảnh không có trong events.json: {missing}")
        print("           Sự kiện của các ảnh này sẽ tính là SAI (cho Huy biết, ENG-09).")

    device = search.pick_device(args.device)
    print("=" * 78)
    print(f"  EVAL top-{TOPK} — leave-one-out · model: {args.model}")
    print(f"  Gallery: {gallery} ({len(files)} ảnh) · thiết bị: {device}")
    print("=" * 78)

    hits = evaluate(files, truth, args)
    report = build_report(hits, args, gallery)
    print_table(report)

    wrong_id = [h for h in hits if not h.hit1_id]
    if wrong_id:
        print("\n  Chi tiết ảnh bị nhận nhầm (chất liệu cho slide):")
        for h in wrong_id:
            print(f"    - {h.query} (đúng sự kiện #{h.expected_id}) → top-1 {name_at(h, 0)} ({h.top[0][1]}%)")
    else:
        print("\n  Không có ảnh nào bị nhận nhầm.")

    if args.out:
        out = Path(args.out)
        write_notes(out, report)
        print(f"\n  Đã ghi báo cáo: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())