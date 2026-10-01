# -*- coding: utf-8 -*-
"""
eval.py — Đo recall@1 của pipeline VLM-Anomalies-Detection (ENG-10, Nam).

Ý tưởng (như đã chốt trong Todo-Tuan-3):
  với MỖI ảnh trong data/, loại chính ảnh đó khỏi gallery (leave-one-out),
  chạy search() → top-1, rồi so với sự kiện ĐÚNG của ảnh đó trong events.json.

  recall@1 = số ảnh có top-1 ĐÚNG / tổng số ảnh. ⚠️ Không được tự-so-với-chính-nó
  (lúc đó recall = 100% vô nghĩa — đã thấy trong smoke test).

Cách chạy:
    python eval.py                          # gallery mặc định: data/ (15 ảnh)
    python eval.py --model google/siglip2-small-patch16-256   # so sánh model (ENG-11)
    python eval.py --gallery data --device cpu --out notes/... (ghi kết quả ra file)

Thuộc tính đo được:
  - recall@1 (0..1) và số đúng X/15
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
from pathlib import Path

import search  # contract nhóm: search(image_path, gallery_dir, model_name, device, topk)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
EVENTS_FILE = ROOT / "events.json"


def load_truth() -> dict:
    """events.json → {tên_file: id_sự_kiện}. File do Huy giữ (ENG-09)."""
    if not EVENTS_FILE.exists() or EVENTS_FILE.stat().st_size == 0:
        sys.exit(f"[eval.py] Không tìm thấy {EVENTS_FILE} — nhờ Huy đẩy lên (ENG-09).")
    with open(EVENTS_FILE, encoding="utf-8-sig") as f:
        events = json.load(f)
    return {ev["file"]: str(ev["id"]) for ev in events}


def leave_one_out_gallery(images: list, excluded: Path) -> Path:
    """Copy 14 ảnh còn lại vào thư mục tạm (loại ảnh query ra khỏi gallery)."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_gallery_"))
    for img in images:
        if img.name != excluded.name:
            shutil.copy2(img, tmp / img.name)
    return tmp


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="EVAL leave-one-out: recall@1 của retrieval 15 ảnh (ENG-10)."
    )
    p.add_argument("--gallery", default=search.DEFAULT_GALLERY,
                   help=f"thư mục ảnh gallery (mặc định: {search.DEFAULT_GALLERY})")
    p.add_argument("--model", default=search.DEFAULT_MODEL,
                   help="model HF (default: SigLIP 2). So sánh model thì đổi bản small (ENG-11).")
    p.add_argument("--device", default=None, help="cpu / cuda ... (mặc định tự chọn).")
    p.add_argument("--out", default=None,
                   help="ghi bảng kết quả ra file (vd notes/2026-09-29-eval.md)")
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

    print("=" * 78)
    print(f"  EVAL recall@1 — leave-one-out · model: {args.model}")
    print(f"  Gallery: {gallery} ({len(files)} ảnh) · thiết bị: {args.device or 'tự chọn'}")
    print("=" * 78)
    header = f"{'ẢNH (query)':<20} {'→ top-1':<20} {'Đúng?':<5} {'Score':>6}   t(s)"
    print(header)
    print("-" * 78)

    rows, elapsed, correct, total = [], 0.0, 0, 0
    for img in files:
        if img.name not in truth:  # ảnh không có nhãn → tính SAI, vẫn chạy
            expected = "?"
        else:
            expected = truth[img.name]

        tmp = leave_one_out_gallery(files, img)
        t0 = time.time()
        try:
            best_file, score = search.search(str(img), gallery_dir=str(tmp),
                                             model_name=args.model, device=args.device, topk=1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        dt = time.time() - t0
        elapsed += dt

        got = truth.get(best_file, "?")
        ok = expected != "?" and got == expected
        correct += 1 if ok else 0
        total += 1
        mark = "✓" if ok else "✗"
        rows.append((img.name, best_file, mark, score, dt, expected, got))
        print(f"{img.name:<20} {best_file:<20} {mark:<5} {score:>5}%  {dt:5.1f}")

    recall = correct / total if total else 0.0
    avg = elapsed / total if total else 0.0
    print("-" * 78)
    print(f"  recall@1 = {correct}/{total} = {recall:.3f}   · TB {avg:.1f}s/ảnh")
    print("=" * 78)

    wrongs = [r for r in rows if r[2] == "✗"]
    if wrongs:
        print("\n  Chi tiết ảnh bị nhận nhầm (chất liệu cho slide):")
        for name, best, mark, score, dt, exp, got in wrongs:
            print(f"    - {name} (đúng sự kiện #{exp}) → nhận nhầm thành {best} (#{got}), {score}%")
    else:
        print("\n  Không có ảnh nào bị nhận nhầm.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# EVAL recall@1 — {args.model}\n\n")
            f.write(f"- Ngày: {time.strftime('%Y-%m-%d %H:%M')} · máy: {os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', '?')}\n")
            f.write(f"- Cách đo: leave-one-out (loại ảnh query khỏi gallery) — KHÔNG tự-so-với-chính-nó\n")
            f.write(f"- Thiết bị: {args.device or 'tự chọn'} · gallery: {gallery.name}\n\n")
            f.write("| Ảnh (query) | → top-1 | Đúng? | Score | Thời gian (s) |\n|---|---|---|---|---|\n")
            for name, best, mark, score, dt, exp, got in rows:
                f.write(f"| {name} | {best} | {mark} | {score}% | {dt:.1f} |\n")
            f.write(f"\n**recall@1 = {correct}/{total} = {recall:.3f}** · TB {avg:.1f}s/ảnh\n")
        print(f"\n  Đã ghi bảng kết quả: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())