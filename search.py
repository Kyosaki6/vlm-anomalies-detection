# -*- coding: utf-8 -*-
"""
search.py — Tìm ảnh giống nhất bằng SigLIP / SigLIP 2 (Hugging Face).

Cách dùng trên terminal:
    pip install -r requirements.txt

    # Cách 1 (khuyên dùng): ảnh query là tham số vị trí
    python search.py anh_moi.jpg --gallery gallery_huy

    # Cách 2: dùng flag --query
    python search.py --query anh_moi.jpg --gallery gallery_huy --model google/siglip-base-patch16-224

    # Đổi sang SigLIP 2:
    python search.py anh_moi.jpg --model google/siglip2-base-patch16-224

    # Xem thêm top-k kết quả:
    python search.py anh_moi.jpg --topk 3

Kết quả in ra đúng 1 dòng chính, ví dụ:
    Ảnh này giống ảnh danh_nhau_1.jpg nhất, độ khớp 89%.

Thư mục gallery mặc định là `gallery_huy/` (chứa ~15 ảnh của bạn Huy).
Bỏ 15 ảnh vào đó rồi chạy lệnh trên. Có thể trỏ sang thư mục khác bằng --gallery.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
except ImportError as e:
    print(f"Thiếu thư viện: {e}. Hãy chạy: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)

DEFAULT_MODEL = "google/siglip-base-patch16-224"  # SigLIP. Muốn SigLIP 2: google/siglip2-base-patch16-224
DEFAULT_GALLERY = "gallery_huy"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="So 1 ảnh mới với thư viện ảnh (mặc định 15 ảnh của Huy) bằng SigLIP/SigLIP2."
    )
    p.add_argument("query_pos", nargs="?", default=None, help="Đường dẫn ảnh mới cần tra cứu.")
    p.add_argument("--query", "-q", default=None, help="Đường dẫn ảnh mới cần tra cứu (dạng flag).")
    p.add_argument(
        "--gallery", "-g", default=DEFAULT_GALLERY,
        help=f"Thư mục chứa ảnh gốc để so sánh (mặc định: {DEFAULT_GALLERY}).",
    )
    p.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=(
            "Model Hugging Face. Ví dụ SigLIP: google/siglip-base-patch16-224 | "
            "SigLIP 2: google/siglip2-base-patch16-224"
        ),
    )
    p.add_argument("--topk", "-k", type=int, default=1, help="Số kết quả giống nhất muốn xem (mặc định: 1).")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (mặc định tự chọn).")
    return p.parse_args(argv)


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder: Path):
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files


@torch.no_grad()
def embed_images(paths, processor, model, device, batch_size=8):
    """Trả về tensor (N, D) đã L2-normalize."""
    feats = []
    model.eval()
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = model.get_image_features(pixel_values=pixel_values)
        # AutoModel có thể trả về tensor hoặc object (BaseModelOutputWithPooling)
        if not isinstance(out, torch.Tensor):
            out = getattr(out, "image_embeds", getattr(out, "pooler_output", out))
        out = F.normalize(out, p=2, dim=-1)
        feats.append(out.cpu())
    if not feats:
        return torch.empty(0, 0)
    return torch.cat(feats, dim=0)


def main(argv=None):
    args = parse_args(argv)

    query_path = args.query or args.query_pos
    if not query_path:
        print("Thiếu ảnh đầu vào. Ví dụ: python search.py anh_moi.jpg --gallery gallery_huy", file=sys.stderr)
        return 2
    query_path = Path(query_path)
    if not query_path.is_file():
        print(f"Không tìm thấy ảnh query: {query_path}", file=sys.stderr)
        return 2

    gallery_dir = Path(args.gallery)
    if not gallery_dir.is_dir():
        print(
            f"Không tìm thấy thư mục gallery: {gallery_dir}\n"
            f"Hãy tạo thư mục '{gallery_dir}' và bỏ ~15 ảnh của bạn Huy vào đó, "
            f"rồi chạy lại. Ví dụ: python search.py {query_path} --gallery {gallery_dir}",
            file=sys.stderr,
        )
        return 2

    gallery_files = list_images(gallery_dir)
    if not gallery_files:
        print(f"Thư mục gallery '{gallery_dir}' chưa có ảnh nào (hỗ trợ: {sorted(IMG_EXTS)}).", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(device)
    except Exception as e:
        print(f"Không tải được model '{args.model}': {e}", file=sys.stderr)
        print("Gợi ý: kiểm tra mạng / tên model. SigLIP: google/siglip-base-patch16-224, "
              "SigLIP 2: google/siglip2-base-patch16-224", file=sys.stderr)
        return 2

    gallery_embs = embed_images(gallery_files, processor, model, device)  # (N, D)
    query_emb = embed_images([query_path], processor, model, device)      # (1, D)

    sims = (query_emb @ gallery_embs.T).squeeze(0)  # cosine vì đã normalize, range [-1, 1]
    k = max(1, min(args.topk, len(gallery_files)))
    top_vals, top_idx = torch.topk(sims, k=k)

    best_file = gallery_files[int(top_idx[0])].name
    best_pct = int(round(float(torch.clamp(top_vals[0], 0.0, 1.0)) * 100))

    # Dòng kết quả chính — đúng format yêu cầu:
    print(f"Ảnh này giống ảnh {best_file} nhất, độ khớp {best_pct}%.")

    if k > 1:
        for rank in range(1, k):
            name = gallery_files[int(top_idx[rank])].name
            pct = int(round(float(torch.clamp(top_vals[rank], 0.0, 1.0)) * 100))
            print(f"Top {rank + 1}: {name} — độ khớp {pct}%.")

    return 0


if __name__ == "__main__":
    # Bắt buộc UTF-8 khi in tiếng Việt trên terminal Windows
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main())
