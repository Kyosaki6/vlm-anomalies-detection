# -*- coding: utf-8 -*-
"""
search.py — Tìm ảnh giống nhất bằng SigLIP 2 (Hugging Face).

Contract nhóm (README): ``search(image_path)`` -> ``(file, score)``
tên ảnh khớp nhất + độ khớp 0..100. Ví dụ::

    from search import search

    file, score = search("anh_moi.jpg")
    print(f"Ảnh này giống ảnh {file} nhất, độ khớp {score}%.")
    # Ảnh này giống ảnh danh_nhau_1.jpg nhất, độ khớp 89%.

Gọi kèm tùy chọn (vẫn tương thích contract)::

    search("anh_moi.jpg", gallery_dir="data")   # thư mục ảnh gốc
    search("anh_moi.jpg", topk=3)               # -> [(file, score), ...]

Chạy trên terminal::

    pip install -r requirements.txt
    python search.py anh_moi.jpg                       # gallery mặc định: data/
    python search.py anh_moi.jpg --topk 3
    python search.py --query anh_moi.jpg --gallery data --model google/siglip2-base-patch16-384

Thư mục gallery mặc định là ``data/`` (15 ảnh của bạn Huy).
Model mặc định là SigLIP 2 theo chốt của nhóm; máy yếu đổi sang
``google/siglip2-small-patch16-256`` (xem README).
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

__all__ = ["search", "load_model", "format_result", "main", "DEFAULT_MODEL", "DEFAULT_GALLERY"]

DEFAULT_MODEL = "google/siglip2-base-patch16-384"  # SigLIP 2 (nhóm chốt). Máy yếu: google/siglip2-small-patch16-256
DEFAULT_GALLERY = "data"  # 15 ảnh của bạn Huy
GALLERY_FALLBACK = "gallery_huy"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cache model theo (model_name, device) để gọi search() nhiều lần
# trong cùng 1 process (ví dụ main.py) không phải tải lại model.
_MODEL_CACHE: dict = {}


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_device(device):
    if device is None:
        return pick_device(None)
    return device if isinstance(device, torch.device) else torch.device(device)


def resolve_gallery(gallery_dir=None) -> Path:
    """Trả về Path thư mục gallery. Raise FileNotFoundError nếu không có."""
    if gallery_dir:
        gallery = Path(gallery_dir)
    elif Path(DEFAULT_GALLERY).is_dir():
        gallery = Path(DEFAULT_GALLERY)
    else:
        gallery = Path(GALLERY_FALLBACK)
    if not gallery.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục gallery: {gallery}. "
            f"Thư mục mặc định '{DEFAULT_GALLERY}' chứa 15 ảnh của bạn Huy."
        )
    return gallery


def list_images(folder: Path):
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files


def load_model(model_name=None, device=None):
    """Tải (có cache) processor + model SigLIP 2. Trả về (processor, model, device)."""
    model_name = model_name or DEFAULT_MODEL
    device = _resolve_device(device)
    key = (model_name, str(device))
    if key not in _MODEL_CACHE:
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            raise RuntimeError(
                f"Không tải được model '{model_name}': {e}. Gợi ý: kiểm tra mạng / tên model. "
                f"SigLIP 2: google/siglip2-base-patch16-384 (máy yếu: google/siglip2-small-patch16-256)."
            ) from e
        model.eval()
        _MODEL_CACHE[key] = (processor, model, device)
    return _MODEL_CACHE[key]


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


def _rank(query_path, gallery_files, processor, model, device, topk=1):
    """Chấm cosine query vs gallery, trả về [(tên_file, độ_khớp_0_100), ...] giảm dần."""
    gallery_embs = embed_images(gallery_files, processor, model, device)  # (N, D)
    query_emb = embed_images([query_path], processor, model, device)      # (1, D)
    sims = (query_emb @ gallery_embs.T).squeeze(0)  # cosine vì đã normalize, range [-1, 1]
    k = max(1, min(topk, len(gallery_files)))
    top_vals, top_idx = torch.topk(sims, k=k)
    results = []
    for rank in range(k):
        name = gallery_files[int(top_idx[rank])].name
        pct = int(round(float(torch.clamp(top_vals[rank], 0.0, 1.0)) * 100))
        results.append((name, pct))
    return results


def search(image_path, gallery_dir=None, model_name=None, device=None, topk=1):
    """Tìm ảnh giống nhất — đúng contract nhóm.

    Args:
        image_path: đường dẫn ảnh mới cần tra cứu.
        gallery_dir: thư mục ảnh gốc (mặc định ``data/``).
        model_name: model Hugging Face (mặc định SigLIP 2).
        device: ``cpu`` / ``cuda`` ... (mặc định tự chọn).
        topk: số kết quả muốn lấy. ``topk=1`` (mặc định) trả về
            tuple ``(file, score)`` đúng contract; ``topk>1`` trả về
            list ``[(file, score), ...]`` giảm dần.

    Returns:
        ``(file, score)`` — tên ảnh khớp nhất + độ khớp 0..100.

    Raises:
        FileNotFoundError: ảnh query / thư mục gallery không tồn tại.
        ValueError: gallery không có ảnh nào.
        RuntimeError: không tải được model.
    """
    query_path = Path(image_path)
    if not query_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh query: {query_path}")

    gallery = resolve_gallery(gallery_dir)
    gallery_files = list_images(gallery)
    if not gallery_files:
        raise ValueError(f"Thư mục gallery '{gallery}' chưa có ảnh nào (hỗ trợ: {sorted(IMG_EXTS)}).")

    processor, model, device = load_model(model_name, device)
    results = _rank(query_path, gallery_files, processor, model, device, topk=topk)
    if topk == 1:
        return results[0]
    return results


def format_result(best_file, best_score=None) -> str:
    """Dựng câu kết quả tiếng Việt. Nhận tuple (file, score) hoặc 2 tham số rời."""
    if best_score is None:
        best_file, best_score = best_file
    return f"Ảnh này giống ảnh {best_file} nhất, độ khớp {best_score}%."


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="So 1 ảnh mới với thư viện ảnh (mặc định 15 ảnh của Huy) bằng SigLIP 2."
    )
    p.add_argument("query_pos", nargs="?", default=None, help="Đường dẫn ảnh mới cần tra cứu.")
    p.add_argument("--query", "-q", default=None, help="Đường dẫn ảnh mới cần tra cứu (dạng flag).")
    p.add_argument(
        "--gallery", "-g", default=None,
        help=f"Thư mục chứa ảnh gốc để so sánh (mặc định: {DEFAULT_GALLERY}).",
    )
    p.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=(
            "Model Hugging Face. Mặc định SigLIP 2: google/siglip2-base-patch16-384 | "
            "máy yếu: google/siglip2-small-patch16-256"
        ),
    )
    p.add_argument("--topk", "-k", type=int, default=1, help="Số kết quả giống nhất muốn xem (mặc định: 1).")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (mặc định tự chọn).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    query_path = args.query or args.query_pos
    if not query_path:
        print("Thiếu ảnh đầu vào. Ví dụ: python search.py anh_moi.jpg", file=sys.stderr)
        return 2

    try:
        results = search(
            query_path,
            gallery_dir=args.gallery,
            model_name=args.model,
            device=args.device,
            topk=args.topk,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 2

    if args.topk == 1:
        # Dòng kết quả chính — đúng format yêu cầu:
        print(format_result(results))
    else:
        best_file, best_pct = results[0]
        print(format_result(best_file, best_pct))
        for rank, (name, pct) in enumerate(results[1:], start=2):
            print(f"Top {rank}: {name} — độ khớp {pct}%.")

    return 0


if __name__ == "__main__":
    # Bắt buộc UTF-8 khi in tiếng Việt trên terminal Windows
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main())
