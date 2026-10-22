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
    raise ImportError(
        f"Thiếu thư viện: {e}. Hãy chạy: pip install -r requirements.txt"
    ) from e

__all__ = [
    "search", "anomaly_score", "load_model", "gallery_embeddings",
    "format_result", "main", "DEFAULT_MODEL", "DEFAULT_GALLERY",
    "embed_texts", "text_anomaly_score", "text_abnormal_probability", "fused_anomaly_score", "combine_scores",
    "NORMAL_PROMPT", "ABNORMAL_PROMPTS",
]

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "google/siglip2-base-patch16-384"  # SigLIP 2 (nhóm chốt). Máy yếu: google/siglip2-small-patch16-256
DEFAULT_GALLERY = str(ROOT / "data")  # 15 ảnh của bạn Huy
GALLERY_FALLBACK = str(ROOT / "gallery_huy")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cache model theo (model_name, device) để gọi search() nhiều lần
# trong cùng 1 process (ví dụ main.py) không phải tải lại model.
_MODEL_CACHE: dict = {}

# Cache embedding gallery theo (model, device, danh sách file + mtime + size).
# Nhờ vậy gọi search()/anomaly_score() nhiều lần trong cùng process không phải
# encode lại 15 ảnh — đây là tối ưu ENG-06 (trước 8.4x khi so 15 lần gọi).
_GALLERY_CACHE: dict = {}


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
        if not gallery.is_dir() and not gallery.is_absolute():
            candidate = ROOT / gallery
            if candidate.is_dir():
                gallery = candidate
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


def _files_fingerprint(files):
    """Chữ ký danh sách file: tên + mtime + size (để cache tự vô hiệu khi ảnh đổi)."""
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)


def gallery_embeddings(gallery_files, processor, model, device, model_name=None):
    """Embedding gallery, có cache trong RAM theo (model, device, file fingerprint).

    Dùng chung cho ``_rank()`` và ``anomaly_score()``.
    """
    model_name = model_name or DEFAULT_MODEL
    # Có thư mục trong key để hai gallery tạm khác nhau (cùng tên file, cùng mtime
    # do copy2) không dùng nhầm cache của nhau.
    parent = str(gallery_files[0].parent.resolve()) if gallery_files else ""
    key = (model_name, str(device), parent, _files_fingerprint(gallery_files))
    if key not in _GALLERY_CACHE:
        _GALLERY_CACHE[key] = embed_images(gallery_files, processor, model, device)
    return _GALLERY_CACHE[key]


def _rank(query_path, gallery_files, processor, model, device, topk=1, model_name=None):
    """Chấm cosine query vs gallery, trả về [(tên_file, độ_khớp_0_100), ...] giảm dần."""
    gallery_embs = gallery_embeddings(gallery_files, processor, model, device, model_name)
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
    results = _rank(query_path, gallery_files, processor, model, device,
                    topk=topk, model_name=model_name)
    if topk == 1:
        return results[0]
    return results


def anomaly_score(image_path, gallery_dir=None, normal_prefix="duong_pho",
                  model_name=None, device=None):
    """Điểm bất thường = ``1 - max cosine similarity`` tới gallery ảnh BÌNH THƯỜNG.

    Vì sao không dùng "độ khớp với ảnh giống nhất"? Vì gallery có cả ảnh bất
    thường, nên một vụ tai nạn sẽ "khớp" rất cao với vụ tai nạn khác. Muốn biết
    ảnh có bất thường không, phải so với **ảnh bình thường** làm mốc.

    Nếu chính ảnh query nằm trong gallery bình thường (ví dụ khi chạy eval
    leave-one-out), ảnh đó được loại khỏi gallery để không tự so với chính nó.

    Returns:
        ``(max_similarity, anomaly_score)`` — cả hai trong khoảng 0..1.
    """
    query_path = Path(image_path)
    if not query_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh query: {query_path}")

    gallery = resolve_gallery(gallery_dir)
    normal_files = [p for p in list_images(gallery) if p.name.startswith(normal_prefix)]
    # Leave-one-out: không cho ảnh query tự so với chính nó.
    query_resolved = query_path.resolve()
    normal_files = [p for p in normal_files if p.resolve() != query_resolved]
    if not normal_files:
        raise ValueError(
            f"Gallery '{gallery}' không có ảnh bình thường nào (tiền tố '{normal_prefix}')."
        )

    processor, model, device = load_model(model_name, device)
    normal_embs = gallery_embeddings(normal_files, processor, model, device, model_name)
    query_emb = embed_images([query_path], processor, model, device)
    max_sim = float((query_emb @ normal_embs.T).max().item())
    max_sim = max(0.0, min(1.0, max_sim))
    return max_sim, 1.0 - max_sim


# ---------------------------------------------------------------------------
# Nhánh văn bản (zero-shot) — dùng text encoder của SigLIP 2.
# Ảnh và prompt cùng nằm trong một không gian embedding nên so cosine trực tiếp.
# ---------------------------------------------------------------------------
NORMAL_PROMPT = "A photo of normal street traffic and daily activities."
ABNORMAL_PROMPTS = [
    "A photo of people fighting or physical violence.",
    "A photo of a car accident or vehicle collision on the road.",
]
TEXT_MAX_LENGTH = 64  # SigLIP 2 được huấn luyện với padding="max_length", 64 token
_DEFAULT_LOGIT_SCALE = 100.0  # dùng khi model không có logit_scale
_TEXT_CACHE: dict = {}


@torch.no_grad()
def embed_texts(texts, processor, model, device) -> "torch.Tensor":
    """Trả về tensor (N, D) embedding văn bản đã L2-normalize."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if not texts:
        return torch.empty(0, 0)
    model.eval()
    inputs = processor(text=texts, padding="max_length", max_length=TEXT_MAX_LENGTH,
                       truncation=True, return_tensors="pt")
    out = model.get_text_features(input_ids=inputs["input_ids"].to(device))
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "text_embeds", getattr(out, "pooler_output", out))
    return F.normalize(out, p=2, dim=-1).cpu()


def _prompt_embeddings(processor, model, device, model_name, normal_prompt, abnormal_prompts):
    """Embedding (có cache) của 1 prompt bình thường + các prompt bất thường."""
    abnormal_prompts = list(abnormal_prompts)
    key = (model_name or DEFAULT_MODEL, str(device), normal_prompt, tuple(abnormal_prompts))
    if key not in _TEXT_CACHE:
        embs = embed_texts([normal_prompt] + abnormal_prompts, processor, model, device)
        _TEXT_CACHE[key] = (embs[:1], embs[1:])
    return _TEXT_CACHE[key]


def _logit_scale(model) -> float:
    scale = getattr(model, "logit_scale", None)
    try:
        return float(scale.exp().item()) if scale is not None else _DEFAULT_LOGIT_SCALE
    except Exception:
        return _DEFAULT_LOGIT_SCALE


def text_anomaly_score(image_path_or_emb, processor=None, model=None, device=None,
                       model_name=None, normal_prompt=NORMAL_PROMPT,
                       abnormal_prompts=ABNORMAL_PROMPTS):
    """Chấm zero-shot: cosine của ảnh với prompt BÌNH THƯỜNG vs prompt BẤT THƯỜNG.

    ``image_path_or_emb`` là đường dẫn ảnh hoặc embedding ảnh đã L2-normalize ``(D,)``/``(1, D)``.
    Nếu truyền path, file được kiểm tra TRƯỚC khi tải model.

    Returns:
        ``(normal_sim, abnormal_sim)`` — cosine tới prompt bình thường, và cosine
        lớn nhất tới các prompt bất thường. Điểm càng nghiêng về ``abnormal_sim``
        thì ảnh càng "giống mô tả bất thường" (xem ``text_abnormal_probability``).
    """
    if isinstance(image_path_or_emb, torch.Tensor):
        image_emb = image_path_or_emb.reshape(1, -1).cpu()
    else:
        path = Path(image_path_or_emb)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh query: {path}")
        if processor is None or model is None:
            processor, model, device = load_model(model_name, device)
        image_emb = embed_images([path], processor, model, device)
    if processor is None or model is None:
        processor, model, device = load_model(model_name, device)
    device = _resolve_device(device)
    normal_emb, abnormal_embs = _prompt_embeddings(
        processor, model, device, model_name, normal_prompt, abnormal_prompts)
    normal_sim = float((image_emb @ normal_emb.T).max().item())
    abnormal_sim = float((image_emb @ abnormal_embs.T).max().item())
    return normal_sim, abnormal_sim


def text_abnormal_probability(normal_sim: float, abnormal_sim: float, scale: float = _DEFAULT_LOGIT_SCALE) -> float:
    """softmax 2 lớp trên cosine (nhân ``scale``) -> xác suất 'bất thường' trong 0..1."""
    import math
    a = (abnormal_sim - normal_sim) * scale
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    e = math.exp(a)
    return e / (1.0 + e)


def combine_scores(image_score: float, text_score: float, alpha: float = 0.5,
                   threshold: float = 0.30) -> float:
    """``alpha * min(1, image_score / (2 * threshold)) + (1 - alpha) * text_score``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha phải trong [0, 1], nhận {alpha}.")
    image_component = min(1.0, image_score / (2.0 * threshold)) if threshold > 0 else image_score
    return alpha * image_component + (1.0 - alpha) * text_score


def fused_anomaly_score(image_path, alpha=0.5, gallery_dir=None, normal_prefix="duong_pho",
                        model_name=None, device=None, threshold=0.30):
    """Kết hợp điểm ảnh (so gallery bình thường) và điểm văn bản (zero-shot).

    ``image_component = min(1, image_score / (2 * threshold))`` — đưa điểm ảnh về 0..1 sao cho
    đúng ngưỡng mặc định thì bằng 0,5, cùng thang với xác suất văn bản.
    ``fused = alpha * image_component + (1 - alpha) * text_score``.

    Returns:
        dict: ``image_score``, ``image_similarity``, ``text_normal_sim``, ``text_abnormal_sim``,
        ``text_score`` (xác suất bất thường), ``fused_score``, ``alpha``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha phải trong [0, 1], nhận {alpha}.")
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh query: {path}")
    sim, img_score = anomaly_score(path, gallery_dir=gallery_dir, normal_prefix=normal_prefix,
                                   model_name=model_name, device=device)
    processor, model, dev = load_model(model_name, device)
    query_emb = embed_images([path], processor, model, dev)
    n_sim, a_sim = text_anomaly_score(query_emb, processor, model, dev, model_name=model_name)
    text_score = text_abnormal_probability(n_sim, a_sim, _logit_scale(model))
    return {
        "image_score": img_score,
        "image_similarity": sim,
        "text_normal_sim": n_sim,
        "text_abnormal_sim": a_sim,
        "text_score": text_score,
        "fused_score": combine_scores(img_score, text_score, alpha, threshold),
        "alpha": alpha,
    }


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
