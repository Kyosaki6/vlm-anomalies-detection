# -*- coding: utf-8 -*-
"""A1 (Phước): đo VRAM đỉnh và giây/cửa sổ của các ứng viên VLM sinh ngôn ngữ, 4-bit.

Chạy THẬT trên GPU (không nằm trong test suite; CI không gọi file này):

    pip install -r requirements-vlm.txt
    python scripts/bench_vlm.py                        # 3 ứng viên mặc định, 4 frame/cửa sổ
    python scripts/bench_vlm.py --model Qwen/Qwen2-VL-2B-Instruct --runs 5

Quyết định (task-assignment A1): peak VRAM < 5.5 GB và <= 3 s/window -> GO local 3050,
ngược lại pivot Colab T4. Số đo ghi vào notes/vlm-selection.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Ứng viên lớp 2-3B theo đề xuất của task sheet A1 + một model nhỏ để so sánh.
# transformers 5.x: class riêng từng gia đình; Auto* không luôn map đúng SmolVLM2.
CANDIDATES: dict[str, str] = {
    "Qwen/Qwen2-VL-2B-Instruct": "Qwen2VLForConditionalGeneration",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLForConditionalGeneration",
    "HuggingFaceTB/SmolVLM2-2.2B-instruct": "SmolVLMForConditionalGeneration",
}

DEFAULT_FRAMES = [
    "data/duong_pho_1.jpg",
    "data/duong_pho_2.jpg",
    "data/duong_pho_3.jpg",
    "data/duong_pho_4.jpg",
]

# Prompt verifier rút gọn đúng format JSON mà parse_verifier_json yêu cầu.
VERIFIER_PROMPT = (
    "You are a surveillance-video verifier. You are given 4 consecutive frames "
    "from one candidate anomaly window. Answer with ONE JSON object only, no extra text:\n"
    '{"is_anomaly": true/false, "event_type": "<one of: Fighting|Accident|Explosion|'
    'Abuse|Robbery|Shooting|Stealing|Vandalism|Burglary|Arson|Normal>", '
    '"caption": "<one short sentence>", "confidence": <float 0-1>}'
)


def load_model(model_id: str, class_name: str):
    """Nạp VLM 4-bit qua bitsandbytes. Trả về (model, processor, load_seconds, load_vram_gb)."""
    import transformers
    from transformers import BitsAndBytesConfig

    cls = getattr(transformers, class_name)
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = cls.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    return model, processor, load_s, load_vram_gb


def one_window(model, processor, frame_paths: list[str], max_new_tokens: int) -> tuple[str, float]:
    """Một cửa sổ = 4 frame + prompt -> text đầu ra. Trả về (raw_text, giây)."""
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": VERIFIER_PROMPT}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode([gen], skip_special_tokens=True)[0]
    return raw.strip(), elapsed


def bench_model(model_id: str, class_name: str, frames: list[str], runs: int, max_new_tokens: int) -> dict:
    print(f"\n=== {model_id} ({class_name}) ===", flush=True)
    result: dict = {"model_id": model_id, "class": class_name, "ok": False}
    try:
        model, processor, load_s, load_vram_gb = load_model(model_id, class_name)
    except Exception as e:  # OOM khi nạp = số đo hợp lệ, không phải crash cả bench
        result["error"] = f"load failed: {type(e).__name__}: {e}"
        print(f"  LOAD THẤT BẠI: {result['error']}", flush=True)
        return result

    peak_after_load = torch.cuda.max_memory_allocated() / 1024**3
    times: list[float] = []
    sample_out = ""
    result["ok"] = True
    result["load_seconds"] = round(load_s, 1)
    result["load_vram_gb"] = round(load_vram_gb, 2)
    for i in range(runs):
        try:
            raw, elapsed = one_window(model, processor, frames, max_new_tokens)
        except Exception as e:
            result["error"] = f"generate failed on run {i}: {type(e).__name__}: {e}"
            print(f"  GENERATE THẤT BẠI (run {i}): {result['error']}", flush=True)
            break
        times.append(elapsed)
        if not sample_out:
            sample_out = raw
        print(f"  run {i + 1}/{runs}: {elapsed:.2f}s", flush=True)

    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    result["peak_vram_gb"] = round(peak_gb, 2)
    result["peak_vram_after_load_gb"] = round(peak_after_load, 2)
    if times:
        times_sorted = sorted(times)
        result["runs"] = [round(t, 2) for t in times]
        result["mean_s_per_window"] = round(sum(times) / len(times), 2)
        result["median_s_per_window"] = round(times_sorted[len(times_sorted) // 2], 2)
        result["output_tokens_cap"] = max_new_tokens
        result["sample_output"] = sample_out[:400]
    print(
        f"  -> peak VRAM {result['peak_vram_gb']} GB | load {load_s:.1f}s | "
        f"mean {result.get('mean_s_per_window', 'n/a')}s/window",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    # Console Windows mặc định cp1252: in thông báo tiếng Việt (thất bại) sẽ raise -> bật utf-8.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="A1: benchmark VLM 4-bit trên GPU thật")
    ap.add_argument("--model", action="append", default=None,
                    help="HF id để đo (lặp lại được); mặc định: 3 ứng viên trong CANDIDATES")
    ap.add_argument("--runs", type=int, default=3, help="số cửa sổ đo (mặc định 3)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--frames", nargs="*", default=None,
                    help="đường dẫn 4 frame; mặc định data/duong_pho_{1..4}.jpg")
    ap.add_argument("--out", default=None, help="ghi JSON kết quả vào file này")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Cần GPU thật cho benchmark A1 (torch.cuda.is_available() = False)")

    frames = args.frames or DEFAULT_FRAMES
    missing = [p for p in frames if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Thiếu frame: {missing}")
    if len(frames) != 4:
        print(f"Cảnh báo: {len(frames)} frame (A1 spec là 4/cửa sổ)")

    props = torch.cuda.get_device_properties(0)
    header = {
        "gpu": props.name,
        "gpu_total_vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": frames,
    }
    print(f"GPU: {header['gpu']} ({header['gpu_total_vram_gb']} GB), torch {header['torch']}", flush=True)

    model_ids = args.model or list(CANDIDATES)
    results = []
    for mid in model_ids:
        cls_name = CANDIDATES.get(mid)
        if cls_name is None:
            raise SystemExit(f"Không có class mặc định cho {mid!r}; thêm vào CANDIDATES trong script")
        results.append(bench_model(mid, cls_name, frames, args.runs, args.max_new_tokens))

    payload = {"header": header, "results": results}
    print("\n=== TỔNG KẾT ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
